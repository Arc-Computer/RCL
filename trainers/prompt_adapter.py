import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypedDict, Union

from gepa.core.adapter import GEPAAdapter, EvaluationBatch
from .extraction_utils import ATLASExtractionUtils


class ATLASDataInst(TypedDict):
    question: str
    ground_truth: str
    additional_context: Optional[Dict[str, str]]


class ATLASTrajectory(TypedDict):
    question: str
    student_approach: str
    teacher_response: str
    student_baseline: str
    student_with_teaching: str
    ground_truth: str
    reward: float


class ATLASRolloutOutput(TypedDict):
    teacher_response: str
    student_with_teaching: str
    student_baseline: str
    reward: float


class ATLASGEPAAdapter(GEPAAdapter[ATLASDataInst, ATLASTrajectory, ATLASRolloutOutput]):
    
    def __init__(
        self,
        teacher_model: Union[str, Callable],
        student_model: Union[str, Callable],
        reward_function: Optional[Callable] = None,
        trace_storage_path: str = "traces/gepa_traces.jsonl",
        max_litellm_workers: int = 10,
        use_vllm_client: bool = False,
        vllm_host: Optional[str] = None,
        vllm_port: Optional[int] = None,
    ):
        self.reward_function = reward_function
        self.trace_storage_path = Path(trace_storage_path)
        self.trace_storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_litellm_workers = max_litellm_workers
        self.use_vllm_client = use_vllm_client
        
        if isinstance(teacher_model, str):
            if use_vllm_client and vllm_host and vllm_port:
                from trainers.vllm_client import VLLMClient
                vllm_client = VLLMClient(host=vllm_host, server_port=vllm_port)
                self.teacher_model = lambda prompts: vllm_client.generate(
                    prompts=prompts if isinstance(prompts, list) else [prompts],
                    n=1, temperature=0.7, max_tokens=4096
                )[0] if isinstance(prompts, str) else [ids[0] for ids in vllm_client.generate(
                    prompts=prompts, n=1, temperature=0.7, max_tokens=4096
                )]
            else:
                import litellm
                self.teacher_model = lambda prompts: self._litellm_generate(litellm, teacher_model, prompts)
        else:
            self.teacher_model = teacher_model
        
        if isinstance(student_model, str):
            if use_vllm_client and vllm_host and vllm_port:
                from trainers.vllm_client import VLLMClient
                vllm_client = VLLMClient(host=vllm_host, server_port=vllm_port) if not hasattr(self, 'vllm_client') else self.vllm_client
                self.student_model = lambda prompts: vllm_client.generate(
                    prompts=prompts if isinstance(prompts, list) else [prompts],
                    n=1, temperature=0.7, max_tokens=4096
                )[0] if isinstance(prompts, str) else [ids[0] for ids in vllm_client.generate(
                    prompts=prompts, n=1, temperature=0.7, max_tokens=4096
                )]
            else:
                import litellm
                self.student_model = lambda prompts: self._litellm_generate(litellm, student_model, prompts)
        else:
            self.student_model = student_model
    
    def _litellm_generate(
        self,
        litellm,
        model: str,
        prompts: Union[str, List[str]],
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> Union[str, List[str]]:
        if isinstance(prompts, str):
            prompts = [prompts]
            single = True
        else:
            single = False
        
        messages_batch = [[{"role": "user", "content": p}] for p in prompts]
        
        try:
            responses = [
                resp.choices[0].message.content.strip()
                for resp in litellm.batch_completion(
                    model=model,
                    messages=messages_batch,
                    max_workers=self.max_litellm_workers,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            ]
            return responses[0] if single else responses
        except Exception as e:
            error_responses = [f"Error: {str(e)}"] * len(prompts)
            return error_responses[0] if single else error_responses
    
    def evaluate(
        self,
        batch: List[ATLASDataInst],
        candidate: Dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[ATLASTrajectory, ATLASRolloutOutput]:
        
        outputs: List[ATLASRolloutOutput] = []
        scores: List[float] = []
        trajectories: Optional[List[ATLASTrajectory]] = [] if capture_traces else None
        
        questions = [data_inst["question"] for data_inst in batch]
        ground_truths = [data_inst["ground_truth"] for data_inst in batch]
        
        student_diagnostic_template = candidate["student_diagnostic_template"]
        teacher_adaptive_template = candidate["teacher_adaptive_template"]
        student_baseline_template = candidate["student_baseline_template"]
        student_with_teaching_template = candidate["student_with_teaching_template"]
        
        baseline_prompts = [
            student_baseline_template.format(question=q) for q in questions
        ]
        baseline_completions = self.student_model(baseline_prompts)
        if not isinstance(baseline_completions, list):
            baseline_completions = [baseline_completions]
        
        approach_prompts = [
            student_diagnostic_template.format(question=q) for q in questions
        ]
        student_approaches = self.student_model(approach_prompts)
        if not isinstance(student_approaches, list):
            student_approaches = [student_approaches]
        
        teacher_prompts = [
            teacher_adaptive_template.format(question=q, approach=a)
            for q, a in zip(questions, student_approaches)
        ]
        teacher_completions = self.teacher_model(teacher_prompts)
        if not isinstance(teacher_completions, list):
            teacher_completions = [teacher_completions]
        
        teaching_contents = [
            ATLASExtractionUtils.extract_teaching_content(tc) 
            for tc in teacher_completions
        ]
        
        student_with_teaching_prompts = [
            student_with_teaching_template.format(question=q, teaching=t)
            for q, t in zip(questions, teaching_contents)
        ]
        student_with_teaching_completions = self.student_model(student_with_teaching_prompts)
        if not isinstance(student_with_teaching_completions, list):
            student_with_teaching_completions = [student_with_teaching_completions]
        
        baseline_solutions = ATLASExtractionUtils.extract_solutions(baseline_completions)
        student_solutions = ATLASExtractionUtils.extract_solutions(student_with_teaching_completions)
        
        if self.reward_function:
            rewards = self.reward_function(
                prompts=questions,
                completions=teacher_completions,
                baseline_solutions=baseline_solutions,
                solutions=student_solutions,
                ground_truths=ground_truths,
            )
        else:
            from .teacher_rewards import AdaptiveTeachingReward
            
            class SimpleTokenizer:
                def encode(self, text):
                    return text.split()
            
            default_reward = AdaptiveTeachingReward(tokenizer=SimpleTokenizer())
            rewards = default_reward(
                prompts=questions,
                completions=teacher_completions,
                baseline_solutions=baseline_solutions,
                solutions=student_solutions,
                ground_truths=ground_truths,
            )
        
        for i in range(len(batch)):
            try:
                output = {
                    "teacher_response": teacher_completions[i] if i < len(teacher_completions) else "",
                    "student_with_teaching": student_with_teaching_completions[i] if i < len(student_with_teaching_completions) else "",
                    "student_baseline": baseline_completions[i] if i < len(baseline_completions) else "",
                    "reward": rewards[i] if i < len(rewards) else 0.0,
                }
                outputs.append(output)
                scores.append(rewards[i] if i < len(rewards) else 0.0)
                
                if capture_traces:
                    trajectory = {
                        "question": questions[i],
                        "student_approach": student_approaches[i] if i < len(student_approaches) else "",
                        "teacher_response": teacher_completions[i] if i < len(teacher_completions) else "",
                        "student_baseline": baseline_completions[i] if i < len(baseline_completions) else "",
                        "student_with_teaching": student_with_teaching_completions[i] if i < len(student_with_teaching_completions) else "",
                        "ground_truth": ground_truths[i],
                        "reward": rewards[i] if i < len(rewards) else 0.0,
                    }
                    trajectories.append(trajectory)
                    self.store_trace(trajectory)
            except Exception as e:
                output = {
                    "teacher_response": f"Error: {str(e)}",
                    "student_with_teaching": "",
                    "student_baseline": "",
                    "reward": 0.0,
                }
                outputs.append(output)
                scores.append(0.0)
                
                if capture_traces:
                    trajectory = {
                        "question": questions[i],
                        "student_approach": f"Error: {str(e)}",
                        "teacher_response": f"Error: {str(e)}",
                        "student_baseline": "",
                        "student_with_teaching": "",
                        "ground_truth": ground_truths[i],
                        "reward": 0.0,
                    }
                    trajectories.append(trajectory)
        
        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )
    
    def make_reflective_dataset(
        self,
        candidate: Dict[str, str],
        eval_batch: EvaluationBatch[ATLASTrajectory, ATLASRolloutOutput],
        components_to_update: List[str],
    ) -> Dict[str, List[Dict[str, Any]]]:
        
        reflective_data = {}
        
        for component in components_to_update:
            items = []
            
            for trajectory, score in zip(eval_batch.trajectories, eval_batch.scores):
                baseline_solution = ATLASExtractionUtils.extract_solution(trajectory["student_baseline"])
                teaching_solution = ATLASExtractionUtils.extract_solution(trajectory["student_with_teaching"])
                
                item = {
                    "Inputs": {
                        "question": trajectory["question"],
                        "student_approach": trajectory["student_approach"],
                    },
                    "Generated Outputs": {
                        "teacher_response": trajectory["teacher_response"],
                        "student_baseline_solution": baseline_solution,
                        "student_with_teaching_solution": teaching_solution,
                    },
                    "Feedback": f"Expected answer: {trajectory['ground_truth']}. Score: {score}",
                }
                
                items.append(item)
            
            reflective_data[component] = items
        
        return reflective_data
    
    def store_trace(self, trajectory: ATLASTrajectory):
        with open(self.trace_storage_path, "a") as f:
            f.write(json.dumps(trajectory) + "\n")