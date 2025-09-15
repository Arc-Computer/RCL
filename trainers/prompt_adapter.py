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
        all_prompts: Optional[Dict[str, str]] = None,
        generation_config: Optional[Dict[str, any]] = None,
        max_litellm_workers: int = 10,
        use_vllm_client: bool = False,
        vllm_host: Optional[str] = None,
        vllm_port: Optional[int] = None,
    ):
        self.reward_function = reward_function
        self.trace_storage_path = Path(trace_storage_path)
        self.trace_storage_dir = self.trace_storage_path.parent / self.trace_storage_path.stem
        self.trace_storage_dir.mkdir(parents=True, exist_ok=True)
        self.iteration_count = 0
        self.eval_count = 0
        self.max_litellm_workers = max_litellm_workers
        self.use_vllm_client = use_vllm_client
        self.all_prompts = all_prompts or {}
        self.generation_config = generation_config or {}
        
        if isinstance(teacher_model, str):
            if use_vllm_client and vllm_host and vllm_port:
                from trainers.vllm_client import VLLMClient
                vllm_client = VLLMClient(host=vllm_host, server_port=vllm_port)
                self.teacher_model = lambda prompts: vllm_client.generate(
                    prompts=prompts if isinstance(prompts, list) else [prompts],
                    n=1, 
                    temperature=self.generation_config['temperature'], 
                    max_tokens=self.generation_config['max_tokens']
                )[0] if isinstance(prompts, str) else [ids[0] for ids in vllm_client.generate(
                    prompts=prompts, 
                    n=1, 
                    temperature=self.generation_config['temperature'], 
                    max_tokens=self.generation_config['max_tokens']
                )]
            else:
                import litellm
                self.teacher_model = lambda prompts: self._litellm_generate(litellm, teacher_model, prompts)
        else:
            self.teacher_model = teacher_model
        
        self.student_model_str = student_model if isinstance(student_model, str) else None
        
        if isinstance(student_model, str):
            if use_vllm_client and vllm_host and vllm_port:
                from trainers.vllm_client import VLLMClient
                vllm_client = VLLMClient(host=vllm_host, server_port=vllm_port) if not hasattr(self, 'vllm_client') else self.vllm_client
                self.student_model = lambda prompts: vllm_client.generate(
                    prompts=prompts if isinstance(prompts, list) else [prompts],
                    n=1, 
                    temperature=self.generation_config['temperature'], 
                    max_tokens=self.generation_config['max_tokens']
                )[0] if isinstance(prompts, str) else [ids[0] for ids in vllm_client.generate(
                    prompts=prompts, 
                    n=1, 
                    temperature=self.generation_config['temperature'], 
                    max_tokens=self.generation_config['max_tokens']
                )]
            else:
                import litellm
                self.student_model = lambda prompts: self._litellm_generate(litellm, student_model, prompts)
        else:
            self.student_model = student_model
    
    def _generate_with_student(self, prompts, max_tokens=None):
        if max_tokens is None:
            max_tokens = self.generation_config['max_tokens']
        if self.student_model_str:
            import litellm
            return self._litellm_generate(litellm, self.student_model_str, prompts, max_tokens=max_tokens)
        else:
            return self.student_model(prompts)
    
    def _litellm_generate(
        self,
        litellm,
        model: str,
        prompts: Union[str, List[str]],
        max_tokens: int = None,
        temperature: float = None,
    ) -> Union[str, List[str]]:
        if isinstance(prompts, str):
            prompts = [prompts]
            single = True
        else:
            single = False
        
        if max_tokens is None:
            max_tokens = self.generation_config['max_tokens']
        if temperature is None:
            temperature = self.generation_config['temperature']
        
        print(f"[LiteLLM] Generating for {len(prompts)} prompts with model: {model[:50]}...")
        messages_batch = [[{"role": "user", "content": p}] for p in prompts]
        
        try:
            if model.startswith("https://") and "huggingface.cloud" in model:
                import requests
                import os
                import time
                from concurrent.futures import ThreadPoolExecutor, as_completed

                def make_hf_request(idx, msg):
                    payload = {
                        "inputs": msg[0]["content"],
                        "parameters": {
                            "max_new_tokens": max_tokens,
                            "temperature": temperature,
                            "return_full_text": False
                        }
                    }
                    headers = {"Content-Type": "application/json"}

                    retries = 3
                    for attempt in range(retries):
                        try:
                            response = requests.post(model, json=payload, headers=headers, timeout=120)
                            break
                        except requests.exceptions.Timeout:
                            if attempt < retries - 1:
                                time.sleep(5)
                            else:
                                response = None

                    if response and response.status_code == 200:
                        result = response.json()
                        if isinstance(result, list) and len(result) > 0:
                            return idx, result[0].get("generated_text", "")
                        elif isinstance(result, dict) and "generated_text" in result:
                            return idx, result["generated_text"]
                        else:
                            return idx, str(result)
                    elif response:
                        return idx, f"Error: {response.status_code}"
                    else:
                        return idx, "Error: Timeout"

                print(f"[HF] Processing {len(messages_batch)} requests in parallel with {self.max_litellm_workers} workers...")
                responses = [None] * len(messages_batch)

                with ThreadPoolExecutor(max_workers=self.max_litellm_workers) as executor:
                    futures = {executor.submit(make_hf_request, i, messages): i
                              for i, messages in enumerate(messages_batch)}

                    completed = 0
                    for future in as_completed(futures):
                        idx, result = future.result()
                        responses[idx] = result
                        completed += 1
                        if completed % 50 == 0 or completed == len(messages_batch):
                            print(f"[HF] Completed {completed}/{len(messages_batch)} requests")
            else:
                timeout = self.generation_config.get('timeout', 300)
                request_timeout = self.generation_config.get('request_timeout', 300)
                responses = [
                    resp.choices[0].message.content.strip()
                    for resp in litellm.batch_completion(
                        model=model,
                        messages=messages_batch,
                        max_workers=self.max_litellm_workers,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=timeout,
                        request_timeout=request_timeout,
                    )
                ]
            return responses[0] if single else responses
        except Exception as e:
            print(f"Generation error: {str(e)}")
            error_responses = [f"Error: {str(e)}"] * len(prompts)
            return error_responses[0] if single else error_responses
    
    def evaluate(
        self,
        batch: List[ATLASDataInst],
        candidate: Dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[ATLASTrajectory, ATLASRolloutOutput]:
        
        print(f"\n[Adapter] Evaluating batch of {len(batch)} examples...")
        self.eval_count += 1
        
        self.store_prompt_evolution(candidate)
        
        outputs: List[ATLASRolloutOutput] = []
        scores: List[float] = []
        trajectories: Optional[List[ATLASTrajectory]] = [] if capture_traces else None
        
        questions = [data_inst["question"] for data_inst in batch]
        ground_truths = [data_inst["ground_truth"] for data_inst in batch]
        
        student_diagnostic_template = candidate.get("student_diagnostic_template") or self.all_prompts.get("student_diagnostic_template")
        teacher_adaptive_template = candidate.get("teacher_adaptive_template") or self.all_prompts.get("teacher_adaptive_template")
        student_baseline_template = self.all_prompts.get("student_baseline_template", candidate.get("student_baseline_template", ""))
        student_with_teaching_template = candidate.get("student_with_teaching_template") or self.all_prompts.get("student_with_teaching_template")
        
        baseline_prompts = [
            student_baseline_template.format(question=q) for q in questions
        ]
        print(f"[Adapter] Getting baseline solutions from student...")
        baseline_completions = self.student_model(baseline_prompts)
        if not isinstance(baseline_completions, list):
            baseline_completions = [baseline_completions]
        print(f"[Adapter] Got {len(baseline_completions)} baseline solutions")
        
        approach_prompts = [
            student_diagnostic_template.format(question=q) for q in questions
        ]
        diagnostic_max_tokens = self.generation_config['diagnostic_max_tokens']
        print(f"[Adapter] Getting student diagnostic approaches (max {diagnostic_max_tokens} tokens)...")
        student_approaches = self._generate_with_student(approach_prompts, max_tokens=diagnostic_max_tokens)
        if not isinstance(student_approaches, list):
            student_approaches = [student_approaches]
        print(f"[Adapter] Got {len(student_approaches)} approaches")
        
        teacher_prompts = [
            teacher_adaptive_template.format(question=q, approach=a)
            for q, a in zip(questions, student_approaches)
        ]
        print(f"[Adapter] Getting teacher responses...")
        teacher_completions = self.teacher_model(teacher_prompts)
        if not isinstance(teacher_completions, list):
            teacher_completions = [teacher_completions]
        print(f"[Adapter] Got {len(teacher_completions)} teacher responses")
        
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
            from .online_teaching_reward import OnlineTeachingReward
            
            class SimpleTokenizer:
                def encode(self, text):
                    return text.split()
            
            default_reward = OnlineTeachingReward(tokenizer=SimpleTokenizer())
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
                        "student_with_teaching_solution": teaching_solution,
                    },
                    "Comparison": {
                        "baseline_solution": baseline_solution,
                        "baseline_correct": ATLASExtractionUtils.check_correctness(baseline_solution, trajectory["ground_truth"]),
                        "teaching_correct": ATLASExtractionUtils.check_correctness(teaching_solution, trajectory["ground_truth"]),
                    },
                    "Feedback": f"Expected answer: {trajectory['ground_truth']}. Score: {score}",
                }
                
                items.append(item)
            
            reflective_data[component] = items
        
        return reflective_data
    
    def store_prompt_evolution(self, candidate: Dict[str, str]):
        import time
        prompt_dir = self.trace_storage_dir / "prompt_evolution"
        prompt_dir.mkdir(exist_ok=True)
        
        prompt_data = {
            "eval_count": self.eval_count,
            "timestamp": time.time(),
            "prompts": candidate
        }
        
        filename = f"prompts_eval_{self.eval_count:04d}.json"
        filepath = prompt_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(prompt_data, f, indent=2)
    
    def store_trace(self, trajectory: ATLASTrajectory):
        import time
        trace_data = {
            "eval_count": self.eval_count,
            "timestamp": time.time(),
            "trajectory": trajectory
        }
        
        filename = f"eval_{self.eval_count:04d}.jsonl"
        filepath = self.trace_storage_dir / filename
        
        with open(filepath, "a") as f:
            f.write(json.dumps(trace_data) + "\n")