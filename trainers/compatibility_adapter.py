from pathlib import Path
from string import Template
from typing import Any, Callable, Dict, List, Optional, Union

from gepa.core.adapter import GEPAAdapter, EvaluationBatch
from .extraction_utils import ATLASExtractionUtils
from .online_teaching_reward import OnlineTeachingReward
from .prompt_adapter import ATLASDataInst, ATLASTrajectory, ATLASRolloutOutput


class CompatibilityAdapter(GEPAAdapter[ATLASDataInst, ATLASTrajectory, ATLASRolloutOutput]):
    """Adapter for testing existing agents with ATLAS teaching framework."""

    def __init__(
        self,
        teacher_model: Union[str, Callable],
        user_agent: Callable,
        trace_storage_path: str = "traces/compatibility_traces.jsonl",
        generation_config: Optional[Dict[str, Any]] = None,
        max_litellm_workers: int = 10,
    ):
        self.teacher_model = teacher_model
        self.user_agent = user_agent
        self.trace_storage_path = Path(trace_storage_path)
        self.trace_storage_dir = self.trace_storage_path.parent / self.trace_storage_path.stem
        self.trace_storage_dir.mkdir(parents=True, exist_ok=True)
        self.eval_count = 0
        self.generation_config = generation_config or {}
        self.max_litellm_workers = max_litellm_workers

        if isinstance(teacher_model, str):
            import litellm
            self.teacher_model = lambda prompts: self._litellm_generate(litellm, teacher_model, prompts)

    def _safe_format(self, template_str: str, **kwargs) -> str:
        converted_template = template_str
        for key in kwargs:
            placeholder_pattern = '{' + key + '}'
            replacement_pattern = '$' + key
            converted_template = converted_template.replace(placeholder_pattern, replacement_pattern)
        template = Template(converted_template)
        return template.safe_substitute(**kwargs)

    def _litellm_generate(self, litellm, model: str, prompts: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(prompts, str):
            prompts = [prompts]
            single = True
        else:
            single = False

        messages_batch = [[{"role": "user", "content": p}] for p in prompts]
        responses = [
            resp.choices[0].message.content.strip()
            for resp in litellm.batch_completion(
                model=model,
                messages=messages_batch,
                max_workers=self.max_litellm_workers,
                temperature=self.generation_config.get('temperature', 0.7),
                max_tokens=self.generation_config.get('max_tokens', 2048),
                timeout=self.generation_config.get('timeout', 300),
            )
        ]
        return responses[0] if single else responses

    def evaluate(
        self,
        batch: List[ATLASDataInst],
        candidate: Dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[ATLASTrajectory, ATLASRolloutOutput]:

        self.eval_count += 1
        outputs: List[ATLASRolloutOutput] = []
        scores: List[float] = []
        trajectories: Optional[List[ATLASTrajectory]] = [] if capture_traces else None

        questions = [data_inst["question"] for data_inst in batch]
        ground_truths = [data_inst["ground_truth"] for data_inst in batch]

        teacher_adaptive_template = candidate.get("teacher_adaptive_template",
            "You are an expert teacher. The student gave this response: {baseline_response}\n\n"
            "To the question: {question}\n\n"
            "Provide focused teaching to help them improve. Wrap teaching in <teaching> tags.")

        print(f"[Compatibility] Getting baseline responses from user agent...")
        baseline_responses = self.user_agent(questions)
        if not isinstance(baseline_responses, list):
            baseline_responses = [baseline_responses]

        teacher_prompts = [
            self._safe_format(teacher_adaptive_template,
                question=q,
                baseline_response=br)
            for q, br in zip(questions, baseline_responses)
        ]

        print(f"[Compatibility] Generating teaching based on baseline responses...")
        teacher_responses = self.teacher_model(teacher_prompts)
        if not isinstance(teacher_responses, list):
            teacher_responses = [teacher_responses]

        teaching_contents = [
            ATLASExtractionUtils.extract_teaching_content(tr)
            for tr in teacher_responses
        ]

        enhanced_prompts = [
            f"Context: {teaching}\n\nQuestion: {question}"
            for teaching, question in zip(teaching_contents, questions)
        ]

        print(f"[Compatibility] Getting enhanced responses with teaching context...")
        enhanced_responses = self.user_agent(enhanced_prompts)
        if not isinstance(enhanced_responses, list):
            enhanced_responses = [enhanced_responses]

        baseline_solutions = ATLASExtractionUtils.extract_solutions(baseline_responses)
        enhanced_solutions = ATLASExtractionUtils.extract_solutions(enhanced_responses)

        class SimpleTokenizer:
            def encode(self, text):
                return text.split()

        reward_calculator = OnlineTeachingReward(tokenizer=SimpleTokenizer())
        rewards = reward_calculator(
            prompts=questions,
            completions=teacher_responses,
            baseline_solutions=baseline_solutions,
            solutions=enhanced_solutions,
            ground_truths=ground_truths,
        )

        for i in range(len(batch)):
            output = {
                "teacher_response": teacher_responses[i],
                "student_with_teaching": enhanced_responses[i],
                "student_baseline": baseline_responses[i],
                "reward": rewards[i],
            }
            outputs.append(output)
            scores.append(rewards[i])

            if capture_traces:
                trajectory = {
                    "question": questions[i],
                    "student_approach": baseline_responses[i],
                    "teacher_response": teacher_responses[i],
                    "student_baseline": baseline_responses[i],
                    "student_with_teaching": enhanced_responses[i],
                    "ground_truth": ground_truths[i],
                    "reward": rewards[i],
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

            items.append({
                "OPTIMIZATION_TARGET": "teacher_adaptive_template",
                "GOAL": "Generate better teaching based on user agent's baseline response teacher needs to improve student's perfromance according to the task given to the student teacher's focus should be on how teach this student so that if can comlete the request or question without overthinking"
            })

            for trajectory, score in zip(eval_batch.trajectories, eval_batch.scores):
                baseline_solution = ATLASExtractionUtils.extract_solution(trajectory["student_baseline"])
                enhanced_solution = ATLASExtractionUtils.extract_solution(trajectory["student_with_teaching"])

                item = {
                    "Inputs": {
                        "question": trajectory["question"],
                        "baseline_response": trajectory["student_baseline"],
                    },
                    "Teaching": {
                        "teacher_response": trajectory["teacher_response"],
                    },
                    "Outputs": {
                        "enhanced_solution": enhanced_solution,
                        "baseline_solution": baseline_solution,
                    },
                    "Performance": {
                        "improved": score > 0.5,
                        "score": score,
                        "ground_truth": trajectory["ground_truth"],
                    }
                }
                items.append(item)

            reflective_data[component] = items

        return reflective_data