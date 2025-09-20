import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable

import gepa
from trainers.prompt_adapter import ATLASGEPAAdapter, ATLASDataInst
from datasets import load_dataset


def load_arc_atlas_dataset_from_hf() -> List[ATLASDataInst]:
    dataset = load_dataset("Arc-Intelligence/Arc-ATLAS-Teach-v0", data_files="training/rl.jsonl", split="train")
    result = []
    for example in dataset:
        result.append({
            "question": example["prompt"],
            "ground_truth": example["ground_truth"],
            "additional_context": {},
        })
    return result


def load_gsm8k_zh_dataset() -> List[ATLASDataInst]:
    dataset = load_dataset("meta-math/GSM8K_zh", split="train")
    result = []
    for example in dataset:
        result.append({
            "question": example["question"],
            "ground_truth": example["answer_only"],
            "additional_context": {},
        })
    return result


def load_dataset_from_jsonl(path: str, column_config: Optional[Dict[str, str]] = None) -> List[ATLASDataInst]:
    dataset = []
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if column_config:
                question_col = column_config.get('question', 'question')
                answer_col = column_config.get('answer', 'ground_truth')
                context_col = column_config.get('context')

                dataset.append({
                    "question": data.get(question_col, ""),
                    "ground_truth": data.get(answer_col, ""),
                    "additional_context": data.get(context_col, {}) if context_col else {},
                })
            else:
                dataset.append({
                    "question": data.get("question", data.get("prompt", "")),
                    "ground_truth": data.get("ground_truth", data.get("answer", "")),
                    "additional_context": data.get("additional_context", {}),
                })
    return dataset




def run_gepa_optimization(
    teacher_model: Union[str, Callable],
    student_model: Union[str, Callable],
    trainset: List[ATLASDataInst],
    valset: Optional[List[ATLASDataInst]],
    max_metric_calls: int,
    reflection_lm: str,
    trace_storage_path: str,
    seed_prompts: Dict[str, str],
    all_prompts: Dict[str, str],
    gepa_config: Dict[str, Any],
    generation_config: Dict[str, Any],
    wandb_config: Optional[Dict[str, Any]] = None,
    use_vllm_client: bool = False,
    vllm_host: Optional[str] = None,
    vllm_port: Optional[int] = None,
    compatibility_mode: bool = False,
    user_agent: Optional[Callable] = None,
    reflection_instructions: Optional[Dict[str, str]] = None,
    evaluation_config: Optional[Dict[str, Any]] = None,
    optimization_targets: Optional[Dict[str, Any]] = None,
) -> Dict:

    if compatibility_mode and user_agent:
        from trainers.compatibility_adapter import CompatibilityAdapter
        adapter = CompatibilityAdapter(
            teacher_model=teacher_model,
            user_agent=user_agent,
            trace_storage_path=trace_storage_path,
            generation_config=generation_config,
            max_litellm_workers=generation_config.get('max_litellm_workers', 10),
            reflection_instructions=reflection_instructions,
            evaluation_config=evaluation_config,
            optimization_targets=optimization_targets,
        )
    else:
        adapter = ATLASGEPAAdapter(
            teacher_model=teacher_model,
            student_model=student_model,
            reward_function=None,
            trace_storage_path=trace_storage_path,
            all_prompts=all_prompts,
            generation_config=generation_config,
            max_litellm_workers=generation_config.get('max_litellm_workers', 10),
            use_vllm_client=use_vllm_client,
            vllm_host=vllm_host,
            vllm_port=vllm_port,
        )

    import litellm
    def reflection_lm_func(prompt: str) -> str:
        try:
            print(f"[DEBUG] Calling reflection LM: {reflection_lm}")
            print(f"[DEBUG] Prompt length: {len(prompt)} chars")

            response = litellm.completion(
                model=reflection_lm,
                messages=[{"role": "user", "content": prompt}],
                timeout=generation_config.get('timeout', 600),
                request_timeout=generation_config.get('request_timeout', 600),
                max_tokens=generation_config.get('reflection_max_tokens', 32768),
                temperature=generation_config.get('temperature', 0.7)
            )
            if response and response.choices and len(response.choices) > 0 and response.choices[0].message:
                content = response.choices[0].message.content
                if content is None:
                    print("[ERROR] Reflection LM returned None content")
                    print(f"[DEBUG] Response object: {response}")
                    raise ValueError("Reflection LM returned None content")

                print(f"[DEBUG] Reflection LM response length: {len(content)} chars")
                if "```" in content:
                    print("[DEBUG] Response contains ``` blocks")
                else:
                    print("[WARNING] Response missing ``` blocks - GEPA may fail to extract")

                return content
            else:
                raise ValueError("Invalid response structure from reflection LM")

        except Exception as e:
            print(f"[ERROR] Reflection LM failed: {e} (Model: {reflection_lm})")
            raise

    result = gepa.optimize(
        seed_candidate=seed_prompts,
        trainset=trainset,
        valset=valset if valset else trainset,
        adapter=adapter,
        reflection_lm=reflection_lm_func,
        max_metric_calls=max_metric_calls,
        **gepa_config
    )
    
    return result


def save_optimized_prompts(result, output_path: str, initial_score: float = None):
    output_data = {
        "best_candidate": result.best_candidate,
        "best_score": float(result.best_score) if hasattr(result, 'best_score') else None,
        "initial_score": initial_score,
        "improvement": float(result.best_score - initial_score) if initial_score and hasattr(result, 'best_score') else None,
        "improvement_percentage": float((result.best_score - initial_score) / initial_score * 100) if initial_score and hasattr(result, 'best_score') and initial_score > 0 else None,
        "pareto_frontier": result.pareto_frontier if hasattr(result, 'pareto_frontier') else None,
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nOptimized prompts saved to: {output_path}")
    if output_data.get('best_score') is not None:
        print(f"Best score achieved: {output_data['best_score']:.4f}")
    if output_data.get('initial_score') is not None:
        print(f"Initial score: {output_data['initial_score']:.4f}")
    if output_data.get('improvement_percentage') is not None:
        print(f"Performance gain: {output_data['improvement_percentage']:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Optimize ATLAS teaching prompts using reflective evolution")
    
    parser.add_argument(
        "--trainset",
        type=str,
        required=True,
        help="Path to training dataset (JSONL format)",
    )
    parser.add_argument(
        "--valset",
        type=str,
        default=None,
        help="Path to validation dataset (JSONL format)",
    )
    parser.add_argument(
        "--student-model",
        type=str,
        required=True,
        help="Student model",
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        required=True,
        help="Teacher model",
    )
    parser.add_argument(
        "--reflection-lm",
        type=str,
        default=None,
        help="Language model for GEPA reflection (defaults to teacher model)",
    )
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=150,
        help="Maximum number of metric evaluations",
    )
    parser.add_argument(
        "--trace-storage",
        type=str,
        default="traces/gepa_traces.jsonl",
        help="Path to store execution traces",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="optimized_prompts.json",
        help="Path to save optimized prompts",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/optimize/default.yaml",
        help="Path to configuration file with seed prompts",
    )
    parser.add_argument(
        "--use-vllm-client",
        action="store_true",
        help="Use vLLM client for generation",
    )
    parser.add_argument(
        "--vllm-host",
        type=str,
        default=None,
        help="vLLM server host (required if --use-vllm-client)",
    )
    parser.add_argument(
        "--vllm-port",
        type=int,
        default=None,
        help="vLLM server port (required if --use-vllm-client)",
    )
    
    args = parser.parse_args()
    
    if args.use_vllm_client and (not args.vllm_host or not args.vllm_port):
        parser.error("--vllm-host and --vllm-port are required when using --use-vllm-client")
    
    print("Loading configuration...")
    import yaml
    from pathlib import Path

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if 'defaults' in config:
        base_configs = config.get('defaults', [])
        base_config = {}
        config_dir = Path(args.config).parent

        for base_name in base_configs:
            base_path = config_dir / f"{base_name}.yaml"
            if base_path.exists():
                with open(base_path, 'r') as f:
                    base = yaml.safe_load(f)
                    base_config.update(base)

        base_config.update(config)
        config = base_config
    
    compatibility_mode = config.get('compatibility_mode', False)

    reflection_lm = args.reflection_lm or config.get('reflection_lm') or config.get('teacher_model') or args.teacher_model

    seed_prompts = config.get('seed_prompts', {})
    if not seed_prompts and not compatibility_mode:
        parser.error(f"No seed_prompts found in config file: {args.config}")
    
    fixed_prompts = config.get('fixed_prompts', {})
    all_prompts = {**seed_prompts, **fixed_prompts}
    
    gepa_config = config.get('gepa_config', {})
    wandb_config = config.get('wandb', {})
    generation_config = config.get('generation_config', {})
    generation_config['max_litellm_workers'] = config.get('max_litellm_workers', 10)
    reflection_instructions = config.get('reflection_instructions') or {}
    evaluation_config = config.get('evaluation') or {}
    optimization_targets = config.get('optimization_targets') or {}
    
    print("Loading datasets...")
    data_config = config.get('data', {})
    column_config = data_config.get('columns')

    if args.trainset == "arc-atlas-rl":
        trainset = load_arc_atlas_dataset_from_hf()
        valset = None
    elif args.trainset == "gsm8k-zh":
        trainset = load_gsm8k_zh_dataset()
        valset = None
    else:
        trainset = load_dataset_from_jsonl(args.trainset, column_config)
        valset = load_dataset_from_jsonl(args.valset, column_config) if args.valset else None

    max_examples = config.get('max_examples')
    if max_examples:
        trainset = trainset[:max_examples]
        if valset:
            valset = valset[:max_examples]
        print(f"Limited dataset to {max_examples} examples")

    print(f"Loaded {len(trainset)} training examples")
    
    if valset:
        print(f"Loaded {len(valset)} validation examples")
    
    if config.get('teacher_wrapper'):
        from wrappers import load_wrapper
        teacher_model = load_wrapper(
            config['teacher_wrapper']['type'],
            config['teacher_wrapper']['config']
        )
    else:
        teacher_model = args.teacher_model

    if config.get('student_wrapper'):
        from wrappers import load_wrapper
        student_model = load_wrapper(
            config['student_wrapper']['type'],
            config['student_wrapper']['config']
        )
    else:
        student_model = args.student_model

    user_agent = None

    if compatibility_mode:
        print("\n=== COMPATIBILITY MODE ===")
        print("Testing existing agent with ATLAS teaching")

        agent_type = config.get('agent_type')
        agent_config = config.get('agent_config', {})

        if agent_type:
            from wrappers import load_wrapper
            user_agent = load_wrapper(agent_type, agent_config)
            student_model = None
        elif config.get('user_agent'):
            from wrappers import load_wrapper
            user_agent = load_wrapper(
                config['user_agent']['type'],
                config['user_agent']['config']
            )
            student_model = None
        else:
            raise ValueError("Compatibility mode requires agent_type or user_agent configuration")

        if not seed_prompts:
            seed_prompts = {
                "teacher_adaptive_template":
                    "You are an expert teacher. The student gave this response: {baseline_response}\n\n"
                    "To the question: {question}\n\n"
                    "Provide focused teaching to help them improve. Wrap teaching in <teaching> tags."
            }

    print(f"\nTeacher model: {args.teacher_model}")
    if not compatibility_mode:
        print(f"Student model: {args.student_model}")
    else:
        print(f"User agent: Configured via {config['user_agent']['type']}")
    print(f"Reflection LM: {reflection_lm}")
    print(f"Max metric calls: {args.max_metric_calls}")
    print(f"Trace storage: {args.trace_storage}")

    if args.use_vllm_client:
        print(f"Using vLLM client at {args.vllm_host}:{args.vllm_port}")

    print("\nStarting GEPA optimization...")

    result = run_gepa_optimization(
        teacher_model=teacher_model,
        student_model=student_model,
        trainset=trainset,
        valset=valset,
        max_metric_calls=args.max_metric_calls,
        reflection_lm=reflection_lm,
        trace_storage_path=args.trace_storage,
        seed_prompts=seed_prompts,
        all_prompts=all_prompts,
        gepa_config=gepa_config,
        generation_config=generation_config,
        wandb_config=wandb_config,
        use_vllm_client=args.use_vllm_client,
        vllm_host=args.vllm_host,
        vllm_port=args.vllm_port,
        compatibility_mode=compatibility_mode,
        user_agent=user_agent,
        reflection_instructions=reflection_instructions,
        evaluation_config=evaluation_config,
        optimization_targets=optimization_targets,
    )
    
    initial_score = result.val_aggregate_scores[0] if result.val_aggregate_scores else None
    save_optimized_prompts(result, args.output, initial_score=initial_score)
    
    print("\n=== Optimized Templates ===")
    for key, value in result.best_candidate.items():
        print(f"\n{key}:")
        print(value)


if __name__ == "__main__":
    main()