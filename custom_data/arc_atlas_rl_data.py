from datasets import load_dataset
from typing import Dict, Any, Optional
from transformers import PreTrainedTokenizer


def get_arc_atlas_rl_dataset(
    tokenizer: PreTrainedTokenizer,
    dataset_id_or_path: str = "Arc-Intelligence/Arc-ATLAS-Teach-v0",
    dataset_split: str = "rl",
    dataset_max_samples: Optional[int] = None,
    eval_split_ratio: float = 0.1,
) -> Dict[str, Any]:
    
    dataset = load_dataset(dataset_id_or_path, data_files="training/rl.jsonl", split="train")
    
    if dataset_max_samples is not None:
        dataset = dataset.select(range(min(dataset_max_samples, len(dataset))))
    
    def format_example(example):
        return {
            "prompt": example["prompt"],
            "ground_truth": example["ground_truth"],
        }
    
    formatted_dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    
    dataset_dict = formatted_dataset.train_test_split(test_size=eval_split_ratio, seed=42)
    
    return {
        "train_dataset": dataset_dict["train"],
        "eval_dataset": dataset_dict["test"],
    }