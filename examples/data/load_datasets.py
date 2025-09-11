"""
Dataset loading utilities for ATLAS examples.

Loads real datasets from HuggingFace for demonstration.
"""

from datasets import load_dataset
import json
import random
from typing import List, Dict, Any, Optional


def load_atlas_teach_dataset(split: str = "train", num_samples: Optional[int] = 20) -> List[Dict[str, Any]]:
    """
    Load Arc-Intelligence/Arc-ATLAS-Teach-v0 dataset.
    
    Args:
        split: Dataset split to load ("train", "test", "validation")
        num_samples: Number of samples to load (None for all)
        
    Returns:
        List of problem dictionaries
    """
    print("Loading Arc-Intelligence/Arc-ATLAS-Teach-v0 dataset...")
    
    try:
        # Load specific training file directly using HuggingFace Hub
        from huggingface_hub import hf_hub_download
        import json
        
        # Download the RL training file which has all the fields we need
        file_path = hf_hub_download(
            repo_id="Arc-Intelligence/Arc-ATLAS-Teach-v0",
            filename="training/rl.jsonl",
            repo_type="dataset"
        )
        
        # Load the JSONL file
        problems = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    item = json.loads(line)
                    
                    # Extract fields based on the actual RL schema
                    problem_text = item.get("prompt", "")  # Field is 'prompt' in RL file
                    ground_truth = item.get("ground_truth", "")
                    
                    if problem_text:
                        problem_dict = {
                            "problem": problem_text,
                            "solution": ground_truth,
                            "source": "Arc-ATLAS-Teach-v0",
                            "problem_id": item.get("problem_id", ""),
                            "student_level": item.get("student_level", ""),
                            "baseline_score": item.get("baseline_score", 0),
                            "with_teaching_score": item.get("with_teaching_score", 0),
                            "teaching": item.get("teaching", ""),
                            "reward": item.get("reward", 0)
                        }
                        
                        # Extract numerical answer from ground_truth
                        import re
                        if ground_truth:
                            numbers = re.findall(r"[-+]?\d*\.?\d+", str(ground_truth))
                            if numbers:
                                try:
                                    problem_dict["answer"] = float(numbers[-1])
                                except:
                                    pass
                        
                        problems.append(problem_dict)
        
        # Sample if requested
        if num_samples and len(problems) > num_samples:
            problems = random.sample(problems, num_samples)
        
        print(f"Loaded {len(problems)} problems from Arc-ATLAS-Teach dataset")
        return problems
        
    except Exception as e:
        print(f"Error loading Arc-ATLAS-Teach dataset: {e}")
        print("Falling back to sample problems...")
        return get_sample_math_problems()


def load_bigmath_dataset(split: str = "train", num_samples: Optional[int] = 20) -> List[Dict[str, Any]]:
    """
    Load open-r1/Big-Math-RL-Verified-Processed dataset subset.
    
    Args:
        split: Dataset split to load
        num_samples: Number of samples to load (None for all)
        
    Returns:
        List of problem dictionaries
    """
    print("Loading open-r1/Big-Math-RL-Verified-Processed dataset...")
    
    try:
        # Use a specific config since the dataset requires one
        configs_to_try = ["level_1", "level_2", "all", "quintile_1"]
        
        dataset = None
        for config in configs_to_try:
            try:
                dataset = load_dataset("open-r1/Big-Math-RL-Verified-Processed", config, split=split)
                print(f"Successfully loaded with config '{config}'")
                break
            except:
                continue
        
        if dataset is None:
            raise Exception("No valid configuration found")
        
        # Convert to list of dictionaries
        problems = []
        for item in dataset:
            # Try different field names that might exist
            problem_text = (
                item.get("problem", "") or 
                item.get("question", "") or
                item.get("input", "")
            )
            
            solution_text = (
                item.get("solution", "") or
                item.get("answer", "") or
                item.get("output", "") or
                item.get("final_answer", "")
            )
            
            if problem_text:  # Only include if we have a problem
                problem_dict = {
                    "problem": problem_text,
                    "solution": solution_text,
                    "source": "Big-Math-RL-Verified-Processed"
                }
                
                # Extract numerical answer if available
                if "answer" in item:
                    problem_dict["answer"] = item["answer"]
                elif "final_answer" in item:
                    problem_dict["answer"] = item["final_answer"]
                
                problems.append(problem_dict)
        
        # Sample if requested
        if num_samples and len(problems) > num_samples:
            problems = random.sample(problems, num_samples)
        
        print(f"Loaded {len(problems)} problems from Big-Math dataset")
        return problems
        
    except Exception as e:
        print(f"Error loading Big-Math dataset: {e}")
        print("Falling back to sample problems...")
        return get_sample_math_problems()


def load_code_problems(num_samples: Optional[int] = 15) -> List[Dict[str, Any]]:
    """
    Load coding problems for code generation demo.
    
    Args:
        num_samples: Number of samples to load
        
    Returns:
        List of coding problem dictionaries
    """
    print("Loading coding problems...")
    
    try:
        # Try loading from HumanEval or similar coding dataset
        dataset = load_dataset("openai_humaneval", split="test")
        
        problems = []
        for item in dataset:
            problem_dict = {
                "problem": item.get("prompt", ""),
                "expected_behavior": item.get("docstring", ""),
                "test_cases": item.get("test", ""),
                "canonical_solution": item.get("canonical_solution", ""),
                "source": "HumanEval"
            }
            problems.append(problem_dict)
        
        # Sample if requested
        if num_samples and len(problems) > num_samples:
            problems = random.sample(problems, num_samples)
        
        print(f"Loaded {len(problems)} coding problems")
        return problems
        
    except Exception as e:
        print(f"Error loading coding dataset: {e}")
        print("Using sample coding problems...")
        return get_sample_code_problems()


def get_sample_math_problems() -> List[Dict[str, Any]]:
    """Fallback sample math problems."""
    return [
        {
            "problem": "Sarah has 24 apples. She gives 1/3 of them to her brother and 1/4 of the remaining apples to her sister. How many apples does Sarah have left?",
            "answer": 12,
            "solution": "Sarah starts with 24 apples. She gives 1/3 to her brother: 24 × 1/3 = 8 apples. Remaining: 24 - 8 = 16 apples. She gives 1/4 of remaining to her sister: 16 × 1/4 = 4 apples. Final amount: 16 - 4 = 12 apples.",
            "source": "sample"
        },
        {
            "problem": "A train travels 120 miles in 2 hours. If it maintains the same speed, how far will it travel in 5 hours?",
            "answer": 300,
            "solution": "Speed = Distance ÷ Time = 120 miles ÷ 2 hours = 60 miles per hour. Distance in 5 hours = Speed × Time = 60 mph × 5 hours = 300 miles.",
            "source": "sample"
        },
        {
            "problem": "The sum of two consecutive even numbers is 46. What are the two numbers?",
            "answer": "22 and 24",
            "solution": "Let the first even number be x. The next consecutive even number is x + 2. Sum: x + (x + 2) = 46. Solving: 2x + 2 = 46, so 2x = 44, and x = 22. The two numbers are 22 and 24.",
            "source": "sample"
        }
    ]


def get_sample_code_problems() -> List[Dict[str, Any]]:
    """Fallback sample coding problems."""
    return [
        {
            "problem": "Write a function that returns the factorial of a positive integer n.",
            "expected_behavior": "factorial(5) should return 120, factorial(0) should return 1",
            "canonical_solution": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
            "source": "sample"
        },
        {
            "problem": "Write a function that checks if a string is a palindrome (reads the same forwards and backwards).",
            "expected_behavior": "is_palindrome('racecar') should return True, is_palindrome('hello') should return False",
            "canonical_solution": "def is_palindrome(s):\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]",
            "source": "sample"
        }
    ]


def save_problems_to_json(problems: List[Dict[str, Any]], filename: str) -> None:
    """Save problems to JSON file for offline use."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(problems, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(problems)} problems to {filename}")


if __name__ == "__main__":
    # Load and save datasets for offline use
    
    # Math problems from ATLAS dataset
    try:
        atlas_problems = load_atlas_teach_dataset(num_samples=20)
        save_problems_to_json(atlas_problems, "math_problems_atlas.json")
    except Exception as e:
        print(f"Could not load ATLAS dataset: {e}")
    
    # Math problems from Big-Math dataset
    try:
        bigmath_problems = load_bigmath_dataset(num_samples=20)
        save_problems_to_json(bigmath_problems, "math_problems_bigmath.json")
    except Exception as e:
        print(f"Could not load Big-Math dataset: {e}")
    
    # Coding problems
    try:
        code_problems = load_code_problems(num_samples=15)
        save_problems_to_json(code_problems, "code_problems.json")
    except Exception as e:
        print(f"Could not load coding dataset: {e}")