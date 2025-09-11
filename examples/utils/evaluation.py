"""
Evaluation utilities for ATLAS examples.

Functions for measuring accuracy, correctness, and performance improvements.
"""

import re
import json
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import Counter


def extract_numerical_answer(response: str) -> Optional[float]:
    """
    Extract numerical answer from math problem response.
    
    Args:
        response: Generated response text
        
    Returns:
        Extracted numerical value or None if not found
    """
    # Look for common answer patterns
    patterns = [
        r"(?:answer|result|solution)(?:\s*is\s*|\s*:\s*)([+-]?\d+\.?\d*)",
        r"(?:final|total|equals?)(?:\s*is\s*|\s*:\s*)([+-]?\d+\.?\d*)", 
        r"(?:\$|USD\s*)?([+-]?\d+\.?\d*)(?:\s*dollars?|\s*USD)?",
        r"([+-]?\d+\.?\d*)\s*(?:is the|is our|is my)?(?:\s*answer|\s*result|\s*solution)",
        r"(?:^|\s)([+-]?\d+\.?\d*)(?:\s*$|\s*\.|\s*,)",
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response.lower(), re.MULTILINE)
        if matches:
            try:
                return float(matches[-1])  # Take last match
            except (ValueError, IndexError):
                continue
    
    # Fallback: extract any number from the response
    numbers = re.findall(r"[+-]?\d+\.?\d*", response)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass
    
    return None


def check_code_correctness(code: str, expected_behavior: str) -> Dict[str, any]:
    """
    Basic code correctness checking (simplified for demo).
    
    Args:
        code: Generated code
        expected_behavior: Description of expected behavior
        
    Returns:
        Dict with correctness assessment
    """
    # Simple heuristics for code quality
    quality_score = 0
    issues = []
    
    # Check for basic structure
    if "def " in code:
        quality_score += 1
    else:
        issues.append("No function definition found")
    
    if "return " in code:
        quality_score += 1
    else:
        issues.append("No return statement found")
    
    # Check for common patterns
    if any(keyword in code for keyword in ["if ", "for ", "while "]):
        quality_score += 1
    
    # Check for documentation
    if '"""' in code or "'''" in code or "#" in code:
        quality_score += 1
    
    # Check for error handling
    if "try:" in code or "except" in code:
        quality_score += 1
    
    # Normalize score
    quality_score = min(quality_score / 5.0, 1.0)
    
    return {
        "quality_score": quality_score,
        "has_function": "def " in code,
        "has_return": "return " in code,
        "has_documentation": '"""' in code or "'''" in code or "#" in code,
        "issues": issues,
        "estimated_correctness": quality_score > 0.6
    }


def evaluate_math_responses(
    problems: List[Dict[str, any]],
    baseline_responses: List[str],
    guided_responses: List[str]
) -> Dict[str, any]:
    """
    Evaluate math problem solving performance.
    
    Args:
        problems: List of problems with expected answers
        baseline_responses: Student-only responses
        guided_responses: Student+teacher responses
        
    Returns:
        Evaluation metrics and comparisons
    """
    baseline_correct = 0
    guided_correct = 0
    improvements = 0
    degradations = 0
    
    detailed_results = []
    
    for i, (problem, baseline, guided) in enumerate(zip(problems, baseline_responses, guided_responses)):
        expected = problem.get("answer", problem.get("solution"))
        if isinstance(expected, str):
            expected_num = extract_numerical_answer(expected)
        else:
            expected_num = expected
        
        baseline_answer = extract_numerical_answer(baseline)
        guided_answer = extract_numerical_answer(guided)
        
        baseline_is_correct = (
            baseline_answer is not None and 
            expected_num is not None and
            abs(baseline_answer - expected_num) < 0.01
        )
        
        guided_is_correct = (
            guided_answer is not None and
            expected_num is not None and
            abs(guided_answer - expected_num) < 0.01
        )
        
        if baseline_is_correct:
            baseline_correct += 1
        if guided_is_correct:
            guided_correct += 1
        
        if guided_is_correct and not baseline_is_correct:
            improvements += 1
        elif baseline_is_correct and not guided_is_correct:
            degradations += 1
        
        detailed_results.append({
            "problem_index": i,
            "problem": problem.get("problem", ""),
            "expected_answer": expected_num,
            "baseline_answer": baseline_answer,
            "guided_answer": guided_answer,
            "baseline_correct": baseline_is_correct,
            "guided_correct": guided_is_correct,
            "improved": guided_is_correct and not baseline_is_correct,
            "degraded": baseline_is_correct and not guided_is_correct
        })
    
    total_problems = len(problems)
    baseline_accuracy = baseline_correct / total_problems if total_problems > 0 else 0
    guided_accuracy = guided_correct / total_problems if total_problems > 0 else 0
    accuracy_improvement = guided_accuracy - baseline_accuracy
    
    non_degradation_rate = 1 - (degradations / total_problems) if total_problems > 0 else 1.0
    
    # Calculate asymmetric reward with 2x degradation penalty
    # Based on docs/concepts/adaptive-teaching.md degradation_penalty_multiplier: 2.0
    total_reward = 0
    for result in detailed_results:
        if result["improved"]:
            total_reward += 1.0  # Positive reward for improvement
        elif result["degraded"]:
            total_reward -= 2.0  # 2x penalty for degradation (asymmetric)
        # No change = 0 reward
    
    avg_reward = total_reward / total_problems if total_problems > 0 else 0
    
    return {
        "total_problems": total_problems,
        "baseline_accuracy": baseline_accuracy,
        "guided_accuracy": guided_accuracy,
        "accuracy_improvement": accuracy_improvement,
        "improvement_percentage": accuracy_improvement * 100,
        "improvements": improvements,
        "degradations": degradations,
        "non_degradation_rate": non_degradation_rate,
        "asymmetric_reward": avg_reward,
        "total_reward": total_reward,
        "detailed_results": detailed_results
    }


def evaluate_code_responses(
    problems: List[Dict[str, any]],
    baseline_responses: List[str], 
    guided_responses: List[str]
) -> Dict[str, any]:
    """
    Evaluate code generation performance.
    
    Args:
        problems: List of coding problems
        baseline_responses: Student-only responses
        guided_responses: Student+teacher responses
        
    Returns:
        Evaluation metrics and comparisons
    """
    baseline_scores = []
    guided_scores = []
    improvements = 0
    degradations = 0
    
    detailed_results = []
    
    for i, (problem, baseline, guided) in enumerate(zip(problems, baseline_responses, guided_responses)):
        expected = problem.get("expected_behavior", "")
        
        baseline_eval = check_code_correctness(baseline, expected)
        guided_eval = check_code_correctness(guided, expected)
        
        baseline_score = baseline_eval["quality_score"]
        guided_score = guided_eval["quality_score"]
        
        baseline_scores.append(baseline_score)
        guided_scores.append(guided_score)
        
        if guided_score > baseline_score + 0.1:  # Significant improvement
            improvements += 1
        elif baseline_score > guided_score + 0.1:  # Significant degradation
            degradations += 1
        
        detailed_results.append({
            "problem_index": i,
            "problem": problem.get("problem", ""),
            "baseline_eval": baseline_eval,
            "guided_eval": guided_eval,
            "baseline_score": baseline_score,
            "guided_score": guided_score,
            "improved": guided_score > baseline_score + 0.1,
            "degraded": baseline_score > guided_score + 0.1
        })
    
    total_problems = len(problems)
    avg_baseline_score = np.mean(baseline_scores) if baseline_scores else 0
    avg_guided_score = np.mean(guided_scores) if guided_scores else 0
    score_improvement = avg_guided_score - avg_baseline_score
    
    non_degradation_rate = 1 - (degradations / total_problems) if total_problems > 0 else 1.0
    
    return {
        "total_problems": total_problems,
        "avg_baseline_score": avg_baseline_score,
        "avg_guided_score": avg_guided_score, 
        "score_improvement": score_improvement,
        "improvement_percentage": score_improvement * 100,
        "improvements": improvements,
        "degradations": degradations,
        "non_degradation_rate": non_degradation_rate,
        "detailed_results": detailed_results
    }


def calculate_token_efficiency(
    baseline_responses: List[str],
    guided_responses: List[str],
    teaching_guidance: List[str]
) -> Dict[str, any]:
    """
    Calculate token efficiency metrics.
    
    Args:
        baseline_responses: Student-only responses
        guided_responses: Student+teacher responses  
        teaching_guidance: Teacher guidance provided
        
    Returns:
        Token efficiency metrics
    """
    baseline_lengths = [len(response.split()) for response in baseline_responses]
    guided_lengths = [len(response.split()) for response in guided_responses]
    guidance_lengths = [len(guidance.split()) for guidance in teaching_guidance]
    
    avg_baseline_tokens = np.mean(baseline_lengths)
    avg_guided_tokens = np.mean(guided_lengths)
    avg_guidance_tokens = np.mean(guidance_lengths)
    
    # Total tokens includes both guidance and response
    total_guided_tokens = avg_guided_tokens + avg_guidance_tokens
    
    efficiency_ratio = avg_baseline_tokens / total_guided_tokens if total_guided_tokens > 0 else 1.0
    token_overhead = total_guided_tokens - avg_baseline_tokens
    
    return {
        "avg_baseline_tokens": avg_baseline_tokens,
        "avg_guided_tokens": avg_guided_tokens,
        "avg_guidance_tokens": avg_guidance_tokens,
        "total_guided_tokens": total_guided_tokens,
        "token_overhead": token_overhead,
        "efficiency_ratio": efficiency_ratio,
        "efficiency_improvement": (1 - efficiency_ratio) * 100
    }


def calculate_metrics(
    problems: List[Dict[str, any]],
    results: List[Dict[str, any]],
    task_type: str = "math"
) -> Dict[str, any]:
    """
    Calculate comprehensive metrics for ATLAS evaluation.
    
    Args:
        problems: Original problems
        results: ATLAS inference results
        task_type: Type of task ("math" or "code")
        
    Returns:
        Complete metrics dictionary
    """
    baseline_responses = [r["baseline_response"] for r in results]
    guided_responses = [r["guided_response"] for r in results]
    teaching_guidance = [r["teaching"]["teaching_guidance"] for r in results]
    
    # Task-specific evaluation
    if task_type == "math":
        task_metrics = evaluate_math_responses(problems, baseline_responses, guided_responses)
    else:
        task_metrics = evaluate_code_responses(problems, baseline_responses, guided_responses)
    
    # Token efficiency
    efficiency_metrics = calculate_token_efficiency(
        baseline_responses, guided_responses, teaching_guidance
    )
    
    # Teaching strategy distribution
    strategies = [r["teaching"]["strategy"] for r in results]
    strategy_counts = Counter(strategies)
    
    # Diagnostic score distribution  
    diagnostic_scores = [r["diagnostic"]["capability_score"] for r in results]
    avg_diagnostic_score = np.mean(diagnostic_scores)
    
    return {
        **task_metrics,
        **efficiency_metrics,
        "teaching_strategies": dict(strategy_counts),
        "avg_diagnostic_score": avg_diagnostic_score,
        "diagnostic_scores": diagnostic_scores
    }