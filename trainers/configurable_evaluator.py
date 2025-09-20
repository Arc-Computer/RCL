import re
import ast
import operator
from typing import Any, Dict, List, Optional, Callable
from .extraction_utils import ATLASExtractionUtils


class ConfigurableEvaluator:
    """User-configurable evaluation system for ATLAS optimization."""

    def __init__(self, evaluation_config: Optional[Dict[str, Any]] = None):
        self.config = evaluation_config or {}
        self.metrics_config = self.config.get('metrics', [])
        self.reward_formula = self.config.get('reward_formula')
        self.custom_functions = self.config.get('custom_functions', {})
        self._compiled_patterns = {}
        self.validate_config()

    def validate_config(self):
        """Validate evaluation configuration."""
        if not self.metrics_config:
            self.metrics_config = [
                {'name': 'correctness', 'type': 'exact_match', 'weight': 0.7},
                {'name': 'efficiency', 'type': 'token_reduction', 'weight': 0.3}
            ]

        total_weight = sum(m.get('weight', 0) for m in self.metrics_config)
        if abs(total_weight - 1.0) > 0.01:
            print(f"Warning: Metric weights sum to {total_weight}, normalizing to 1.0")
            if total_weight > 0:
                for metric in self.metrics_config:
                    metric['weight'] = metric.get('weight', 0) / total_weight

    def calculate_metric(
        self,
        metric_config: Dict[str, Any],
        response: str,
        baseline: str,
        ground_truth: str,
        question: str
    ) -> float:
        """Calculate a single metric based on configuration."""
        metric_type = metric_config.get('type', 'exact_match')

        if metric_type == 'exact_match':
            extracted = ATLASExtractionUtils.extract_solution(response)
            return 1.0 if ATLASExtractionUtils.check_correctness(extracted, ground_truth) else 0.0

        elif metric_type == 'contains':
            target = metric_config.get('target', ground_truth)
            return 1.0 if target.lower() in response.lower() else 0.0

        elif metric_type == 'regex':
            pattern = metric_config.get('pattern')
            if pattern:
                try:
                    compiled_pattern = re.compile(pattern)
                    return 1.0 if compiled_pattern.search(response) else 0.0
                except re.error:
                    return 0.0
            return 0.0

        elif metric_type == 'token_reduction':
            baseline_tokens = len(baseline.split())
            response_tokens = len(response.split())
            if baseline_tokens > 0 and response_tokens < baseline_tokens:
                return (baseline_tokens - response_tokens) / baseline_tokens
            return 0.0

        elif metric_type == 'length_penalty':
            max_length = metric_config.get('max_length', 500)
            response_length = len(response.split())
            if response_length <= max_length:
                return 1.0
            else:
                penalty = (response_length - max_length) / max_length
                return max(0, 1.0 - penalty)

        elif metric_type == 'custom':
            function_name = metric_config.get('function')
            if function_name in self.custom_functions:
                try:
                    custom_func = self.custom_functions[function_name]
                    return custom_func(response, baseline, ground_truth, question)
                except Exception as e:
                    print(f"Custom function {function_name} failed: {e}")
                    return 0.0

        return 0.0

    def evaluate(
        self,
        response: str,
        baseline: str,
        ground_truth: str,
        question: str
    ) -> Dict[str, float]:
        """Evaluate response using all configured metrics."""
        metrics = {}

        for metric_config in self.metrics_config:
            metric_name = metric_config.get('name', 'unnamed')
            metric_value = self.calculate_metric(
                metric_config, response, baseline, ground_truth, question
            )
            metrics[metric_name] = metric_value

        return metrics

    def _safe_eval_formula(self, formula: str, variables: Dict[str, Any]) -> float:
        """Safely evaluate a mathematical formula with restricted operations."""
        allowed_names = {
            'metrics': variables.get('metrics', {}),
            'baseline_metrics': variables.get('baseline_metrics', {}),
        }

        allowed_ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
            ast.Compare: lambda: None,
            ast.Gt: operator.gt,
            ast.Lt: operator.lt,
            ast.GtE: operator.ge,
            ast.LtE: operator.le,
            ast.Eq: operator.eq,
            ast.NotEq: operator.ne,
        }

        try:
            tree = ast.parse(formula, mode='eval')

            def _eval_node(node):
                if isinstance(node, ast.Constant):
                    return node.value
                elif isinstance(node, ast.Name):
                    if node.id in allowed_names:
                        return allowed_names[node.id]
                    raise ValueError(f"Name '{node.id}' not allowed")
                elif isinstance(node, ast.BinOp):
                    op = type(node.op)
                    if op in allowed_ops:
                        left = _eval_node(node.left)
                        right = _eval_node(node.right)
                        return allowed_ops[op](left, right)
                    raise ValueError(f"Operation {op} not allowed")
                elif isinstance(node, ast.Subscript):
                    value = _eval_node(node.value)
                    if isinstance(node.slice, ast.Constant):
                        key = node.slice.value
                    elif isinstance(node.slice, ast.Index):
                        key = _eval_node(node.slice.value)
                    else:
                        raise ValueError("Complex subscript not allowed")
                    return value.get(key, 0.0) if isinstance(value, dict) else 0.0
                elif isinstance(node, ast.IfExp):
                    test = _eval_node(node.test)
                    if test:
                        return _eval_node(node.body)
                    else:
                        return _eval_node(node.orelse)
                elif isinstance(node, ast.Compare):
                    left = _eval_node(node.left)
                    for op, comparator in zip(node.ops, node.comparators):
                        right = _eval_node(comparator)
                        op_type = type(op)
                        if op_type in allowed_ops:
                            if not allowed_ops[op_type](left, right):
                                return False
                            left = right
                        else:
                            raise ValueError(f"Comparison {op_type} not allowed")
                    return True
                else:
                    raise ValueError(f"Node type {type(node)} not allowed")

            return float(_eval_node(tree.body))
        except Exception:
            return 0.0

    def calculate_reward(
        self,
        metrics: Dict[str, float],
        baseline_metrics: Dict[str, float]
    ) -> float:
        """Calculate final reward using configured formula or default logic."""

        if self.reward_formula:
            result = self._safe_eval_formula(self.reward_formula, {
                'metrics': metrics,
                'baseline_metrics': baseline_metrics
            })
            if result != 0.0:
                return result

        # Default reward logic (backward compatible)
        weighted_score = 0.0
        for metric_config in self.metrics_config:
            metric_name = metric_config.get('name')
            weight = metric_config.get('weight', 0)

            if metric_name in metrics:
                improvement = metrics[metric_name] - baseline_metrics.get(metric_name, 0)
                weighted_score += improvement * weight

        return weighted_score

    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        **kwargs
    ) -> List[float]:
        """Compatible interface with OnlineTeachingReward."""
        rewards = []
        ground_truths = kwargs.get('ground_truths', [])
        baseline_solutions = kwargs.get('baseline_solutions', [])
        solutions = kwargs.get('solutions', [])
        questions = kwargs.get('questions', prompts)

        for i in range(len(prompts)):
            ground_truth = ground_truths[i] if i < len(ground_truths) else ""
            baseline = baseline_solutions[i] if i < len(baseline_solutions) else ""
            solution = solutions[i] if i < len(solutions) else ""
            question = questions[i] if i < len(questions) else ""

            if not ground_truth:
                rewards.append(0.0)
                continue

            # Calculate metrics for both responses
            solution_metrics = self.evaluate(solution, baseline, ground_truth, question)
            baseline_metrics = self.evaluate(baseline, baseline, ground_truth, question)

            # Calculate reward
            reward = self.calculate_reward(solution_metrics, baseline_metrics)
            rewards.append(reward)

        return rewards