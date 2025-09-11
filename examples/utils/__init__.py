"""
ATLAS Example Utilities

Core components for demonstrating ATLAS two-pass inference protocol.
"""

from .atlas_inference import ATLASInference, load_atlas_models
from .evaluation import (
    extract_numerical_answer, check_code_correctness, 
    evaluate_math_responses, evaluate_code_responses,
    calculate_metrics, calculate_token_efficiency
)
from .visualization import (
    plot_comparison, display_results_table, show_example_comparisons,
    create_diagnostic_analysis
)

__all__ = [
    'ATLASInference',
    'load_atlas_models', 
    'extract_numerical_answer',
    'check_code_correctness',
    'evaluate_math_responses',
    'evaluate_code_responses', 
    'calculate_metrics',
    'calculate_token_efficiency',
    'plot_comparison',
    'display_results_table',
    'show_example_comparisons',
    'create_diagnostic_analysis'
]