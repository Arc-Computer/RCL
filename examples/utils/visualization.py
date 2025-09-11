"""
Visualization utilities for ATLAS examples.

Functions for creating charts, tables, and interactive displays.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import seaborn as sns
from IPython.display import display, HTML


def plot_comparison(metrics: Dict[str, Any], task_type: str = "math") -> None:
    """
    Create comparison plots for ATLAS performance.
    
    Args:
        metrics: Metrics dictionary from calculate_metrics
        task_type: Type of task ("math" or "code")
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'ATLAS Performance Analysis - {task_type.title()} Tasks', fontsize=16)
    
    # 1. Accuracy/Score Comparison
    ax1 = axes[0, 0]
    if task_type == "math":
        categories = ['Baseline', 'With ATLAS Teacher']
        values = [metrics['baseline_accuracy'] * 100, metrics['guided_accuracy'] * 100]
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Math Problem Accuracy')
    else:
        categories = ['Baseline', 'With ATLAS Teacher']
        values = [metrics['avg_baseline_score'] * 100, metrics['avg_guided_score'] * 100]
        ax1.set_ylabel('Quality Score (%)')
        ax1.set_title('Code Quality Score')
    
    bars1 = ax1.bar(categories, values, color=['#ff7f0e', '#2ca02c'], alpha=0.8)
    ax1.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, value in zip(bars1, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Improvements vs Degradations
    ax2 = axes[0, 1]
    improvements = metrics['improvements']
    degradations = metrics['degradations']
    unchanged = metrics['total_problems'] - improvements - degradations
    
    labels = ['Improved', 'Unchanged', 'Degraded']
    values = [improvements, unchanged, degradations]
    colors = ['#2ca02c', '#d3d3d3', '#ff4444']
    
    wedges, texts, autotexts = ax2.pie(values, labels=labels, colors=colors, 
                                       autopct='%1.1f%%', startangle=90)
    ax2.set_title('Response Quality Changes')
    
    # 3. Token Efficiency
    ax3 = axes[1, 0]
    token_categories = ['Baseline\n(Student Only)', 'Guidance\n(Teacher)', 'Response\n(With Teacher)']
    token_values = [
        metrics['avg_baseline_tokens'],
        metrics['avg_guidance_tokens'], 
        metrics['avg_guided_tokens']
    ]
    
    bars3 = ax3.bar(token_categories, token_values, 
                   color=['#ff7f0e', '#1f77b4', '#2ca02c'], alpha=0.8)
    ax3.set_ylabel('Average Tokens')
    ax3.set_title('Token Usage Comparison')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value in zip(bars3, token_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Teaching Strategy Distribution
    ax4 = axes[1, 1]
    strategies = metrics['teaching_strategies']
    strategy_labels = list(strategies.keys())
    strategy_values = list(strategies.values())
    strategy_colors = ['#ff9999', '#66b3ff', '#99ff99'][:len(strategy_labels)]
    
    bars4 = ax4.bar(strategy_labels, strategy_values, color=strategy_colors, alpha=0.8)
    ax4.set_ylabel('Number of Problems')
    ax4.set_title('Teaching Strategy Distribution')
    
    # Add value labels
    for bar, value in zip(bars4, strategy_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{value}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def display_results_table(metrics: Dict[str, Any], task_type: str = "math") -> None:
    """
    Display formatted results table.
    
    Args:
        metrics: Metrics dictionary from calculate_metrics
        task_type: Type of task ("math" or "code")
    """
    if task_type == "math":
        data = {
            "Metric": [
                "Average Accuracy",
                "Maximum Improvement", 
                "Problems Improved",
                "Problems Degraded",
                "Non-Degradation Rate",
                "Token Efficiency"
            ],
            "Student Alone": [
                f"{metrics['baseline_accuracy']:.1%}",
                "-",
                "-", 
                "-",
                "-",
                f"{metrics['avg_baseline_tokens']:.0f} tokens"
            ],
            "Student + ATLAS Teacher": [
                f"{metrics['guided_accuracy']:.1%}",
                f"+{metrics['improvement_percentage']:.1f}%",
                f"{metrics['improvements']} problems",
                f"{metrics['degradations']} problems", 
                f"{metrics['non_degradation_rate']:.1%}",
                f"{metrics['total_guided_tokens']:.0f} tokens"
            ],
            "Improvement": [
                f"+{metrics['improvement_percentage']:.1f}%",
                f"+{metrics['improvement_percentage']:.1f}%",
                f"+{metrics['improvements']}",
                f"{metrics['degradations']} degraded",
                f"{metrics['non_degradation_rate']:.1%}",
                f"{metrics['efficiency_improvement']:.1f}% overhead"
            ]
        }
    else:
        data = {
            "Metric": [
                "Average Quality Score",
                "Score Improvement",
                "Problems Improved", 
                "Problems Degraded",
                "Non-Degradation Rate",
                "Token Efficiency"
            ],
            "Student Alone": [
                f"{metrics['avg_baseline_score']:.1%}",
                "-",
                "-",
                "-", 
                "-",
                f"{metrics['avg_baseline_tokens']:.0f} tokens"
            ],
            "Student + ATLAS Teacher": [
                f"{metrics['avg_guided_score']:.1%}",
                f"+{metrics['improvement_percentage']:.1f}%",
                f"{metrics['improvements']} problems",
                f"{metrics['degradations']} problems",
                f"{metrics['non_degradation_rate']:.1%}",
                f"{metrics['total_guided_tokens']:.0f} tokens"
            ],
            "Improvement": [
                f"+{metrics['improvement_percentage']:.1f}%",
                f"+{metrics['improvement_percentage']:.1f}%", 
                f"+{metrics['improvements']}",
                f"{metrics['degradations']} degraded",
                f"{metrics['non_degradation_rate']:.1%}",
                f"{metrics['efficiency_improvement']:.1f}% overhead"
            ]
        }
    
    df = pd.DataFrame(data)
    
    # Style the table
    styled_df = df.style.set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#4CAF50'), ('color', 'white'), ('font-weight', 'bold')]},
        {'selector': 'td', 'props': [('text-align', 'center'), ('padding', '8px')]},
        {'selector': 'tr:nth-of-type(odd)', 'props': [('background-color', '#f9f9f9')]},
        {'selector': 'tr:hover', 'props': [('background-color', '#f0f0f0')]}
    ]).set_properties(**{'text-align': 'center'})
    
    display(HTML(f"<h3>ATLAS Performance Summary - {task_type.title()} Tasks</h3>"))
    display(styled_df)


def show_example_comparisons(
    problems: List[Dict[str, Any]], 
    results: List[Dict[str, Any]], 
    num_examples: int = 3
) -> None:
    """
    Show side-by-side example comparisons.
    
    Args:
        problems: Original problems
        results: ATLAS inference results
        num_examples: Number of examples to show
    """
    display(HTML("<h3>Example Comparisons: Student vs Student+Teacher</h3>"))
    
    for i in range(min(num_examples, len(results))):
        problem = problems[i]
        result = results[i]
        
        # Create comparison table for this example
        comparison_html = f"""
        <div style='margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; border-radius: 8px;'>
            <h4>Example {i+1}: {problem.get('problem', 'Problem')[:100]}...</h4>
            
            <div style='display: flex; gap: 20px;'>
                <div style='flex: 1; background-color: #fff3cd; padding: 10px; border-radius: 5px;'>
                    <h5 style='color: #856404; margin-top: 0;'>📝 Problem</h5>
                    <p style='margin: 0; font-size: 14px;'>{problem.get('problem', 'N/A')}</p>
                </div>
            </div>
            
            <div style='display: flex; gap: 20px; margin-top: 15px;'>
                <div style='flex: 1; background-color: #f8d7da; padding: 10px; border-radius: 5px;'>
                    <h5 style='color: #721c24; margin-top: 0;'>👤 Student Alone</h5>
                    <p style='margin: 0; font-size: 14px;'>{result['baseline_response'][:300]}{"..." if len(result['baseline_response']) > 300 else ""}</p>
                </div>
                
                <div style='flex: 1; background-color: #d1ecf1; padding: 10px; border-radius: 5px;'>
                    <h5 style='color: #0c5460; margin-top: 0;'>🤖 Teacher Guidance</h5>
                    <p style='margin: 0; font-size: 12px; font-style: italic;'><strong>Strategy:</strong> {result['teaching']['strategy']}</p>
                    <p style='margin: 5px 0 0 0; font-size: 14px;'>{result['teaching']['teaching_guidance'][:200]}{"..." if len(result['teaching']['teaching_guidance']) > 200 else ""}</p>
                </div>
                
                <div style='flex: 1; background-color: #d4edda; padding: 10px; border-radius: 5px;'>
                    <h5 style='color: #155724; margin-top: 0;'>✨ Student + Teacher</h5>
                    <p style='margin: 0; font-size: 14px;'>{result['guided_response'][:300]}{"..." if len(result['guided_response']) > 300 else ""}</p>
                </div>
            </div>
        </div>
        """
        
        display(HTML(comparison_html))


def create_diagnostic_analysis(results: List[Dict[str, Any]]) -> None:
    """
    Create analysis of diagnostic probing results.
    
    Args:
        results: ATLAS inference results
    """
    # Extract diagnostic data
    capability_scores = [r['diagnostic']['capability_score'] for r in results]
    strategies = [r['teaching']['strategy'] for r in results]
    
    # Create diagnostic distribution plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Capability score distribution
    ax1.hist(capability_scores, bins=range(1, 7), alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Capability Score (1-5)')
    ax1.set_ylabel('Number of Problems')
    ax1.set_title('Diagnostic Capability Assessment')
    ax1.set_xticks(range(1, 6))
    
    # Strategy mapping
    strategy_mapping = {'Light': 1, 'Medium': 2, 'Heavy': 3}
    strategy_colors = {'Light': '#2ca02c', 'Medium': '#ff7f0e', 'Heavy': '#d62728'}
    
    for i, (score, strategy) in enumerate(zip(capability_scores, strategies)):
        ax2.scatter(score, strategy_mapping[strategy], 
                   color=strategy_colors[strategy], alpha=0.6, s=50)
    
    ax2.set_xlabel('Diagnostic Score')
    ax2.set_ylabel('Teaching Strategy')
    ax2.set_title('Diagnostic Score → Teaching Strategy Mapping')
    ax2.set_yticks([1, 2, 3])
    ax2.set_yticklabels(['Light', 'Medium', 'Heavy'])
    ax2.set_xticks(range(1, 6))
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Summary statistics
    avg_score = np.mean(capability_scores)
    strategy_distribution = pd.Series(strategies).value_counts()
    
    print(f"\nDiagnostic Analysis Summary:")
    print(f"Average capability score: {avg_score:.2f}")
    print(f"Strategy distribution:")
    for strategy, count in strategy_distribution.items():
        print(f"  {strategy}: {count} problems ({count/len(strategies)*100:.1f}%)")