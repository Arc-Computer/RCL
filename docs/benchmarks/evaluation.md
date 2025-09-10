
# Evaluation Methodology

## Core Evaluation Principles

Verify that adaptive teaching improves student outcomes without degrading performance for strong students. The evaluation framework measures both quantitative metrics and qualitative teaching effectiveness.

## Evaluation Protocol

### Baseline Comparison
- **Student Alone**: Run inference without teacher guidance
- **Teacher+Student**: Run full two-pass adaptive teaching protocol
- **Metrics**: Compare accuracy, completion rates, token efficiency, and generation time

### Non-Degradation Verification
- Track cases where teaching hurts student performance
- Measure Non-Degradation Rate (NDR): target ≥99% non-harmful interactions
- Identify failure modes in adaptive templates if degradation exceeds threshold

### Efficiency Metrics
- **Learning Rate (LR)**: Performance change per interaction
- **Teaching Efficiency Score (TES)**: Performance gain per teaching token
- **Token Efficiency**: Average efficiency term from reward computation

## Evaluation Commands

### Full Benchmark Evaluation

```bash
# Run complete evaluation with logging
scripts/launch_with_server.sh 1 3 configs/run/teacher_rcl.yaml \
  model_name_or_path=results/pre_rl_model \
  dataset_id_or_path=Arc-Intelligence/Arc-ATLAS-Teach-v0 \
  eval_steps=50 log_completions=true save_completions_probability=0.1
```

### Quick Validation

```bash
# Minimal evaluation for development
scripts/launch_with_server.sh 1 1 configs/run/teacher_rcl.yaml \
  report_to=null max_steps=4 eval_steps=1
```

## Data Collection

### Quantitative Metrics
- Accuracy improvements vs baseline
- Completion rate comparisons
- Token usage and generation time
- GPU utilization and training stability

### Qualitative Analysis
- Review teacher guidance quality across difficulty levels
- Analyze probe effectiveness in capability diagnosis
- Examine failure cases for template improvements

### Reproducibility Requirements
- Record exact environment specifications (hardware, software versions)
- Document all configuration overrides
- Save representative examples (avoid committing large artifacts)
- Include training logs and metric summaries

## Expected Outcomes

Successful evaluation should demonstrate:
- **+15-30%** accuracy improvements
- **Near 100%** completion rates (vs ~69% student-alone)
- **50%** reduction in token usage
- **13-15%** faster generation times
- **≥99%** non-degradation rate for strong students

## Analysis Framework

### Statistical Significance
- Use held-out test sets for unbiased evaluation
- Report confidence intervals for key metrics
- Validate across multiple random seeds

### Error Analysis
- Categorize failure modes by difficulty level
- Identify systematic issues in teaching templates
- Measure probe diagnostic accuracy

### Scalability Testing
- Evaluate across different model sizes
- Test distributed training configurations
- Validate vLLM server performance under load

