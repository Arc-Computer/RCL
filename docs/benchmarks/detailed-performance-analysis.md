# Detailed Performance Analysis

Comprehensive evaluation results demonstrating ATLAS teaching effectiveness across multiple dimensions.

## Environment

- **Hardware**: 4×H100 GPUs
- **Dataset**: [Arc-Intelligence/Arc-ATLAS-Teach-v0](https://huggingface.co/datasets/Arc-Intelligence/Arc-ATLAS-Teach-v0)
- **Models**: [ATLAS-8B-Thinking](https://huggingface.co/Arc-Intelligence/ATLAS-8B-Thinking), [ATLAS-8B-Instruct](https://huggingface.co/Arc-Intelligence/ATLAS-8B-Instruct)
- **Seed**: 42
- **Evaluation**: 32 samples per problem

## Core Performance Metrics

### Teaching Effectiveness

| Metric | Teacher+Student | Student Alone | Delta |
|--------|----------------|---------------|-------|
| Average accuracy improvement | +15.7% | baseline | **+15.7%** |
| Maximum single improvement | +29.6% | - | **+29.6%** |
| Completion rate | ~100% | ~69% | **+31%** |
| Non-degradation rate | 97% | - | **97%** |

### Teaching Efficiency

| Metric | Teacher+Student | Student Alone | Improvement |
|--------|----------------|---------------|-------------|
| Token efficiency | 0.372 | - | *(efficiency metric)* |
| Average response length | ~2k tokens | ~4k tokens | **-50%** |
| Generation time (32 samples) | ~1:10 | ~1:21 | **-13.6%** |

## Key Findings

- **Non-degradation**: 97% of teaching interactions maintain or improve student performance
- **Efficiency gains**: Students achieve better results with 50% fewer tokens
- **Completion improvement**: Near-perfect completion rate vs 69% for students working alone
- **Speed improvement**: 13.6% faster generation while maintaining quality

## Methodology

ATLAS uses a two-pass evaluation protocol:

1. **Pass 1**: Diagnostic probing (≤50 tokens) to assess student capability
2. **Pass 2**: Adaptive teaching based on diagnosed capability level
3. **Reward calculation**: 0.0 reward for degradation, positive rewards for improvements

Students are evaluated on their final problem-solving accuracy after receiving ATLAS teaching versus working independently.

## Degradation Analysis

- **Overall degradation rate**: 3%
- **Degradation instances**: Primarily attributed to normalization issues in parsing
- **Non-degradation target**: ≥99%
- **Achieved rate**: 97%

The 3% degradation rate represents cases where student performance decreased after teaching, primarily due to technical parsing issues rather than harmful teaching content.

## Reproduction

```bash
# SFT Warmup
scripts/launch.sh 4 configs/run/teacher_sft.yaml dataset_id_or_path=Arc-Intelligence/Arc-ATLAS-Teach-v0

# RL Training
scripts/launch_with_server.sh 1 3 configs/run/teacher_rcl.yaml \
  model_name_or_path=results/pre_rl_model \
  dataset_id_or_path=Arc-Intelligence/Arc-ATLAS-Teach-v0 num_generations=32
```

See [reproduction.md](reproduction.md) for detailed reproduction steps and [evaluation.md](evaluation.md) for complete evaluation methodology.