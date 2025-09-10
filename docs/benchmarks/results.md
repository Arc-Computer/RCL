
# Benchmark Results

## Performance Summary

Verified results demonstrate non-degradation teaching with significant efficiency gains across all metrics.

**Environment**: 4×H100 GPUs, seed=42
**Dataset**: [Arc-Intelligence/Arc-ATLAS-Teach-v0](https://huggingface.co/datasets/Arc-Intelligence/Arc-ATLAS-Teach-v0)
**Models**: [ATLAS-8B-Thinking](https://huggingface.co/Arc-Intelligence/ATLAS-8B-Thinking), [ATLAS-8B-Instruct](https://huggingface.co/Arc-Intelligence/ATLAS-8B-Instruct)

| Metric | Teacher+Student | Student Alone | Delta |
|--------|----------------|---------------|-------|
| Average accuracy | +0.1573 | baseline | **+15.73%** |
| Max per-item improvement | +0.296 | - | **+29.6%** |
| Token efficiency | 0.372 | - | (avg. efficiency term) |
| Generation time (32 samples) | ~1:10 | ~1:21 | **-13.6%** |
| Average length | ~2k tokens | ~4k tokens | **-50%** |
| Completion rate | ~100% | ~69% | **+31%** |

## Key Findings

- **Non-degradation teaching**: Teacher+student consistently achieves higher accuracy with fewer tokens
- **Efficiency gains**: Near-100% completion rate vs. ~69% student-alone
- **Scalable performance**: Consistent improvements across problem difficulty levels
- **Production ready**: Faster generation times with reduced token usage

## Learning Metrics

- **Learning Rate (LR)**: Performance change per interaction
- **Non-Degradation Rate (NDR)**: Interactions that don't hurt performance (target: ≥99%)
- **Teaching Efficiency Score (TES)**: Performance gain per teaching token

## Hardware Requirements

- **Minimum**: Single GPU for quickstart validation
- **Recommended**: 4×H100 GPUs for production training
- **Memory**: Use `offload` flag for constrained GPU setups
- **Network**: High-bandwidth interconnect for distributed vLLM server training

