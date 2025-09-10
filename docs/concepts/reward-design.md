
# Reward Design (Asymmetric)

The reward encourages teaching that improves student performance and discourages interventions that degrade already-strong students.

- Degradation penalty: 2× multiplier (configurable)
- Efficiency bonus: short, targeted teaching receives a positive length-aware bonus

Configuration (excerpt from `configs/trainer/reward/adaptive_teaching.yaml`):

```yaml
degradation_penalty_multiplier: 2.0
efficiency_weight: 1.0
max_probe_tokens: 50
```

Reference implementation: `trainers/teacher_rewards.py::AdaptiveTeachingReward`.

Notes:

- The baseline is the student's solution without teacher guidance.
- Reward combines delta vs. baseline and a token-length-based efficiency term.

## See Also

- [Adaptive Teaching](adaptive-teaching.md) - Two-pass diagnostic and teaching protocol
- [RL Training](../guides/rl-training.md) - How rewards guide GRPO optimization
- [Performance Results](../../README.md#performance-results) - Complete results and reproducibility

