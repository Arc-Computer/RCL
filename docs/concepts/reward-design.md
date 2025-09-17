
# Reward Design

The reward system encourages teaching that improves student performance while preventing harmful interventions.

## Implementation

The reward function (`trainers/teacher_rewards.py::AdaptiveTeachingReward`) uses a simple but effective approach:

- **Performance degradation**: `reward = 0.0` (no reward for harmful teaching)
- **Performance unchanged + correct**: `reward = 0.5 × (1 + efficiency_bonus)`
- **Performance improvement**: `reward = delta × (1 + efficiency_bonus)`

Where `efficiency_bonus = 100 / (100 + teaching_length)` encourages concise guidance.

## Configuration

From `configs/trainer/reward/adaptive_teaching.yaml`:

```yaml
degradation_penalty_multiplier: 2.0  # Note: Not used in current implementation
efficiency_weight: 1.0                # Scales the efficiency bonus
max_probe_tokens: 500                 # Token limit for diagnostic probing
```

## Design Rationale

- **Zero reward for degradation**: Prevents harmful interventions without complex negative penalties
- **Efficiency weighting**: Rewards concise, targeted teaching over verbose explanations
- **Performance delta focus**: Direct correlation between student improvement and reward signal

The baseline is the student's solution without teacher guidance.

## See Also

- [Adaptive Teaching](adaptive-teaching.md) - Two-pass diagnostic and teaching protocol
- [RL Training](../guides/rl-training.md) - How rewards guide GRPO optimization
- [Performance Results](../../README.md#performance-results) - Complete results and reproducibility

