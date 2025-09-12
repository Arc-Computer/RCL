# Adaptive Learning

ATLAS moves beyond "reasoning alignment" toward pedagogical effectiveness. The teacher model first diagnoses the student's capability with a brief probe, then generates targeted guidance that improves outcomes without harming strong students.

## Diagnostic Probing

- A diagnostic probe elicits the student's approach and reveals capability
- Token limit configured via `max_probe_tokens` (default: 500 tokens)
- See `GRPOConfig.max_probe_tokens` and templates in `TeacherGRPOTrainer`

## Targeted Guidance

- The teacher model conditions on the probe to deliver calibrated help (hints, scaffolding, corrections) only as needed
- Guidance must be concise and effective; excessive intervention is discouraged
- Efficiency bonus rewards shorter, more targeted teaching

## Reward Signal

The reward system uses a simple but effective approach:

```yaml
# From configs/trainer/reward/adaptive_teaching.yaml
degradation_penalty_multiplier: 2.0  # Configured but not used in current implementation
efficiency_weight: 1.0                # Scales the efficiency bonus  
max_probe_tokens: 500                 # Token limit for diagnostic probing
```

Implementation: `trainers/teacher_rewards.py::AdaptiveTeachingReward` computes performance deltas vs. a baseline (no guidance) and:
- Returns 0 reward for performance degradation (preventing harmful interventions)
- Rewards improvements with `delta × (1 + efficiency_bonus)`
- Gives partial reward (0.5) for maintaining correct performance with efficient teaching

## See Also

- [Reward Design](reward-design.md) - Details on reward structure and efficiency bonuses
- [RL Training](../guides/rl-training.md) - GRPO implementation with adaptive learning
- [Performance Results](../../README.md#performance-results) - Complete results and reproducibility