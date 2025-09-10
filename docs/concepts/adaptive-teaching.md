---
title: Adaptive Teaching
description: Core RCL concept—diagnose capability, then teach to avoid degradation.
---

# Adaptive Teaching

RCL moves beyond “reasoning alignment” toward pedagogical effectiveness. The teacher first diagnoses the student’s capability with a brief probe, then generates targeted guidance that improves outcomes without harming strong students.

## Diagnostic Probing

- A short probe (≤50 tokens) elicits the student’s approach and reveals capability.
- See `GRPOConfig.max_probe_tokens` and templates in `TeacherGRPOTrainer`.

## Targeted Teaching

- The teacher conditions on the probe to deliver calibrated help (hints, scaffolding, corrections) only as needed.
- Teaching must be concise and effective; excessive guidance is discouraged.

## Reward Signal

- Asymmetric: degradation is penalized more than improvements are rewarded.
- See `configs/trainer/reward/adaptive_teaching.yaml`:

```yaml
degradation_penalty_multiplier: 2.0
efficiency_weight: 1.0
max_probe_tokens: 50
```

Implementation: `trainers/teacher_rewards.py::AdaptiveTeachingReward` computes performance deltas vs. a baseline (no teaching), applies a 2× penalty for degradation, and adds a length-aware efficiency bonus.

## See Also

- [Reward Design](reward-design.md) - Details on asymmetric rewards and efficiency bonuses
- [RL Training](rl-training.md) - GRPO implementation with adaptive teaching
- [Methodology & Performance](../../README.md#methodology--performance) - Complete results and reproducibility

