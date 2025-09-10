
# RL Training (GRPO)

GRPO optimizes the teacher with an asymmetric reward that discourages degradation and rewards concise, effective teaching.

## Command

```bash
scripts/launch_with_server.sh 1 3 configs/run/teacher_rcl.yaml \
  model_name_or_path=results/pre_rl_model \
  dataset_id_or_path=Arc-Intelligence/Arc-ATLAS-Teach-v0 \
  num_generations=64 generation_aggregation_steps=256 \
  temperature=0.7 beta=0.04 max_steps=1000
```

## Important Parameters

- Sampling: `temperature`, `top_p`, `repetition_penalty`
- Throughput: `num_generations`, `generation_aggregation_steps`
- KL/Ref sync: `beta`, `sync_ref_model`, `ref_model_mixup_alpha`, `ref_model_sync_steps`
- Memory: `offload_untrained_models`, `ds3_gather_for_generation`
- Reward: `degradation_penalty_multiplier`, `efficiency_weight` (via reward config)

## Probing and Templates

`TeacherGRPOTrainer` supports:

- `max_probe_tokens`: upper bound for diagnostic approach length
- `student_diagnostic_template`, `teacher_adaptive_template`, `student_*_template`

See `trainers/teacher_trainers.py` and `configs/trainer/reward/adaptive_teaching.yaml`.

## See Also

- [Adaptive Teaching](../concepts/adaptive-teaching.md) - Core diagnostic and teaching protocol
- [Reward Design](../concepts/reward-design.md) - Asymmetric reward structure
- [Distributed Training](distributed-training.md) - Multi-GPU setup and memory optimization
- [Methodology & Performance](../../README.md#methodology--performance) - Complete results and reproducibility

