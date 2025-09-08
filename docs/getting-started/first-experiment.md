---
title: First Experiment
description: End-to-end SFT warmup and RL training on Arc‑ATLAS‑Teach.
---

# First Experiment

This walkthrough runs the SFT warmup then GRPO-based RL with vLLM, targeting 4×H100.

## SFT Warmup

```bash
./launch.sh 4 configs/run/teacher_sft.yaml \
  dataset_id_or_path=bespokelabs/Bespoke-Stratos-17k \
  output_dir=results/pre_rl_model \
  num_train_epochs=3
```

Key options:

- `gradient_accumulation_steps` is computed automatically from global/per-device batch settings.
- Use `offload` or `zero1` flags with `launch.sh` for memory savings.

## RL Training (GRPO)

```bash
./launch_with_server.sh 1 3 configs/run/teacher_rcl.yaml \
  model_name_or_path=results/pre_rl_model \
  dataset_id_or_path=Arc-Intelligence/Arc-ATLAS-Teach-v0 \
  num_generations=64 temperature=0.7 beta=0.04 \
  max_steps=1000 save_final_model=true
```

Notes:

- vLLM servers consume the first `N` GPUs; training uses the remainder.
- Reward uses an asymmetric degradation penalty and efficiency bonus (see Concepts → Reward Design).
- Configure saving cadence via `save_strategy` and `save_steps`.

## Outputs

- Hydra output directory: `configs/train.yaml` → `${results_dir}/${wandb_project}/${wandb_group_name}/${wandb_run_name}/${exp_name}`
- Checkpoints and logs within the above directory; W&B logging if `report_to: wandb`.

