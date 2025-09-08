---
title: Training Pipeline
description: End-to-end workflow—SFT warmup followed by GRPO-based RL.
---

# Training Pipeline

RCL uses a two-phase pipeline: SFT warmup → RL training with GRPO.

## 1) SFT Warmup

```bash
./launch.sh 4 configs/run/teacher_sft.yaml \
  dataset_id_or_path=bespokelabs/Bespoke-Stratos-17k \
  num_train_epochs=3
```

Tips:

- Use `offload` or `zero1` flags to reduce memory.
- Tune `per_device_train_batch_size` and `train_batch_size`; Hydra will infer accumulation steps.

## 2) RL Training (GRPO)

```bash
./launch_with_server.sh 1 3 configs/run/teacher_rcl.yaml \
  model_name_or_path=results/pre_rl_model \
  dataset_id_or_path=Arc-Intelligence/Arc-ATLAS-Teach-v0 \
  num_generations=64 generation_aggregation_steps=256 temperature=0.7
```

Key settings (see `GRPOConfig`):

- Sampling: `temperature`, `top_p`, `repetition_penalty`
- Memory: `ds3_gather_for_generation`, `offload_untrained_models`
- Throughput: `num_generations`, `generation_aggregation_steps`
- KL: `beta`; Reference sync: `sync_ref_model`, `ref_model_mixup_alpha`, `ref_model_sync_steps`

## 3) Checkpoints and Logging

- W&B if `report_to: wandb` (set in `configs/train.yaml`)
- Files in `${output_dir}`; cadence controlled by `save_strategy`/`save_steps`

