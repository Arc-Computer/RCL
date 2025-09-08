---
title: Distributed Training
description: Multi-GPU setup with DeepSpeed ZeRO and memory optimization.
---

# Distributed Training

RCL uses Accelerate + DeepSpeed for efficient multi-GPU training.

## Launch Flags

- `offload`: use `deepspeed_zero3_cpu_offloading.yaml`
- `zero1`: use `deepspeed_zero1.yaml`
- default: `deepspeed_zero3.yaml`

Example:

```bash
./launch.sh 4 configs/run/teacher_sft.yaml offload
```

## Memory and Throughput

- Reduce `per_device_train_batch_size`, increase accumulation.
- Use `generation_aggregation_steps` to improve vLLM utilization.
- Set `ds3_gather_for_generation=false` only if a single GPU lacks capacity; note vLLM compatibility.

## Seeds and Reproducibility

- Global `seed` in `configs/train.yaml`; RL run files may also declare `seed`.
- Log hardware/software versions in PRs for reproducibility.

