---
title: Performance Tuning
description: Optimizing GPU memory, throughput, and training time.
---

# Performance Tuning

## GPU Memory

- Prefer ZeRO-3 with CPU offload (`offload`) on constrained GPUs
- Enable gradient checkpointing and reduce `max_*_length`

## Throughput

- Increase `num_generations` and use `generation_aggregation_steps`
- Tune `vllm_gpu_memory_utilization` and `vllm_max_model_len`

## Stability

- Use conservative `temperature` and `top_p` initially
- Sync reference model periodically (`sync_ref_model`, `ref_model_sync_steps`)

## Target Environment

- 4×H100 recommended; expect slower runs with fewer GPUs or heavy offload

