
# Configuration System (Hydra)

Hydra composes configs from `configs/`:

- `configs/train.yaml`: global defaults (logging, W&B, output dirs, seed)
- `configs/run/*.yaml`: runnable recipes (e.g., `teacher_sft.yaml`, `teacher_rcl.yaml`)
- `configs/{model,data,trainer}/*.yaml`: reusable building blocks
- `accelerate/*.yaml`: DeepSpeed ZeRO-1/3 and CPU offload

## Composition and Overrides

Example:

```bash
scripts/launch.sh 4 configs/run/teacher_sft.yaml \
  dataset_id_or_path=Arc-Intelligence/Arc-ATLAS-Teach-v0 \
  num_train_epochs=3
```

Hydra overrides are passed as `key=value`. `train.py` infers `gradient_accumulation_steps` from global/per-device batch.

## DeepSpeed/Accelerate

`launch.sh` selects the Accelerate config:

- default: `accelerate/deepspeed_zero3.yaml`
- `zero1`: `accelerate/deepspeed_zero1.yaml`
- `offload`: `accelerate/deepspeed_zero3_cpu_offloading.yaml`

Use `offload` for constrained GPUs; expect slower training.

## vLLM Options (GRPO)

Key fields in `GRPOConfig`:

- `use_vllm`, `use_vllm_server`, `vllm_host`, `vllm_port`, `num_vllm_clients`
- `num_generations`, `generation_aggregation_steps`, `temperature`, `top_p`
- `ds3_gather_for_generation`, `offload_untrained_models`

For server mode, use `launch_with_server.sh` which sets `vllm_host` and client count automatically.

