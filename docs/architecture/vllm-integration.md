
# vLLM Integration

RCL can use vLLM for high-throughput generation during RL.

## Launch Strategy

```bash
# Syntax: <vllm_gpus> <training_gpus> <run_yaml> [overrides]
scripts/launch_with_server.sh 1 3 configs/run/teacher_rcl.yaml \
  model_name_or_path=results/pre_rl_model
```

- First `N` GPUs run vLLM servers (`trainers/vllm_server.py`).
- Remaining GPUs run training via `launch.sh` (Accelerate/DeepSpeed).

## Server and Client

- Server: FastAPI endpoints (`/health`, `/generate`, `/init_communicator`, `/update_named_param`, `/reset_prefix_cache`, `/close_communicator`).
- Client: `VLLMClient` handles health checks, generation, and NCCL-based weight sync.

Environment options:

- `enable_prefix_caching=true|false` (export before launching to pass through)
- `HF_HUB_ENABLE_HF_TRANSFER=1` (set in `train.py` for faster downloads)

Operational notes:

- Ensure one GPU remains free for vLLM when `use_vllm` is enabled.
- In server mode, `launch_with_server.sh` updates `vllm_host` and `num_vllm_clients` in the run YAML.
- On macOS, inline `sed -i` semantics differ; run on Linux or adjust locally.

