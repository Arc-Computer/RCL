
# Common Issues

## Hugging Face Authentication

- Symptom: 401/404 when loading datasets/models
- Fix: `huggingface-cli login`; ensure the token has dataset access

## Out of Memory (OOM)

- Prefer ZeRO‑3 (default). If OOM:
  - Use `offload` with `launch.sh` to enable CPU offloading
  - Try `zero1` for simpler memory partitioning
  - Reduce `per_device_train_batch_size`, `num_generations`, or context lengths (`max_seq_length`, `max_*_length`)

## vLLM Server Not Responding

- Check server logs for `Initializing server` and `/health` reachability
- Increase `vllm_server_timeout` and ensure one GPU is reserved for vLLM

## vLLM Port Conflicts

- Symptom: server fails to bind to port or clients can’t connect
- Fix: set a free `vllm_port` in your run YAML (e.g., `vllm_port: 8766`) and re‑launch; ensure firewall allows traffic on that port
- Cleanup: kill leftover processes (`pkill -f vllm_server`) and free NCCL group ports if reused

## Mac vs. Linux Tooling

- `sed -i` and `stat -c` differ on macOS; prefer Linux for server scripts

## Artifacts in Git

- `results/`, `logs/`, `wandb/` are gitignored—avoid committing run outputs

## Dataset Authentication Issues

- Symptom: 401/403 when loading datasets
- Fix: `huggingface-cli login` (ensure correct org access); on headless, set `HF_TOKEN` in the environment
- For gated datasets, request access on the dataset card before running
