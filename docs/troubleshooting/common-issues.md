---
title: Common Issues
description: FAQ for setup, data access, OOM, and vLLM server lifecycle.
---

# Common Issues

## Hugging Face Authentication

- Symptom: 401/404 when loading datasets/models
- Fix: `huggingface-cli login`; ensure the token has dataset access

## Out of Memory (OOM)

- Use `offload` or `zero1` with `launch.sh`
- Reduce `per_device_train_batch_size`, `num_generations`, or context lengths

## vLLM Server Not Responding

- Check server logs for `Initializing server` and `/health` reachability
- Increase `vllm_server_timeout` and ensure one GPU is reserved for vLLM

## Mac vs. Linux Tooling

- `sed -i` and `stat -c` differ on macOS; prefer Linux for server scripts

## Artifacts in Git

- `results/`, `logs/`, `wandb/` are gitignored—avoid committing run outputs

