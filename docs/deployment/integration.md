---
title: Integration
description: Integrate RCL models into existing ML pipelines and systems.
---

# Integration

## Pipelines

- Export checkpoints and load with Transformers for batch scoring.
- Use vLLM for high-throughput serving with HTTP APIs.

## Environments

- Set `HF_HUB_ENABLE_HF_TRANSFER=1` for faster IO.
- Configure `WANDB_API_KEY` only when logging; otherwise `report_to: null`.

## Safety and Governance

- Do not store access tokens in configs; authenticate via CLI or environment.
- Document seeds, exact package versions, and GPU topology for reproducibility.

