---
title: Architecture Overview
description: High-level view of training, configs, and optional vLLM servers.
---

# Architecture Overview

```mermaid
flowchart LR
  subgraph Hydra
    C[configs/run/*.yaml]
    B[configs/{trainer,model,data}]
  end
  C -->|compose| T(train.py)
  B -->|instantiate| T
  T -->|SFT/GRPO| R[trainers/*]
  R -->|datasets| D(custom_data/*)
  R -->|logs/metrics| W[(Weights & Biases)]
  subgraph vLLM (optional)
    Srv[trainers/vllm_server.py]
    Cli[trainers/vllm_client/utils.py]
  end
  R -.generate.-> Cli
  Cli -.HTTP/NCCL.-> Srv
```

Key elements:

- `train.py`: Hydra entry; computes accumulation steps; handles W&B; orchestrates training.
- `trainers/*`: GRPO trainer (`TeacherGRPOTrainer`), reward functions, FSDP/ZeRO helpers, and optional vLLM client/server.
- `custom_data/*`: dataset formatting, tags, and SFT data preparation.
- `launch.sh` and `launch_with_server.sh`: process orchestration, GPU allocation, and vLLM server lifecycle.

