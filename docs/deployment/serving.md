---
title: Serving with vLLM
description: Running vLLM servers and integrating with client code.
---

# Serving with vLLM

## Training-time Servers

Use `launch_with_server.sh` to run vLLM servers and training together:

```bash
./launch_with_server.sh 1 3 configs/run/teacher_rcl.yaml \
  model_name_or_path=results/pre_rl_model
```

## Standalone Server

```bash
python trainers/vllm_server.py \
  --model your/model --port 8765 --gpu_memory_utilization 0.9 --dtype auto
```

Then use the client:

```python
from trainers.vllm_client.utils import VLLMClient

client = VLLMClient(host="0.0.0.0", server_port=8765, group_port=51216, connection_timeout=120)
ids = client.generate(["Hello"], n=1, max_tokens=32)
```

Operational tips:

- Keep one GPU for vLLM; others for training.
- Use `enable_prefix_caching` when supported and beneficial.
- Reset prefix cache between evaluation phases if needed.

