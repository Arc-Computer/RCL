
# Quickstart

Run minimal smoke tests on a single GPU to validate your RCL setup.

## Prerequisites

Before running quickstart commands, ensure you have:
- Completed [installation](installation.md) steps
- Authenticated with HuggingFace: `huggingface-cli login`
- CUDA-compatible GPU available

## SFT Warmup (10 steps)

Test supervised fine-tuning warmup phase:

```bash
./launch.sh 1 configs/run/quickstart_sft.yaml report_to=null save_final_model=false
```

**Expected output**: Initialization logs, dataset loading, and 10 training steps completing successfully.

## RL Training with vLLM (4 steps)

Test reinforcement learning with vLLM server integration:

```bash
./launch_with_server.sh 1 1 configs/run/quickstart_rcl.yaml report_to=null max_steps=4 eval_steps=1
```

**Expected output**: vLLM server startup, GRPO training loop with reward computation, and successful completion.

## Important Notes

- **quickstart_rcl.yaml**: Uses `use_vllm_server: true` by default
- **launch_with_server.sh**: Automatically sets `vllm_host` and validates `vllm_port`
- **Memory optimization**: Add `offload` at the end of commands to reduce GPU memory usage

## Troubleshooting

**GPU Memory Issues**: Add `offload` flag:
```bash
./launch.sh 1 configs/run/quickstart_sft.yaml report_to=null save_final_model=false offload
```

**Port Conflicts**: Override vLLM port:
```bash
./launch_with_server.sh 1 1 configs/run/quickstart_rcl.yaml report_to=null max_steps=4 eval_steps=1 vllm_port=8766
```

**Authentication Errors**: Ensure HuggingFace login:
```bash
huggingface-cli login
```

## Next Steps

- [Test Your Installation](testing-installation.md) - Verify environment setup
- [First Experiment](first-experiment.md) - Complete training workflow
