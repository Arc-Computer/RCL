
# Testing Your Installation

Verify your RCL environment is correctly set up before running full experiments.

## Environment Verification

First, verify core dependencies are installed correctly:

```bash
# Check CUDA availability
python -c "import torch, transformers, datasets; print('CUDA available:', torch.cuda.is_available())"

# Check Accelerate version
accelerate --version

# Verify HuggingFace authentication
huggingface-cli whoami
```

**Expected output**: CUDA should be `True`, Accelerate version displayed, and your HuggingFace username shown.

## Quickstart Validation

Run the official quickstart commands to validate complete setup:

### SFT Warmup Test (1 epoch)

```bash
scripts/launch.sh 1 configs/run/teacher_sft.yaml report_to=null save_final_model=false num_train_epochs=1
```

**Expected**: Model initialization, dataset loading (Arc-ATLAS-Teach-v0), and 1 training epoch completing successfully.

### vLLM Integration Test (4 steps)

```bash
scripts/launch_with_server.sh 1 1 configs/run/teacher_rcl.yaml report_to=null max_steps=4 eval_steps=1
```

**Expected**: vLLM server startup, GRPO training with adaptive teaching rewards, and completion without errors.

## Memory-Constrained Testing

For limited GPU memory, use CPU offloading:

```bash
# SFT with offload
scripts/launch.sh 1 configs/run/teacher_sft.yaml report_to=null save_final_model=false num_train_epochs=1 offload

# RL with offload
scripts/launch_with_server.sh 1 1 configs/run/teacher_rcl.yaml report_to=null max_steps=4 eval_steps=1 offload
```

## Troubleshooting Common Issues

### CUDA/GPU Issues
```bash
# Check GPU memory
nvidia-smi

# Verify CUDA installation
nvcc --version
```

### vLLM Server Issues
```bash
# Check if port is in use
curl http://localhost:8765/health

# Use alternative port
scripts/launch_with_server.sh 1 1 configs/run/teacher_rcl.yaml report_to=null max_steps=4 eval_steps=1 vllm_port=8766
```

### Memory Management
- **Reduce batch size**: Add `per_device_train_batch_size=1` to commands
- **Use ZeRO-1**: Add `zero1` flag instead of `offload`
- **Check available memory**: Monitor with `nvidia-smi -l 1`

### Authentication Issues
```bash
# Re-authenticate with HuggingFace
huggingface-cli login

# Verify dataset access
python -c "from datasets import load_dataset; load_dataset('Arc-Intelligence/Arc-ATLAS-Teach-v0', split='train[:1]')"
```

## Success Indicators

Your installation is working correctly if:
- ✅ Both quickstart commands complete without errors
- ✅ vLLM server starts and responds to health checks
- ✅ Training logs show reward computation and policy updates
- ✅ GPU memory utilization is stable (check with `nvidia-smi`)

## Next Steps

Once tests pass successfully:
- Proceed to [First Experiment](first-experiment.md) for complete training workflow
- Review [README Troubleshooting](../../README.md#troubleshooting) for additional help

