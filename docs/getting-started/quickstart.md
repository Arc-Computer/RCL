---
title: Quickstart
description: 5-minute guide to validate setup and see first results with RCL training.
---

# Quickstart: 5 Minutes to First Result

Get your first RCL training results in under 5 minutes. This guide validates your setup with minimal configurations before moving to full training.

## Prerequisites

- **GPU**: Minimum 4x A100/H100 (40GB+ VRAM each)
- **CUDA**: 12.4+ installed
- **Storage**: 50GB free space
- **Network**: Access to HuggingFace Hub

## Quick Validation (< 5 minutes)

### Step 1: Environment Setup (30 seconds)

```bash
# Quick dependency check
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Install if needed (first-time only, ~3 minutes)
bash scripts/install_08.sh

# Authenticate
huggingface-cli login
```

**Expected output:**
```
PyTorch: 2.6.0+cu124
CUDA available: True
```

### Step 2: Minimal SFT Warmup (2 minutes)

```bash
# Quick validation run - 10 steps only
./launch.sh 4 configs/run/quickstart_sft.yaml \
  output_dir=results/quickstart_sft
```

**Expected logs (within 30 seconds):**
```
[INFO] Loading model: Qwen/Qwen2.5-7B-Instruct
[INFO] Model initialized successfully
[INFO] Starting training with 10 steps
{'loss': 2.345, 'learning_rate': 1e-05, 'epoch': 0.1}
```

**Success indicators:**
- ✅ Model loads without OOM errors
- ✅ Training loss decreases (2.3 → 1.8)
- ✅ Completes in ~2 minutes

### Step 3: Minimal RL Training (2.5 minutes)

```bash
# Quick RL validation - 4 steps, 2 generations
./launch_with_server.sh 1 3 configs/run/quickstart_rcl.yaml \
  model_name_or_path=results/quickstart_sft
```

**Expected logs:**
```
Running launch script...
[INFO] Starting vLLM server on port 8765
[INFO] Server ready - starting training
{'rewards/mean': 0.234, 'kl': 0.045, 'epoch': 0.1}
[INFO] Step 4/4 completed
```

**Success indicators:**
- ✅ vLLM server starts successfully
- ✅ Generations complete without timeout
- ✅ Positive mean rewards
- ✅ KL divergence < 0.1

## Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Add memory optimization flags
./launch.sh 4 configs/run/quickstart_sft.yaml offload \
  gradient_checkpointing=true \
  per_device_train_batch_size=1
```

#### vLLM Server Timeout
```bash
# Check server logs
tail -f job_*.log

# Increase timeout if needed
vllm_server_timeout=180
```

#### Import Errors
```bash
# Reinstall with specific versions
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install vllm==0.8.3
```

## Next Steps

### 30-Minute Full Validation

Run a complete but small training loop:

```bash
# Full SFT epoch on subset
./launch.sh 4 configs/run/teacher_sft.yaml \
  num_train_epochs=1 \
  max_steps=100 \
  output_dir=results/validation_sft

# Full RL training with evaluation
./launch_with_server.sh 2 2 configs/run/teacher_rcl.yaml \
  model_name_or_path=results/validation_sft \
  max_steps=25 \
  num_generations=8 \
  eval_steps=5
```

### Production Training

For full training runs, see:
- [Training Pipeline Guide](training-pipeline.md) - Complete 2-phase training
- [Distributed Training](../guides/distributed-training.md) - Multi-node setup
- [Configuration Guide](../guides/configuration.md) - All parameters explained

## Automated Quickstart

Run everything automatically:

```bash
# Runs all validation steps with error checking
bash scripts/quickstart.sh

# Expected output:
# ✅ Environment validated
# ✅ SFT training successful (2 min)
# ✅ RL training successful (2.5 min)
# Total time: 4 min 47 sec
```

## Expected Compute Requirements

| Step | GPUs | VRAM/GPU | Time | Purpose |
|------|------|----------|------|---------|
| Quickstart SFT | 4 | 20GB | 2 min | Validate setup |
| Quickstart RL | 4 | 30GB | 2.5 min | Test pipeline |
| Full Validation | 4 | 40GB | 30 min | Small training |
| Production | 8 | 80GB | 2-5 days | Full training |

## Quick Reference

```bash
# Minimal commands for validation
./launch.sh 4 configs/run/quickstart_sft.yaml
./launch_with_server.sh 1 3 configs/run/quickstart_rcl.yaml

# With memory constraints
./launch.sh 4 configs/run/quickstart_sft.yaml offload
./launch_with_server.sh 1 3 configs/run/quickstart_rcl.yaml offload

# Skip all logging/saving for speed
report_to=null save_final_model=false logging_steps=999
```

---

**Next:** [Testing Your Installation](testing-installation.md) → [First Experiment](first-experiment.md)

