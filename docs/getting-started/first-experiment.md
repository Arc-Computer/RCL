
# First Experiment

Complete walkthrough of the RCL two-phase training pipeline: SFT warmup followed by GRPO-based reinforcement learning with vLLM server integration.

## Hardware Requirements

**Recommended**: 4×H100 GPUs (or equivalent) with high-bandwidth interconnect
**Minimum**: Single GPU with memory optimization flags

## Phase 1: SFT Warmup

Train the base model with supervised fine-tuning on reasoning data:

```bash
scripts/launch.sh 8 configs/run/teacher_sft.yaml \
  output_dir=path/to/save/pre_rl_model
```

**Key parameters**:
- **Default dataset**: `Arc-Intelligence/Arc-ATLAS-Teach-v0` (SFT split)
- **Model**: 8B parameter teacher model
- **Duration**: Complete SFT warmup phase
- **Output**: Saved model checkpoint ready for RL training

**Memory optimization** (if needed):
```bash
# Use CPU offloading for memory-constrained setups
scripts/launch.sh 8 configs/run/teacher_sft.yaml \
  output_dir=path/to/save/pre_rl_model \
  offload

# Or use ZeRO-1 for less aggressive optimization
scripts/launch.sh 8 configs/run/teacher_sft.yaml \
  output_dir=path/to/save/pre_rl_model \
  zero1
```

## Phase 2: RL Training (2-3 days)

Train adaptive teaching capabilities using GRPO with vLLM server:

```bash
scripts/launch_with_server.sh 4 4 configs/run/teacher_rcl.yaml \
  model_name_or_path=path/of/saved/pre_rl_model
```

**Key parameters**:
- **GPU allocation**: 4 GPUs for vLLM servers, 4 GPUs for training
- **Default dataset**: `Arc-Intelligence/Arc-ATLAS-Teach-v0` (RL dataset)
- **Training time**: 2-3 days for full convergence
- **Adaptive teaching**: Two-pass protocol with diagnostic probing

**Training configuration**:
```bash
scripts/launch_with_server.sh 4 4 configs/run/teacher_rcl.yaml \
  model_name_or_path=path/of/saved/pre_rl_model \
  num_generations=32 \
  temperature=0.7 \
  beta=0.04 \
  degradation_penalty_multiplier=2.0
```

## Understanding the Process

### SFT Phase
- **Purpose**: Establish base reasoning capabilities
- **Data**: General reasoning problems with solutions
- **Output**: Model capable of solving problems but without adaptive teaching

### RL Phase  
- **Purpose**: Learn adaptive teaching through reward optimization
- **Protocol**: Two-pass inference (diagnostic probe → adaptive teaching)
- **Reward**: Zero for degradation, positive for improvements with efficiency bonus
- **vLLM Integration**: High-throughput generation during policy gradient updates

## Monitoring Progress

### Training Metrics
Monitor these key metrics during RL training:
- **Reward trends**: Should increase over time
- **Non-degradation rate**: Target ≥99% non-harmful interactions
- **Teaching efficiency**: Performance gain per teaching token
- **GPU utilization**: Balanced across vLLM and training processes

### Log Locations
```bash
# Training logs and checkpoints
ls results/

# Weights & Biases dashboard (if enabled)
# View at: https://wandb.ai/your-project

# TensorBoard logs
tensorboard --logdir results/
```

## Expected Outcomes

After successful completion, your model should demonstrate:
- **+15-30%** accuracy improvements through teaching
- **Near 100%** completion rates vs ~69% student-alone baseline
- **Adaptive guidance**: Minimal hints for strong students, comprehensive support for struggling students
- **Token efficiency**: ~50% reduction in token usage with maintained quality

## Troubleshooting

### vLLM Server Issues
```bash
# Check server health
curl http://localhost:8765/health

# Monitor server logs
tail -f results/latest_run/vllm_server.log
```

### Training Problems
```bash
# Reduce memory usage
scripts/launch_with_server.sh 4 4 configs/run/teacher_rcl.yaml \
  model_name_or_path=path/of/saved/pre_rl_model \
  per_device_train_batch_size=1 \
  offload

# Use alternative port if conflicts
scripts/launch_with_server.sh 4 4 configs/run/teacher_rcl.yaml \
  model_name_or_path=path/of/saved/pre_rl_model \
  vllm_port=8766
```

### Authentication Issues
```bash
# Ensure HuggingFace access
huggingface-cli login

# Test dataset access
python -c "from datasets import load_dataset; print('Dataset accessible')"
```

## Next Steps

After completing your first experiment:
- **Evaluate results**: Use trained model for adaptive teaching inference
- **Deploy production**: Set up vLLM server for API serving
- **Customize training**: Modify configs for your specific datasets and models
- **Scale up**: Use distributed training for larger models

See [Benchmarks](../benchmarks/) for performance comparison and [Deployment](../deployment/) for production setup.

