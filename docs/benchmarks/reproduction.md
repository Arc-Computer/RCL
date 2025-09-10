
# Reproducing Benchmark Results

## Complete Reproduction Steps

To reproduce the performance results from the README:

### Phase 1: SFT Warmup

```bash
scripts/launch.sh 4 configs/run/teacher_sft.yaml \
  dataset_id_or_path=Arc-Intelligence/Arc-ATLAS-Teach-v0 \
  output_dir=results/pre_rl_model
```

### Phase 2: RL Training

```bash
scripts/launch_with_server.sh 1 3 configs/run/teacher_rcl.yaml \
  model_name_or_path=results/pre_rl_model \
  dataset_id_or_path=Arc-Intelligence/Arc-ATLAS-Teach-v0 \
  num_generations=32
```

## Environment Specifications

**Hardware**:
- 4×H100 GPUs
- High-bandwidth GPU interconnect
- Sufficient CPU and memory for data processing

**Software**:
- Python 3.11
- PyTorch 2.6.0
- vLLM 0.8.3
- CUDA compatible with PyTorch 2.6.0

**Configuration**:
- seed=42 (set in configs for reproducibility)
- Dataset: Arc-Intelligence/Arc-ATLAS-Teach-v0
- Models: ATLAS-8B-Thinking, ATLAS-8B-Instruct

## Expected Metrics

After training completion, expect:
- **+15.73%** average accuracy improvement
- **+31%** completion rate improvement  
- **-50%** reduction in average token length
- **-13.6%** faster generation time

## Validation Commands

Check training progress and final results:

```bash
# Monitor training logs
tensorboard --logdir results/

# Validate model performance
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Check vLLM server health during training
curl http://localhost:8765/health
```

## Troubleshooting

**Memory Issues**: Add `offload` to launch commands
**Port Conflicts**: Override with `vllm_port=8766`
**Authentication**: Ensure `huggingface-cli login` is completed

