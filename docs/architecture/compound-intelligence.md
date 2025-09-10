
# Compound Intelligence Architecture

Compound Intelligence is RCL's architectural vision for persistent, continuously learning agent systems. ATLAS serves as the foundational outer loop training component implementing adaptive teaching capabilities.

## Current Implementation (ATLAS)

RCL currently implements the offline RL component of Compound Intelligence:

- **Adaptive Teaching Models**: Trained via SFT→RL (GRPO) pipeline
- **Two-Pass Protocol**: Diagnostic probing followed by capability-adapted teaching
- **Non-Degradation Guarantee**: Zero reward for performance degradation
- **vLLM Integration**: Production-ready inference server for distributed training

## Architecture Components

### Training Pipeline
- **Phase 1**: Supervised fine-tuning (SFT) warmup
- **Phase 2**: Reinforcement learning with Group Relative Policy Optimization (GRPO)
- **Reward System**: AdaptiveTeachingReward with degradation penalties and efficiency bonuses

### Inference System
- **Teacher Models**: Probe student capability and provide adaptive guidance
- **vLLM Server**: High-throughput generation with FastAPI endpoints
- **Client Integration**: NCCL-based weight synchronization during training

### Configuration Management
- **Hydra Configs**: Modular composition from `configs/{run,model,data,trainer}/`
- **DeepSpeed Integration**: ZeRO-1/3 optimization with CPU offloading support
- **Parameter Overrides**: Command-line configuration of training parameters

## Future Vision (Roadmap)

The complete Compound Intelligence framework will extend ATLAS with:

- **Persistent Memory**: Organizational knowledge retention across agent interactions
- **Online Learning**: Real-time adaptation based on user interactions
- **Cross-Agent Transfer**: Skill sharing between agents within organizations
- **Continuous Improvement**: Automated capability enhancement through verified learning loops

## Implementation Status

**Stable**: ATLAS training, vLLM server integration, Hydra configuration system
**Roadmap**: Online learning loops, persistent organizational memory, cross-agent knowledge transfer