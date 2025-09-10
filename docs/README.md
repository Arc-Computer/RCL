# RCL Documentation

Reinforced Continual Learning (RCL) is a system for training teacher models that adapt to student capabilities without degrading performance. This documentation provides implementation details, configuration guides, and API references for using RCL in production environments.

## Overview

RCL implements Adaptive Teaching and Learning Alignment System (ATLAS) through a two-phase training pipeline:

1. **Supervised Fine-tuning (SFT)**: Initial warmup phase
2. **Reinforcement Learning (RL/GRPO)**: Adaptive teaching training with vLLM server integration

The system uses a diagnostic probing protocol where teachers first assess student capability, then provide calibrated guidance to improve performance without harmful interventions.

## Quick Start

**Prerequisites**: Python 3.11, PyTorch 2.6.0, vLLM 0.8.3, and authenticated Hugging Face access.

**SFT Training**:
```bash
./launch.sh 1 configs/run/quickstart_sft.yaml report_to=null save_final_model=false
```

**RL Training with vLLM**:
```bash
./launch_with_server.sh 1 1 configs/run/quickstart_rcl.yaml report_to=null max_steps=4 eval_steps=1
```

## Architecture

- **Entry Point**: `train.py` with Hydra configuration management
- **Training**: SFT→RL pipeline using GRPO with adaptive teaching rewards
- **Generation**: vLLM server integration for distributed inference
- **Configuration**: Modular YAML configs in `configs/` directory
- **Data**: Custom dataset handlers in `custom_data/`

## Core Concepts

**Adaptive Teaching Protocol**:
- Pass 1: Diagnostic probing (≤500 tokens) to assess student capability
- Pass 2: Conditional teaching tailored to diagnosed capability level

**Reward Design**:
- Zero reward for performance degradation
- Positive rewards for improvements with efficiency bonuses
- Configurable via `degradation_penalty_multiplier` and `efficiency_weight`

**vLLM Integration**:
- FastAPI server endpoints for health checks and generation
- Client integration with GRPO and Teacher trainers
- Distributed training support with NCCL communication

## Documentation Structure

### Getting Started
- [Installation](getting-started/installation.md) - Environment setup and dependencies
- [Quickstart](getting-started/quickstart.md) - Basic training examples
- [Testing Installation](getting-started/testing-installation.md) - Verification procedures
- [First Experiment](getting-started/first-experiment.md) - Complete workflow example

### Guides  
- [Training Pipeline](guides/training-pipeline.md) - SFT→RL workflow
- [Model Selection](guides/model-selection.md) - Base model configuration
- [Data Requirements](guides/data-requirements.md) - Dataset preparation
- [Custom Datasets](guides/custom-datasets.md) - Data format specifications
- [Distributed Training](guides/distributed-training.md) - Multi-GPU setup
- [RL Training](guides/rl-training.md) - GRPO configuration details

### API Reference
- [Trainers](api-reference/trainers.md) - TeacherGRPOTrainer and GRPOTrainer classes
- [Configurations](api-reference/configs.md) - GRPOConfig and Hydra structure
- [Data Builders](api-reference/data-builders.md) - Dataset processing functions
- [Utilities](api-reference/utils.md) - Helper functions and DeepSpeed integration

### Architecture
- [Overview](architecture/overview.md) - System components and data flow
- [Config System](architecture/config-system.md) - Hydra configuration structure
- [vLLM Integration](architecture/vllm-integration.md) - Server and client architecture
- [Components](architecture/components.md) - Core modules and interfaces

### Concepts
- [Adaptive Teaching](concepts/adaptive-teaching.md) - Two-pass protocol details
- [Reward Design](concepts/reward-design.md) - Teaching effectiveness metrics
- [Two-Pass Protocol](concepts/two-pass-protocol.md) - Implementation specifics

### Deployment
- [Inference](deployment/inference.md) - Production deployment patterns
- [Serving](deployment/serving.md) - vLLM server configuration
- [Integration](deployment/integration.md) - API integration guidelines

### Troubleshooting
- [Common Issues](troubleshooting/common-issues.md) - Frequent problems and solutions
- [Performance](troubleshooting/performance.md) - Optimization and memory management
- [Debugging](troubleshooting/debugging.md) - Diagnostic procedures

### Benchmarks
- [Results](benchmarks/results.md) - Performance metrics and comparisons
- [Evaluation](benchmarks/evaluation.md) - Testing protocols
- [Reproduction](benchmarks/reproduction.md) - Replicating published results

## Configuration

RCL uses Hydra for modular configuration management:

- `configs/run/` - Complete experiment configurations
- `configs/model/` - Model-specific settings  
- `configs/data/` - Dataset configurations
- `configs/trainer/` - Training parameters and reward functions

Override parameters from command line:
```bash
./launch.sh 8 configs/run/teacher_sft.yaml learning_rate=5e-6 output_dir=custom/path
```

## Performance Metrics

Key evaluation metrics for teaching effectiveness:
- **Learning Rate (LR)**: Performance change per interaction
- **Non-Degradation Rate (NDR)**: Percentage of non-harmful interactions
- **Teaching Efficiency Score (TES)**: Performance gain per teaching token

## Support

For implementation questions, refer to the troubleshooting guides or examine the configuration examples in `configs/run/`.