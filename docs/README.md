# ATLAS Documentation

ATLAS (Adaptive Training and Learning Alignment System) is a framework for training teacher models that adapt to student capabilities without degrading performance. This documentation provides implementation details, configuration guides, and API references for using ATLAS in production environments.

## Overview

ATLAS implements adaptive learning through two approaches:

### Inference-Only (Use Pre-trained Models)
Use our pre-trained ATLAS teachers to enhance any existing agent:
- **No training required** - Just load and use
- **Model-agnostic** - Works with any LLM (GPT, Claude, Llama, etc.)
- **15.7% improvement** - Immediate accuracy gains
- **Drop-in replacement** - Minimal code changes

### Training Pipeline (Build Custom Teachers)
Train your own ATLAS teacher models:
1. **Supervised Fine-tuning (SFT)**: Initial warmup phase
2. **Reinforcement Learning (RL/GRPO)**: Adaptive learning optimization with vLLM server integration

The system uses a diagnostic probing protocol where teachers first assess student capability, then provide calibrated guidance to improve performance without harmful interventions.

## Quick Integration

Transform your existing agent into an ATLAS-enhanced system with minimal code changes:

### Option 1: Direct Integration (No Training Required)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from examples.utils.atlas_inference import ATLASInference

# Load your existing agent
student_model = AutoModelForCausalLM.from_pretrained("your-model-name")
student_tokenizer = AutoTokenizer.from_pretrained("your-model-name")

# Load ATLAS teacher (requires trust_remote_code for custom architecture)
teacher_model = AutoModelForCausalLM.from_pretrained(
    "Arc-Intelligence/ATLAS-8B-Thinking",
    trust_remote_code=True
)
teacher_tokenizer = AutoTokenizer.from_pretrained(
    "Arc-Intelligence/ATLAS-8B-Thinking",
    trust_remote_code=True
)

# Initialize ATLAS protocol
atlas = ATLASInference(
    student_model=student_model,
    student_tokenizer=student_tokenizer,
    teacher_model=teacher_model,
    teacher_tokenizer=teacher_tokenizer,
    probe_token_limit=500 
)

# Run complete two-pass protocol
result = atlas.run_full_protocol("Your problem here")

# Access enhanced response with teacher guidance
guided_response = result["guided_response"]  # Student with ATLAS teacher guidance
```

### Option 2: Production API Integration (Example)
```python
import requests

# Deploy ATLAS teacher as API endpoint (example deployment pattern)
ATLAS_API = "https://your-atlas-server.com/v1/enhance"

def your_agent_with_atlas(user_input):
    # Get ATLAS enhancement
    response = requests.post(ATLAS_API, json={"prompt": user_input})
    guidance = response.json()["guidance"]
    
    # Your agent uses the guidance
    return your_agent.generate(f"{user_input}\n{guidance}")
```

### Option 3: Train Your Own ATLAS Teacher

**Prerequisites**: Python 3.11 or 3.12, and authenticated Hugging Face access. See requirements files for package versions.

**SFT Training**:
```bash
scripts/launch.sh 1 configs/run/teacher_sft.yaml report_to=null save_final_model=false num_train_epochs=1
```

**RL Training with vLLM**:
```bash
scripts/launch_with_server.sh 1 1 configs/run/teacher_rcl.yaml report_to=null max_steps=4 eval_steps=1
```

## Architecture

- **Entry Point**: `train.py` with Hydra configuration management
- **Training**: SFT→RL pipeline using GRPO with adaptive learning rewards
- **Generation**: vLLM server integration for distributed inference
- **Configuration**: Modular YAML configs in `configs/` directory
- **Data**: Custom dataset handlers in `custom_data/`

## Core Concepts

**Adaptive Learning Protocol**:
- Pass 1: Diagnostic probing (≤50 tokens) to assess student capability
- Pass 2: Conditional guidance tailored to diagnosed capability level

**Reward Design**:
- Zero reward for performance degradation
- Positive rewards for improvements with efficiency bonuses
- Configurable via `efficiency_weight`

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
- [Online Learning](guides/online-learning.md) - GEPA prompt evolution and compatibility wrappers
- [Model Selection](guides/model-selection.md) - Base model configuration
- [Data Requirements](guides/data-requirements.md) - Dataset preparation
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
- [Compound Intelligence](architecture/compound-intelligence.md) - Vision and roadmap

### Concepts
- [Adaptive Learning](concepts/adaptive-learning.md) - Two-pass diagnostic protocol and learning strategy
- [Reward Design](concepts/reward-design.md) - Learning effectiveness metrics and reward structure

### Deployment
- [Inference](deployment/inference.md) - Production deployment patterns
- [Serving](deployment/serving.md) - vLLM server configuration
- [Integration](deployment/integration.md) - API integration guidelines

### Troubleshooting
- [Debugging](troubleshooting/debugging.md) - Validation tests and verbose logging

### Benchmarks
- [Results](benchmarks/results.md) - Performance metrics and comparisons
- [Evaluation](benchmarks/evaluation.md) - Testing protocols
- [Reproduction](benchmarks/reproduction.md) - Replicating published results

## Configuration

ATLAS uses Hydra for modular configuration management:

- `configs/run/` - Complete experiment configurations
- `configs/model/` - Model-specific settings  
- `configs/data/` - Dataset configurations
- `configs/trainer/` - Training parameters and reward functions

Override parameters from command line:
```bash
scripts/launch.sh 8 configs/run/teacher_sft.yaml learning_rate=5e-6 output_dir=custom/path
```

## Performance Metrics

Key evaluation metric:
- **Non-Degradation Rate (NDR)**: Percentage of non-harmful interactions (97% achieved)

## Support

For implementation questions, refer to the troubleshooting guides or examine the configuration examples in `configs/run/`.