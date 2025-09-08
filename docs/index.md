---
title: RCL - Add Learning to Your Agent in 15 Lines
description: Make agents learn from experience. Not memory. Not fine-tuning. Learning.
---

# Add Learning to Your Agent in 15 Lines

RCL enables **Compound Intelligence**—agents that learn from every interaction, never repeat mistakes, and share skills across your organization.

```python
# Current approach: Train adaptive teaching models
from trainers import TeacherGRPOTrainer
from trainers.teacher_rewards import AdaptiveTeachingReward

# Configure adaptive teaching
config = GRPOConfig(
    model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
    num_generations=8,
    max_probe_tokens=50,  # Brief diagnostic
    degradation_penalty_multiplier=2.0
)

# Train teacher that adapts to student capability
trainer = TeacherGRPOTrainer(
    config=config,
    reward_function=AdaptiveTeachingReward()
)

# Future: SDK for production integration (in development)
```

## Choose Your Path

<div class="grid">

### 🚀 **"I want to train a teacher"**
**Build adaptive teaching models** (2-4 days)
- [Training Pipeline](guides/training-pipeline.md)
- [SFT Warmup](guides/sft-warmup.md)
- [RL Training](guides/rl-training.md)

### 🧪 **"Show me it working"**
**Quick validation run** (5 minutes)
- [Quickstart Guide](getting-started/quickstart.md)
- [First Experiment](getting-started/first-experiment.md)
- [Example Configs](configs/run/)

### 🔬 **"I need custom setup"**
**Advanced configurations** (Advanced)
- [Custom Datasets](guides/custom-datasets.md)
- [Model Selection](guides/model-selection.md)
- [Distributed Training](guides/distributed-training.md)

</div>

## What Makes RCL Different?

| Traditional AI | Compound Intelligence with RCL |
|---------------|--------------------------------|
| Same mistakes forever | Never repeats a learned error |
| Knowledge without experience | Builds procedural expertise |
| Isolated improvements | Skills transfer across teams |
| Static after deployment | Improves with every interaction |

## Training Pipeline in 3 Stages

### Stage 1: SFT Warmup (8-12 hours)
```bash
# Supervised fine-tuning on teaching examples
./launch.sh 8 configs/run/teacher_sft.yaml \
  dataset_id_or_path=bespokelabs/Bespoke-Stratos-17k
```
→ Teacher learns basic instruction patterns

### Stage 2: RL Training (2-3 days)
```bash
# GRPO reinforcement learning with adaptive rewards
./launch_with_server.sh 4 4 configs/run/teacher_rcl.yaml \
  model_name_or_path=results/pre_rl_model
```
→ Teacher learns to diagnose and adapt

### Stage 3: Deployment (Coming Soon)
```python
# Future: Production inference API
# teacher = load_trained_model("results/rcl_teacher")
# response = teacher.teach(student, task)
```
→ Integration SDK in development

## For Your Role

### **ML Engineers** - Advanced Training
- [GRPO Algorithm](api-reference/trainers.md#grpotrainer)
- [Custom Rewards](api-reference/trainers.md#adaptiveteachingreward)
- [Distributed Setup](guides/distributed-training.md)
- See learning curves with W&B integration

### **Developers** - Getting Started
- [Installation Guide](getting-started/installation.md)
- [Quickstart](getting-started/quickstart.md)
- [Configuration System](architecture/config-system.md)
- Train your first model

### **Platform Leads** - Evaluate RCL
- [Resource Requirements](getting-started/for-platform-leads.md)
- [Training Costs](getting-started/for-platform-leads.md#cost-analysis)
- [ROI Calculation](getting-started/for-platform-leads.md#roi-calculation)
- Deployment planning

## Core Concepts

**[Compound Intelligence](architecture/compound-intelligence.md)**: Experience that compounds across agents, time, and teams.

**[Teacher-Student Loop](concepts/adaptive-teaching.md)**: Socratic dialogue that diagnoses gaps and adapts instruction.

**[Verification-Driven Learning](concepts/verification.md)**: Every update traces to a verifiable outcome.

## Getting Started

### Option 1: Quick Validation (5 minutes)
```bash
# Minimal training run to test setup
./launch.sh 4 configs/run/quickstart_sft.yaml
```

### Option 2: Full Training Pipeline (3-4 days)
```bash
# Complete SFT + RL training
bash scripts/install_08.sh
./launch.sh 8 configs/run/teacher_sft.yaml
./launch_with_server.sh 4 4 configs/run/teacher_rcl.yaml
```

### Option 3: Custom Configuration
```bash
# With your own data and models
./launch.sh 8 configs/run/teacher_sft.yaml \
  dataset_id_or_path=your/dataset \
  model_name_or_path=your/model
```

## Documentation Map

- **[Getting Started](getting-started/)** - Installation, quickstart, personas
- **[Concepts](concepts/)** - Adaptive teaching, rewards, protocols
- **[Architecture](architecture/)** - Components, configs, vLLM
- **[Guides](guides/)** - Training pipeline, datasets, distributed
- **[API Reference](api-reference/)** - Trainers, configs, utilities
- **[Troubleshooting](troubleshooting/)** - Common issues and solutions

## Learn More

- [The Era of the Outer Loop](https://arc.computer/blog/outer-loop) - Why learning beats knowing
- [Technical Paper](https://arxiv.org/rcl) - Detailed architecture
- [Discord Community](https://discord.gg/rcl) - Get help and share experiences

---

*The next decade belongs to systems that learn with you. Start compounding today.*

