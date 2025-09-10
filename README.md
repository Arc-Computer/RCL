# Reinforced Continual Learning (RCL) — Compound Intelligence for LLM Agents

<div align="center">

[![ATLAS-8B-Thinking](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ATLAS--8B--Thinking-blue)](https://huggingface.co/Arc-Intelligence/ATLAS-8B-Thinking)
[![ATLAS-8B-Instruct](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ATLAS--8B--Instruct-blue)](https://huggingface.co/Arc-Intelligence/ATLAS-8B-Instruct)
[![Arc-ATLAS-Teach Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Arc--ATLAS--Teach-green)](https://huggingface.co/datasets/Arc-Intelligence/Arc-ATLAS-Teach-v0)

<img src="public/ATLAS.png" alt="ATLAS - Adaptive Teaching and Learning Alignment System" width="1000">

</div>

This repository provides **production-ready code for training adaptive teacher models** that diagnose student capability and provide tailored guidance without performance degradation. Built on GRPO (Group Relative Policy Optimization) with distributed vLLM integration.

**What's included:**
- **ATLAS Training Pipeline**: Complete SFT → RL workflow with GRPO optimization
- **Adaptive Teaching Protocol**: Two-pass diagnostic probing + conditional teaching
- **Production Integration**: vLLM server architecture with distributed training support  
- **Pre-trained Models**: ATLAS-8B teacher models and Arc-ATLAS teaching datasets

**Technical Innovation**: Instead of static demonstrations, ATLAS models first probe student understanding (≤50 tokens), then adapt their teaching approach—providing minimal hints for capable students and comprehensive support for struggling ones. This prevents the common problem of helpful models accidentally degrading strong student performance.

**Roadmap**: This adaptive teaching system serves as the foundation for our broader **Compound Intelligence** framework—combining persistent memory with online learning loops for continuously improving agent systems.

[Diagram Placeholder: ATLAS system architecture — SFT→RL training pipeline with vLLM server integration and adaptive teaching protocol]

## Quickstart

Run minimal smoke tests on a single GPU to validate your setup.

**SFT warmup (1 epoch):**

```bash
./launch.sh 1 configs/run/teacher_sft.yaml report_to=null save_final_model=false num_train_epochs=1
```

**RL with vLLM (4 steps):**

```bash
./launch_with_server.sh 1 1 configs/run/teacher_rcl.yaml report_to=null max_steps=4 eval_steps=1
```

**Notes:**
- `teacher_rcl.yaml` uses `use_vllm_server: true` by default
- `launch_with_server.sh` sets `vllm_host` and validates `vllm_port`
- Add `offload` at the end of commands to reduce GPU memory usage

## Installation

We recommend using **Conda** for managing environments. Our repository and models have been tested with **Python 3.11**. To replicate our Conda environment, use the following installation script:

```sh
scripts/install_08.sh
```

Otherwise, you can install all our dependencies in your own custom environment:

```sh
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install vllm==0.8.3 tensorboard
python -m pip install flash-attn --no-build-isolation
python -m pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.6/

python -m pip install --upgrade -r requirements_08.txt
```

## Training Pipeline

RCL uses a two-phase SFT→RL pipeline managed via [Hydra](https://hydra.cc/) configs. Training is scalable from single GPU to distributed setups with DeepSpeed.

[Diagram Placeholder: RCL Training Pipeline — SFT warmup followed by RL/GRPO training with vLLM server integration]

**Basic Training:**
```sh
# Without vLLM generation
./launch.sh ${NUM_OF_GPUS} configs/run/config.yaml ${hydra_args}

# With vLLM server (for RL training)  
./launch_with_server.sh ${NUM_VLLM_GPUS} ${NUM_TRAINING_GPUS} configs/run/config.yaml ${hydra_args}
```

**Key Parameters:** `degradation_penalty_multiplier`, `dataset_id_or_path`, `model_name_or_path`. Add `offload` for memory optimization. Results save to `results/`.

See [docs/guides/distributed-training.md](docs/guides/) for multi-GPU setup and [docs/guides/rl-training.md](docs/guides/) for detailed RL parameters.

**Production Training (8×H100):**
```sh
# Phase 1: SFT Warmup
./launch.sh 8 configs/run/teacher_sft.yaml output_dir=path/to/save/pre_rl_model

# Phase 2: RL Training (2-3 days)
./launch_with_server.sh 4 4 configs/run/teacher_rcl.yaml \
  model_name_or_path=path/of/saved/pre_rl_model
```

**Datasets:** Defaults to 8B teacher model with `Arc-Intelligence/Arc-ATLAS-Teach-v0` for both SFT and RL phases. Custom datasets need `question`, `solution` columns. See [docs/guides/data-requirements.md](docs/guides/) for formatting details.

## Adaptive Teaching Protocol

RCL's core innovation: diagnostic probing followed by capability-adapted teaching to ensure non-degradation.

[Diagram Placeholder: ATLAS two-pass inference protocol — probe phase for capability diagnosis followed by adaptive teaching phase]

**Pass 1: Diagnostic Probing** - Teacher probes student understanding with minimal interaction (≤50 tokens) to reveal capability level without requiring full solutions.

**Pass 2: Adaptive Teaching** - Conditional teaching tailored to diagnosed capability:
- Strong students: Minimal intervention to avoid degradation  
- Weak students: Comprehensive scaffolding and support

**Reward System** - 0.0 reward for performance degradation, positive rewards for improvements with efficiency bonuses. Encourages helpful teaching while avoiding harmful interventions.

See [docs/concepts/adaptive-teaching.md](docs/concepts/) for detailed protocol and reward design.

### Learning Metrics

RCL introduces metrics to measure learning itself:
- **Learning Rate (LR)**: Performance change per interaction
- **Non-Degradation Rate (NDR)**: Interactions that don't hurt performance (target: ≥99%)  
- **Teaching Efficiency Score (TES)**: Performance gain per teaching token

**Student Training & Evaluation:** Students trained on adaptive teaching outputs with automatic complexity adjustment based on model size. See [docs/concepts/evaluation.md](docs/concepts/) for complete protocols.

**Prerequisites:** `huggingface-cli login` required for model/dataset access. Set `report_to: null` to disable W&B logging. See [docs/getting-started/](docs/getting-started/) for setup details.

## Concepts

**RCL**: Outer loop architecture for Compound Intelligence. Two-phase SFT→RL (GRPO) training creates teachers that diagnose student capability and provide adaptive teaching without harmful interventions. Forms the foundation for persistent, continuously learning agent systems.

**ATLAS**: **A**daptive **T**eaching and **L**earning **A**lignment **S**ystem — teacher model family and inference pipeline implementing the Arc-ATLAS datasets. Uses two-pass inference protocol: probe student understanding, then deliver conditional teaching based on diagnosed capability.

**vLLM Integration**: FastAPI server endpoints (`/health`, `/generate`, `/init_communicator`) with client wiring into GRPO and Teacher trainers. See [docs/architecture/](docs/architecture/) for details.

**Hydra Configs**: Entry point via `train.py`, outputs under `results/`. Configuration system supports modular building blocks for experiments.

## Config System

**Run recipes**: `configs/run/*.yaml` contain complete experiment configurations using building blocks from `configs/{data,model,trainer}/`.

**Common overrides**: 
- `dataset_id_or_path=my/custom/dataset` 
- `model_name_or_path=my/base/model`

**Reward configuration**: `configs/trainer/reward/adaptive_teaching.yaml` controls teaching behavior via `degradation_penalty_multiplier`, `efficiency_weight`, and probe token limits. See [docs/concepts/](docs/concepts/) for detailed reward design.

## Project Structure

- `train.py`: Main entry point for all experiments
- `configs/`: Modular Hydra configurations (run recipes, data, models, trainers)
- `accelerate_configs/`: DeepSpeed configurations (zero1, zero3, cpu offloading)
- `trainers/`: Core training logic including GRPO, teacher rewards, vLLM integration
- `custom_data/`: Dataset handlers and formatting (see `datasets_info.py`)
- `scripts/`: Installation and utility scripts
- `launch.sh`, `launch_with_server.sh`: Shell entry points for single and distributed training

## Troubleshooting

**vLLM Health**: Check server status with `curl http://$vllm_host:$vllm_port/health`

**Port Conflicts**: Override default port via `vllm_port=8766` if 8765 is occupied

**HF Authentication**: Run `huggingface-cli login` if encountering download errors

**OOM Issues**: Add `offload` to commands, reduce `per_device_train_batch_size`, or use `accelerate_configs/zero1.yaml`

For comprehensive troubleshooting, installation guides, and deployment instructions, see [docs/](docs/).

## Status & Roadmap

**Stable (Offline RL)**: ATLAS SFT+RL training, vLLM server integration, Hydra configuration system

**[Unverified/Roadmap] (Online + Persistent Memory)**: Full Compound Intelligence framework with persistent organizational memory, online learning loops, and cross-agent knowledge transfer. ATLAS serves as the foundational outer loop training component for continuously learning agent systems.

[Diagram Placeholder: Compound Intelligence framework — unified architecture combining persistent memory with offline RL (ATLAS) and online learning loops]

## Performance Results

Verified results from our latest training run demonstrate non-degradation teaching with significant efficiency gains.

[Diagram Placeholder: Performance metrics visualization showing accuracy gains across problem difficulty levels]

**Environment**: 4×H100 GPUs, [Arc-Intelligence/Arc-ATLAS-Teach-v0](https://huggingface.co/datasets/Arc-Intelligence/Arc-ATLAS-Teach-v0) dataset, seed=42
**Models**: [ATLAS-8B-Thinking](https://huggingface.co/Arc-Intelligence/ATLAS-8B-Thinking), [ATLAS-8B-Instruct](https://huggingface.co/Arc-Intelligence/ATLAS-8B-Instruct)

| Metric | Teacher+Student | Student Alone | Delta |
|--------|----------------|---------------|-------|
| Average accuracy | +0.1573 | baseline | **+15.73%** |
| Max per-item improvement | +0.296 | - | **+29.6%** |
| Token efficiency | 0.372 | - | (avg. efficiency term) |
| Generation time (32 samples) | ~1:10 | ~1:21 | **-13.6%** |
| Average length | ~2k tokens | ~4k tokens | **-50%** |
| Completion rate | ~100% | ~69% | **+31%** |

**Key Findings**: Teacher+student achieves higher accuracy with fewer tokens. Near-100% completion rate vs. ~69% student-alone. Consistent improvements across problem difficulty levels.

[Diagram Placeholder: Adaptive teaching examples showing different guidance levels — minimal hints for strong students, comprehensive support for struggling students]

**Reproduce Results:**
```bash
# SFT Warmup  
./launch.sh 4 configs/run/teacher_sft.yaml dataset_id_or_path=Arc-Intelligence/Arc-ATLAS-Teach-v0

# RL Training
./launch_with_server.sh 1 3 configs/run/teacher_rcl.yaml \
  model_name_or_path=results/pre_rl_model \
  dataset_id_or_path=Arc-Intelligence/Arc-ATLAS-Teach-v0 num_generations=32
```

See [docs/benchmarks/](docs/benchmarks/) for complete methodology and [docs/benchmarks/reproduction.md](docs/benchmarks/) for detailed reproduction steps.

## Citation

If you find our work or this repository useful and want to cite our work, you can use the following:

```bibtex
@article{rcl2025,
  title     = {Reinforcement Collaborative Learning: Adaptive Teaching at Scale},
  author    = {Author Names},
  journal   = {arXiv preprint},
  year      = {2025}
}
```
