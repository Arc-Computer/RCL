# ATLAS: Adaptive Training Methodology for RL

> A diagnostic teaching approach that delivers consistent performance improvements in reinforcement learning training pipelines.

<p align="center">
  <b>Adaptive Teaching and Learning Alignment System for Agents</b>
</p>

<div align="center">

[![ATLAS-8B-Thinking](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ATLAS--8B--Thinking-blue)](https://huggingface.co/Arc-Intelligence/ATLAS-8B-Thinking)
[![ATLAS-8B-Instruct](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ATLAS--8B--Instruct-blue)](https://huggingface.co/Arc-Intelligence/ATLAS-8B-Instruct)
[![Arc-ATLAS-Teach Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Arc--ATLAS--Teach-green)](https://huggingface.co/datasets/Arc-Intelligence/Arc-ATLAS-Teach-v0)

<img src="public/ATLAS.png" alt="ATLAS - Adaptive Teaching and Learning Alignment System" width="1000" style="border-radius: 12px;">

</div>

---

ATLAS delivers consistent **15.7% accuracy improvements** and **31% completion gains** across any base model through diagnostic probing and conditional teaching. The framework works with **any student model (open or closed source)** — train your own teacher or use our pre-trained **[ATLAS-8B-Thinking](https://huggingface.co/Arc-Intelligence/ATLAS-8B-Thinking)** and **[ATLAS-8B-Instruct](https://huggingface.co/Arc-Intelligence/ATLAS-8B-Instruct)** models.

📄 **[Read the ATLAS Technical Report](docs/ATLAS-Technical-Report.pdf)** for comprehensive methodology, ablation studies, and detailed performance analysis.

## Student Performance Results

Our evaluation demonstrates that ATLAS-trained teachers consistently improve student model performance across multiple metrics with verified reliability.

<div align="center">
<img src="public/performance-chart.png" alt="Student Performance: ATLAS Teacher+Student vs Student Alone - showing accuracy gains and completion rates" width="700" style="border-radius: 12px;">
</div>

<div align="center">

| Metric | Teacher+Student | Student Alone | Improvement |
|--------|----------------|---------------|-------------|
| **Average accuracy** | +15.7% | baseline | **+15.7%** |
| **Maximum improvement** | +29.6% | - | **+29.6%** |
| **Completion rate** | ~100% | ~69% | **+31%** |
| **Response efficiency** | ~2k tokens | ~4k tokens | **-50%** |
| **Generation time** | ~1:10 | ~1:21 | **-13.6%** |

</div>

The system achieves a 97% non-degradation rate with consistent improvements across problem difficulty levels. The framework includes the complete ATLAS training pipeline with SFT to RL workflow using GRPO optimization, the adaptive teaching protocol implementing two-pass diagnostic probing with conditional teaching, production-ready vLLM server architecture with distributed training support, and pre-trained [ATLAS-8B-Thinking](https://huggingface.co/Arc-Intelligence/ATLAS-8B-Thinking) and [ATLAS-8B-Instruct](https://huggingface.co/Arc-Intelligence/ATLAS-8B-Instruct) teacher models with Arc-ATLAS teaching datasets.

---

## Adaptive Teaching Protocol

The protocol achieves reliable performance improvements through diagnostic probing followed by capability-adapted teaching.

<div align="center">
<img src="public/adaptive-teaching.png" alt="ATLAS Two-Pass Inference Protocol - Diagnostic probing followed by adaptive teaching based on capability assessment" width="700" style="border-radius: 12px;">
</div>

The protocol operates in two phases. First, diagnostic probing allows the teacher to assess student understanding through minimal interaction, revealing capability levels without requiring complete solutions. Second, adaptive teaching provides conditional guidance tailored to the diagnosed capability—strong students receive minimal intervention to avoid degradation while weak students receive comprehensive scaffolding and support. The reward system assigns zero reward for performance degradation and positive rewards for improvements with efficiency bonuses, encouraging helpful teaching while preventing harmful interventions.


---

<div align="center">
<img src="public/Training-Pipeline.png" alt="ATLAS Training Pipeline - Two-phase SFT to RL workflow with adaptive teaching protocol" width="700" style="border-radius: 12px;">
</div>

---

## Installation

Conda is recommended for environment management. The repository and models have been validated with Python 3.11 and Python 3.12. Use the appropriate installation script to replicate the environment:

**Python 3.11:**
```sh
bash scripts/install_py311.sh
```

**Python 3.12:**
```sh
bash scripts/install_py312.sh
```
---

## Quickstart

- Examples: Quick inference demos and Colab notebooks in `examples/README.md`
- Colab: Math demo and Code demo
  - Math: https://colab.research.google.com/github/Arc-Computer/ATLAS/blob/main/examples/math_reasoning_demo.ipynb
  - Code: https://colab.research.google.com/github/Arc-Computer/ATLAS/blob/main/examples/code_generation_demo.ipynb

Run minimal smoke tests on a single GPU to validate your setup.

**SFT warmup (1 epoch):**

```bash
scripts/launch.sh 1 configs/run/teacher_sft.yaml report_to=null save_final_model=false num_train_epochs=1
```

**RL with vLLM (4 steps):**

```bash
scripts/launch_with_server.sh 1 1 configs/run/teacher_rcl.yaml report_to=null max_steps=4 eval_steps=1
```

The `teacher_rcl.yaml` configuration uses `use_vllm_server: true` by default, while `launch_with_server.sh` handles `vllm_host` setup and `vllm_port` validation. Adding `offload` to commands reduces GPU memory usage.

---

## Training Pipeline

ATLAS uses a two-phase SFT→RL pipeline managed via [Hydra](https://hydra.cc/) configs. Training is scalable from single GPU to distributed setups with DeepSpeed.

Basic training follows a two-phase approach:

```sh
# SFT training
scripts/launch.sh ${NUM_OF_GPUS} configs/run/teacher_sft.yaml ${hydra_args}

# RL training with vLLM server
scripts/launch_with_server.sh ${NUM_VLLM_GPUS} ${NUM_TRAINING_GPUS} configs/run/teacher_rcl.yaml ${hydra_args}
```

Key parameters include `dataset_id_or_path` and `model_name_or_path`. Memory optimization can be achieved by adding `offload` to commands. All results are saved to `results/`.

See [docs/guides/distributed-training.md](docs/guides/) for multi-GPU setup and [docs/guides/rl-training.md](docs/guides/) for detailed RL parameters.

Production training on 8×H100 infrastructure:

```sh
# Phase 1: SFT Warmup
scripts/launch.sh 8 configs/run/teacher_sft.yaml output_dir=path/to/save/pre_rl_model

# Phase 2: RL Training (2-3 days)
scripts/launch_with_server.sh 4 4 configs/run/teacher_rcl.yaml \
  model_name_or_path=path/of/saved/pre_rl_model
```

The system defaults to the [ATLAS-8B-Thinking](https://huggingface.co/Arc-Intelligence/ATLAS-8B-Thinking) teacher model with `Arc-Intelligence/Arc-ATLAS-Teach-v0` for both SFT and RL phases. Custom datasets require `question` and `solution` columns. Detailed formatting specifications are available in [docs/guides/data-requirements.md](docs/guides/).

See [docs/concepts/adaptive-teaching.md](docs/concepts/) for detailed protocol and reward design.

Access to models and datasets requires `hf auth login`. W&B logging can be disabled by setting `report_to: null`. Setup details are available in [docs/getting-started/](docs/getting-started/).

## Core Concepts

**Teacher-Student Architecture**: ATLAS trains teacher models that can assess any student model's capability and provide conditional guidance. Teachers learn through GRPO optimization to maximize student performance improvements while maintaining reliability.

**Diagnostic Probing**: Teachers assess student understanding through minimal interaction before providing guidance, enabling capability-adapted teaching without requiring full problem solutions.

**Adaptive Teaching**: Based on diagnosed capability, teachers provide targeted instruction - minimal intervention for strong students to prevent degradation, comprehensive scaffolding for weak students to maximize learning gains.

**Training Methodology**: Two-phase SFT→RL pipeline where teachers learn optimal teaching strategies through reward signals based on student performance improvements and efficiency metrics.

## Config System

Run recipes in `configs/run/*.yaml` contain complete experiment configurations built from modular components in `configs/{data,model,trainer}/`. Common parameter overrides include `dataset_id_or_path=my/custom/dataset` and `model_name_or_path=my/base/model`.

Reward configuration through `configs/trainer/reward/adaptive_teaching.yaml` controls teaching behavior via `efficiency_weight` and probe token limits. The reward system uses zero rewards for degradation and positive rewards for improvements. Detailed reward design principles are documented in [docs/concepts/](docs/concepts/).

## Project Structure

- `train.py`: Main entry point for all experiments
- `configs/`: Modular Hydra configurations (run recipes, data, models, trainers)
- `accelerate/`: DeepSpeed configurations (zero1, zero3, cpu offloading)
- `trainers/`: Core training logic including GRPO, teacher rewards, vLLM integration
- `custom_data/`: Dataset handlers and formatting (see `datasets_info.py`)
- `scripts/`: Installation and utility scripts
- `scripts/launch.sh`, `scripts/launch_with_server.sh`: Shell entry points for single and distributed training

## Troubleshooting

Server health can be verified with `curl http://$vllm_host:$vllm_port/health`. Port conflicts can be resolved by setting `vllm_port=8766` if the default 8765 is occupied. Authentication errors typically require running `huggingface-cli login`. Memory issues can be addressed by adding `offload` to commands, reducing `per_device_train_batch_size`, or using `accelerate/deepspeed_zero1.yaml`.

For comprehensive troubleshooting, installation guides, and deployment instructions, see [docs/](docs/).

Results can be reproduced using:
```bash
# SFT Warmup  
scripts/launch.sh 4 configs/run/teacher_sft.yaml dataset_id_or_path=Arc-Intelligence/Arc-ATLAS-Teach-v0

# RL Training
scripts/launch_with_server.sh 1 3 configs/run/teacher_rcl.yaml \
  model_name_or_path=results/pre_rl_model \
  dataset_id_or_path=Arc-Intelligence/Arc-ATLAS-Teach-v0 num_generations=32
```

**Environment**: 4×H100 GPUs, [Arc-Intelligence/Arc-ATLAS-Teach-v0](https://huggingface.co/datasets/Arc-Intelligence/Arc-ATLAS-Teach-v0) dataset, seed=42  
**Models**: [ATLAS-8B-Thinking](https://huggingface.co/Arc-Intelligence/ATLAS-8B-Thinking), [ATLAS-8B-Instruct](https://huggingface.co/Arc-Intelligence/ATLAS-8B-Instruct)

See [docs/benchmarks/](docs/benchmarks/) for complete methodology and detailed performance analysis.

## Citation

If you find our work or this repository useful and want to cite our work, you can use the following:

```bibtex
@article{atlas2025,
  title     = {ATLAS: Adaptive Training Methodology for RL},
  author    = {Arc Intelligence},
  journal   = {arXiv preprint},
  year      = {2025}
}
```
