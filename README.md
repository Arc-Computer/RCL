# Reinforcement Collaborative Learning

This repository implements **Reinforcement Collaborative Learning (RCL)**, a framework for training language models to adaptively teach based on student capability. We provide efficient code to train your own RCL teachers following our adaptive teaching recipe, which is easily extensible for custom datasets and base models. We also provide details on training configurations and best usage practices.

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

## Running experiments

We provide configuration files for training adaptive RCL teachers that diagnose student capabilities and provide tailored teaching to enhance performance without degradation.

Below, we give instructions on performing teacher training with our method and the recommended configuration files in the `configs/run/` folder to train our new class of adaptive collaborative teachers.

Experiments are managed via the amazing [Hydra](https://hydra.cc/) library. Experiments that do not require fast vLLM generation can be run with the following script:

```sh
./launch.sh ${NUM_OF_GPUS} run/file/directory ${extra_hydra_args}
```

RL and other experiments using vLLM can be run with the following script, which first instantiates parallel vLLM servers and later connects with the main trainer class:

```sh
./launch_with_server.sh  ${NUM_OF_VLLM_GPUS} ${NUM_OF_TRAINING_GPUS} run/file/directory ${extra_hydra_args}
```

For example (on 4 total GPUs):

```sh
./launch_with_server.sh 1 3 configs/run/my_run_file.yaml dataset_id_or_path=my/data/path learning_rate=0.0001 ...
```

Additional custom experiments can be conducted by passing extra `hydra` arguments (e.g., `degradation_penalty=2.0`, `dataset_id_or_path=my/data/path`, `model_name_or_path=my/custom/model`) or by modifying the relevant YAML files (e.g., `configs/run/teacher_sft.yaml`).

While we use distributed training with the *DeepSpeed* library and eight H100 GPUs, training should be reproducible even with smaller computational budgets (e.g., four GPUs). You can pass the `offload` argument at the end of any command to automatically activate weight and optimizer offloading to the CPU. By default, checkpoints and results are saved in the `results` folder.

### Instructions on RCL training

We recommend following the two-phase pipeline to train RCL teachers with the example run files provided in `configs/run`, which have been tested on a single compute node with eight H100 GPUs.

For the initial supervised warm-up phase with the RCL format:

```sh
./launch.sh 8 configs/run/teacher_sft.yaml output_dir=path/to/save/pre_rl_model ${extra_hydra_args}
```

This will save a checkpoint for the final RL phase, which can take multiple days:

```sh
./launch_with_server.sh 4 4 configs/run/teacher_rcl.yaml model_name_or_path=path/of/saved/pre_rl_model results_dir=path/to/save/rcl_model ${extra_hydra_args}
```

For detailed performance metrics and results, see [Methodology & Performance](#methodology--performance).

All our scripts and run files currently default to a 7B RCL teacher (`model_name_or_path=Qwen/Qwen2.5-7B-Instruct`) and the `bespokelabs/Bespoke-Stratos-17k` data for benchmarking. Any other custom [dataset](https://huggingface.co/docs/datasets/index) and model can be used by overriding `dataset_id_or_path` or the initial `model_name_or_path` via additional command arguments, or by making separate configuration files (we provide examples in the `configs/data` and `configs/model` folders).

For custom datasets, we assume the columns `question` and `solution` contain each problem's question and solution, respectively. Moreover, an optional extra column `reasoning_trace` can be used to make the data compatible with our SFT warm-up stage. If a custom dataset does not have a `reasoning_trace` entry, we still recommend first performing the SFT warm-up phase on the default `bespokelabs/Bespoke-Stratos-17k` data before performing the RL phase on your custom dataset to obtain the best results. By default, these datasets will be formatted using the think/solution tags and system prompt. To use other tags/system prompts or to extend support to datasets adhering to other custom formats, you can add a new entry to the `DATA_CONFIGS` dictionary defined in `custom_data/reasoning_datasets_info.py`.

### Adaptive Teaching Protocol

Our RCL framework implements a two-pass teaching protocol that adapts to student capabilities:

**Pass 1: Diagnostic Probing** - The teacher probes student understanding with minimal interaction, asking for a brief approach outline (≤50 tokens) to reveal the student's current capability level without requiring a full solution.

**Pass 2: Adaptive Teaching** - Based on the diagnosed capability, the teacher generates conditional teaching tailored to what the student needs. This ensures strong students receive minimal intervention to avoid degradation, while weaker students get comprehensive support.

The reward system heavily penalizes performance degradation (2x penalty) while rewarding improvement with efficiency bonuses for concise teaching. This asymmetric reward structure ensures the teacher learns to be conservative with capable students while providing necessary support to those who need it.

### Student training notes

Our RCL students are trained using the adaptive teaching outputs collected during the RL phase. The framework automatically adjusts teaching complexity based on student model size and capability. For larger students (32B+ parameters), we suggest collecting multiple teaching variations to handle context length constraints while maintaining pedagogical effectiveness.

The evaluation pipeline measures key metrics including Adaptive Performance Gain (APG), Non-Degradation Rate (NDR), and Teaching Efficiency Score (TES). We target NDR ≥ 99% to ensure teaching almost never hurts performance, with APG > 0.1 indicating meaningful improvement across diverse problems.

## Additional notes

Running experiments requires downloading models and datasets hosted on [Hugging Face](https://huggingface.co/). Hence, you must log in to a Hugging Face account with an access token, [as explained here](https://huggingface.co/docs/hub/security-tokens), using the following command:

```sh
huggingface-cli login
```

The default logging functionality saves results both locally and to [Weights & Biases](https://wandb.ai/). To disable Weights & Biases logging, please modify the provided configuration files with:

```yaml
report_to: null
```

## Methodology & Performance

### System Architecture

[Diagram Placeholder: Overall system architecture showing offline RL + online learning + persistent memory pipeline]

Our RCL framework implements a complete adaptive teaching system that learns to diagnose student capability and provide tailored guidance without degradation.

### Methodology

Two-phase pipeline trains an adaptive teacher to improve a student without degradation:

- **SFT Warmup**: Learn structured reasoning/teaching format from curated traces
- **RL (GRPO)**: Two-pass protocol per example:
  - **Probe**: Student provides a ≤50-token approach
  - **Teach**: Teacher adapts guidance to the diagnosed capability
  - **Reward**: Compare student performance with vs. without teaching; apply 2× penalty for degradation and a length-aware efficiency bonus

[Diagram Placeholder: Teacher-student inference pipeline showing the two-pass protocol with probe and teach phases]

### Reward & Constraints

Configured in `configs/trainer/reward/adaptive_teaching.yaml`:

- **Degradation penalty**: 2.0×
- **Efficiency term**: 100/(100 + |teaching_tokens|)
- **Probe cap**: 50 tokens

The reward function<sup>†</sup> balances improvement gains against the risk of degradation, ensuring the teacher learns to be conservative with capable students while providing necessary support to weaker ones.

<sup>†</sup> See implementation: `trainers/teacher_rewards.py::AdaptiveTeachingReward`

### Evaluation Protocol

Offline evaluation compares student-with-teaching vs. baseline student on held-out problems:
- Matching uses string-normalized checks with partial match credit
- We're improving parsing/normalization to reduce artifacts

### Results Snapshot (Latest Run)

[Diagram Placeholder: Improvement rate visualization showing accuracy gains across problem difficulty]

**Environment**: 4×H100 GPUs, Arc-Intelligence/Arc-ATLAS-Teach-v0 dataset, seed=42

| Metric | Teacher+Student | Student Alone | Delta |
|--------|----------------|---------------|-------|
| Average accuracy | +0.1573 | baseline | **+15.73%** |
| Max per-item improvement | +0.296 | - | **+29.6%** |
| Token efficiency | 0.372 | - | (avg. efficiency term) |
| Generation time (32 samples) | ~1:10 | ~1:21 | **-13.6%** |
| Average length | ~2k tokens | ~4k tokens | **-50%** |
| Completion rate | ~100% | ~69% | **+31%** |

**Key findings**:
- Teacher+student achieves higher accuracy with fewer tokens
- Near-100% completion rate vs. ~10/32 failures in student-alone (due to looping)
- Consistent improvements across problem difficulty levels

[Diagram Placeholder: Adaptive teaching method showing 235B→4B/30B/235B data curation and 8B→4B RL training]

### Teaching Examples

[Diagram Placeholder: Real teaching snippets from training showing different adaptation levels - minimal hints for strong students, comprehensive support for struggling ones]

### Reproduce

**SFT Warmup**:
```bash
./launch.sh 4 configs/run/teacher_sft.yaml \
  dataset_id_or_path=bespokelabs/Bespoke-Stratos-17k
```

**RL (GRPO)**:
```bash
./launch_with_server.sh 1 3 configs/run/teacher_rcl.yaml \
  model_name_or_path=results/pre_rl_model \
  dataset_id_or_path=Arc-Intelligence/Arc-ATLAS-Teach-v0 \
  num_generations=32
```

### Notes & Limitations

- Normalization/parsing affects exact/partial match; reducing these artifacts lowers apparent degradation
- Efficiency and latency reflect our current token budgets and sampling settings; tune `num_generations`, context lengths, and `offload` settings for your hardware
- Current implementation focuses on reasoning tasks; extension to tool-use and multi-turn dialogue is in development

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