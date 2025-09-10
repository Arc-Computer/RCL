
# Installation

This project targets Python 3.11, PyTorch 2.6.0, and vLLM 0.8.3. We recommend Conda or uv for environment management and H100-class GPUs for best performance (4×H100).

## Prerequisites

- NVIDIA drivers + CUDA compatible with PyTorch 2.6.0
- GPUs: 4×H100 recommended; smaller setups can use `offload`/`zero1`
- Internet access for Hugging Face datasets/models; run `huggingface-cli login`

## Install

Option A: project script (recommended)

```bash
bash scripts/install_08.sh
```

Option B: manual installation

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install vllm==0.8.3 tensorboard
python -m pip install flash-attn --no-build-isolation
python -m pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.6/
python -m pip install --upgrade -r requirements_08.txt
```

## Authentication and Tracking

```bash
huggingface-cli login  # required for dataset/model access
export WANDB_API_KEY=...  # if using Weights & Biases
```

`train.py` sets `HF_HUB_ENABLE_HF_TRANSFER=1` to speed up downloads. To disable W&B, set `report_to: null` in configs.

## Security Best Practices

- Never commit tokens, `.env` files, checkpoints, or large logs
- Keep `results/`, `logs/`, and `wandb/` out of version control (already in `.gitignore`)
- Use environment variables for secrets (e.g., `HF_TOKEN`, `WANDB_API_KEY`), not config files
- Restrict dataset access to least privilege; log out with `huggingface-cli logout` on shared machines

## Verify

```bash
python -c "import torch, transformers, datasets; print(torch.cuda.is_available())"
accelerate --version
```
