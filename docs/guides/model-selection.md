---
title: Model Selection
description: Choosing base models and configuring tokenizer/model settings.
---

# Model Selection

Default teacher: `Qwen/Qwen2.5-7B-Instruct` (see run configs). Alternatives are configured under `configs/model/*`.

## Configure Model

- Select via `- override /model@_global_:` in run YAMLs or with CLI overrides (e.g., `model_name_or_path=...`).
- `hydra_utils.fix_pad_token` ensures pad token safety for Llama/Qwen/Bespoke variants.

## Practical Guidance

- Prefer 7B–8B for initial experiments; larger models may require stronger offloading and longer training.
- Ensure tokenizer and model are compatible; save tokenizer with final model when exporting.

