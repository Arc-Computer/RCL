---
title: SFT Warmup
description: Supervised fine-tuning setup, data formatting, and tips.
---

# SFT Warmup

The SFT phase preconditions the teacher on structured reasoning traces before RL.

## Command

```bash
./launch.sh 4 configs/run/teacher_sft.yaml \
  dataset_id_or_path=bespokelabs/Bespoke-Stratos-17k \
  output_dir=results/pre_rl_model \
  num_train_epochs=3
```

## Data Formatting

- `custom_data/reasoning_datasets_info.py` defines tags (`<|begin_of_thought|>`, `<|end_of_solution|>`), system prompts, and `DATA_CONFIGS`.
- `custom_data/sft_data.py` applies chat templates for SFT and supports completion-only masking.

## Practical Tips

- Use `bf16=true` when supported; enable gradient checkpointing.
- Adjust `max_seq_length` to fit VRAM; default SFT uses large contexts.
- Keep overrides minimal; prefer editing run files under `configs/run/` for repeatability.

