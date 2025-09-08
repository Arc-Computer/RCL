---
title: Custom Datasets
description: Data column expectations, formatting, and integration with DATA_CONFIGS.
---

# Custom Datasets

RCL expects problem/solution-oriented data. Recommended SFT warmup precedes RL.

## Column Expectations

- Minimal: `question`, `solution`
- Optional: `reasoning_trace` (preferred for SFT)

`custom_data/reasoning_datasets_info.py` maps dataset IDs to tag/format rules via `DATA_CONFIGS`. By default, unknown IDs use `CUSTOM_CONFIG_STRATOS_STYLE`.

## Arc‑ATLAS‑Teach (SFT)

When `dataset_id_or_path=Arc-Intelligence/Arc-ATLAS-Teach-v0`, `custom_data/sft_data.py` loads `training/sft.jsonl` and constructs messages with fields:

- `problem_text`, `student_approach`, `teacher_diagnosis`, `teacher_teaching`

This is formatted under the Adaptive Teaching system prompt (`ADAPTIVE_TEACHING_SYSTEM_PROMPT`).

## Adding a New Dataset

1) Add an entry in `DATA_CONFIGS` (tags/system prompt)
2) Provide an extraction function if schema differs (see `CustomReasoningData`)
3) Use `process_line_fn` to transform to chat text for SFT

Example override:

```bash
./launch.sh 4 configs/run/teacher_sft.yaml \
  dataset_id_or_path=my_corpus \
  data.dataset_local_directory=my_corpus \
  data.keep_columns='["text"]'
```

For RL, ensure your dataset provides ground-truth signals required by the reward or adapt the reward accordingly.

