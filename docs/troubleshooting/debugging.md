---
title: Debugging
description: Techniques for diagnosing training issues and validating changes.
---

# Debugging

## Smoke Tests

Run minimal steps to validate changes (see AGENTS.md):

```bash
./launch.sh 1 configs/run/teacher_sft.yaml \
  num_train_epochs=1 train_batch_size=16 per_device_train_batch_size=1 \
  save_final_model=false report_to=null

./launch_with_server.sh 1 3 configs/run/teacher_rcl.yaml \
  max_steps=4 eval_steps=1 save_final_model=false report_to=null
```

## Verbose Logs

- Set `logging_steps=1` and enable `activate_debugging_logs` in GRPO configs
- Use `log_completions=true` and `save_completions_probability` for sampling

## Check Accumulation and Batch Sizes

- `train.py` prints inferred `gradient_accumulation_steps`; ensure divisibility constraints in GRPO are satisfied

