---
title: Evaluation Methodology
description: Guidance to evaluate adaptive teaching without formal metric definitions.
---

# Evaluation Methodology

Evaluate that teaching improves student outcomes without harming strong cases. Suggested approach:

- Compare student solutions with and without teacher guidance on a held-out set.
- Track degradation cases and intervene in templates/length if needed.
- Record wall-clock, GPU utilization, and training settings for reproducibility.

Example (small evaluation sweep):

```bash
./launch_with_server.sh 1 3 configs/run/teacher_rcl.yaml \
  model_name_or_path=results/pre_rl_model \
  eval_steps=50 log_completions=true save_completions_probability=0.1
```

Store qualitative examples and small samples of completions for review (avoid committing large artifacts).

