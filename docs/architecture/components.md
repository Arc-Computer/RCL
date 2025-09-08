---
title: Components
description: Core modules—trainers, rewards, data, and utilities with their roles.
---

# Components

## Trainers

- `trainers/teacher_trainers.py::TeacherGRPOTrainer`: GRPO-based teacher with two-pass generation (probe → teaching). Accepts templates, supports offloading, and integrates rewards.
- `trainers/grpo.py`, `trainers/grpo_config.py`: GRPO loop and configuration (sampling params, vLLM options, ZeRO settings, reference sync).
- `trainers/teacher_base.py`: Base interfaces for `TeacherTrainer` and `TeacherReward`, logging helpers.

## Rewards

- `trainers/teacher_rewards.py::AdaptiveTeachingReward`: Asymmetric reward with degradation penalty and efficiency bonus. Configured via `configs/trainer/reward/adaptive_teaching.yaml`.

## Data

- `custom_data/sft_data.py`: SFT dataset loader and chat-template formatter. Special handling for `Arc-Intelligence/Arc-ATLAS-Teach-v0`.
- `custom_data/reasoning_datasets_info.py`: Tagging, templates, and `DATA_CONFIGS` mapping; adaptive teaching system prompt.
- `custom_data/utils.py`: collators and helpers for masking/completion-only training.

## vLLM Integration

- `trainers/vllm_server.py`: FastAPI server for generation and NCCL-based weight sync.
- `trainers/vllm_client/utils.py::VLLMClient`: health checks, generate API, and weight updates.

## Utilities

- `hydra_utils.py`: tokenizer pad fixes, simple subprocess helpers.
- `train.py`: Hydra main, W&B integration, dynamic accumulation, checkpoint resume logic.

