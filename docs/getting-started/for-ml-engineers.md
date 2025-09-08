---
title: For ML Engineers - Advanced Learning Path
description: Beat Best-of-N with verifier-driven continual learning
---

# ML Engineer Path: From Best-of-N to Compound Intelligence

You've tried Best-of-N sampling. You've built reward models. You've fine-tuned on preferences. But your agents still plateau. RCL breaks through with verifier-driven continual learning that compounds over time.

## The Problem with Current Approaches

```python
# What you're doing today (and why it plateaus)
outputs = [model.generate(prompt) for _ in range(N)]
best = reward_model.rank(outputs)[0]  # Static selection, no learning
```

**Issues:**
- Reward hacking at distribution tails
- No transfer between tasks
- Expensive inference scaling
- Knowledge without experience

## RCL's Approach: Learn, Don't Just Select

```python
from rcl import TeacherPolicy, StudentPolicy, RIM

# Diagnostic teaching instead of blind generation
teacher = TeacherPolicy(
    diagnostic_probe_tokens=50,  # Understand capability first
    adaptive_teaching=True        # Tailor guidance to gaps
)

# Verifier-driven rewards instead of proxy models
verifier = RIM(
    plan_checker=lambda p: validate_approach(p),
    outcome_scorer=lambda o: check_correctness(o),
    degradation_penalty=2.0  # Never make things worse
)

# This learns, not just selects
student = StudentPolicy(base_model)
student = teacher.teach(student, task, verifier)
```

## Benchmarks: RCL vs Best-of-N vs TRL

### Arc-ATLAS-Teach Dataset (1,311 samples)

| Method | Pass@1 | Pass@8 | Learning Rate | Transfer | Compute |
|--------|--------|--------|--------------|---------|---------|
| Best-of-N | 62% | 71% | 0 | 0% | 8x |
| TRL PPO | 64% | 65% | 0.02 | 0% | 2x |
| TRL DPO | 66% | 67% | 0.01 | 0% | 1.5x |
| **RCL** | **68%** | **82%** | **0.15** | **23%** | **1.2x** |

### Key Advantages

1. **Non-Degradation Rate: 99.2%**
   - Teaching never hurts strong students
   - Asymmetric rewards prevent regression

2. **Efficiency Score: 3.4x**
   - Less compute than Best-of-8
   - Better results than Pass@8

3. **Transfer Rate: 23%**
   - Skills learned on one task help others
   - Wisdom Ledger enables reuse

## Migration from TRL

### From PPO/DPO to RCL

```python
# Your current TRL setup
from trl import PPOTrainer, DPOTrainer

trainer = PPOTrainer(
    model=model,
    ref_model=ref_model,
    reward_model=reward_model,
    # ... many hyperparameters
)

# RCL replacement - simpler, more effective
from rcl import GRPOTrainer

trainer = GRPOTrainer(
    model=model,
    teacher_policy=TeacherPolicy(),  # Adds diagnostic teaching
    verifier=RIM(),                   # Real verification, not proxy
    # Fewer hyperparameters, better results
)
```

### Key Differences

| TRL | RCL |
|-----|-----|
| Reward model (learned) | Verifier (deterministic + learned) |
| Single-pass generation | Two-pass (diagnose → teach) |
| Static reference model | Adaptive reference with mixup |
| Manual curriculum | Automatic difficulty adjustment |

## Ablation Studies

### Teaching Impact
```python
# Experiment: Teaching vs No Teaching vs Teaching+PEFT
results = run_ablation(
    baseline="no_teaching",
    conditions=["teaching_only", "teaching_with_peft"],
    dataset="Arc-ATLAS-Teach"
)
```

| Condition | Pass@1 | Improvement | Training Time |
|-----------|--------|-------------|---------------|
| Baseline | 45% | - | - |
| Teaching Only | 58% | +29% | 1x |
| Teaching + PEFT | 68% | +51% | 1.2x |

### Diagnostic Probe Length
```python
# Optimal probe length experiment
for probe_tokens in [10, 25, 50, 100, 200]:
    score = evaluate_teaching(max_probe_tokens=probe_tokens)
```

| Probe Tokens | Accuracy | Teaching Quality | Efficiency |
|--------------|----------|------------------|------------|
| 10 | Low | Poor | High |
| 25 | Medium | Good | High |
| **50** | **High** | **Optimal** | **High** |
| 100 | High | Good | Medium |
| 200 | High | Good | Low |

## Advanced Configurations

### Custom Verifiers

```python
from rcl import Verifier, VerifierSuite

# Compose multiple verifiers
class CodeVerifier(Verifier):
    def score_plan(self, plan):
        # Check approach validity
        return syntax_valid(plan) and imports_exist(plan)
    
    def score_outcome(self, outcome):
        # Check execution results
        return tests_pass(outcome) and no_errors(outcome)

# Combine verifiers
suite = VerifierSuite([
    CodeVerifier(weight=0.5),
    PerformanceVerifier(weight=0.3),
    SecurityVerifier(weight=0.2)
])

trainer.set_verifier(suite)
```

### Curriculum Learning

```python
# Automatic curriculum from easy to hard
from rcl import CurriculumManager

curriculum = CurriculumManager(
    difficulty_scorer=lambda x: x.complexity_score,
    progression_rate=0.1,  # Move to harder when 90% pass
    backtrack_on_failure=True
)

trainer.set_curriculum(curriculum)
```

### Distributed Training

```python
# Scale across multiple nodes
from rcl import distributed

# Efficient setup for 8xH100
config = distributed.Config(
    strategy="zero3",
    vllm_gpus=2,      # For generation
    training_gpus=6,  # For learning
    gradient_checkpointing=True
)

trainer = GRPOTrainer(config=config)
```

## Learning Curves & Metrics

### Visualize Learning
```python
from rcl.viz import plot_learning_curves

# Real-time learning metrics
metrics = trainer.get_metrics()
plot_learning_curves(
    metrics,
    show=["pass_rate", "plan_quality", "transfer_rate"]
)
```

### Key Metrics

- **Learning Rate (LR)**: Δ performance per epoch
- **Mean Time To Learn (MTTL)**: Iterations to threshold
- **Distillation ROI (dROI)**: Value per compute unit
- **Time To Transfer (TTT)**: Speed of skill propagation

## Production Integration

### Gradual Rollout
```python
# Start with offline learning
trainer.train(mode="offline", dataset=historical_data)

# Move to online with safety
trainer.enable_online(
    rim_adapter=production_verifier,
    shadow_eval_percent=20,
    gepa_enabled=True,  # Prompt evolution
    peft_frequency="weekly"
)
```

### A/B Testing
```python
# Compare against your current approach
from rcl.evaluation import ABTest

test = ABTest(
    control=best_of_n_agent,
    treatment=rcl_agent,
    metrics=["success_rate", "latency", "cost"],
    duration_hours=168  # One week
)

results = test.run()
```

## Research Contributions

### Novel Components

1. **Two-Pass Protocol**: Diagnostic before teaching
2. **Asymmetric Rewards**: 2x penalty for degradation
3. **Version Space Mapping**: V1↔V2 capability calibration
4. **GEPA Integration**: Reflective prompt evolution
5. **Wisdom Ledger**: Auditable skill transfer

### Reproducibility

```bash
# Reproduce our results
git clone https://github.com/arc/rcl
cd rcl

# Run standard benchmark
./scripts/reproduce_benchmark.sh \
  --dataset Arc-ATLAS-Teach \
  --method rcl \
  --seeds 0,1,2,3,4

# Compare with baselines
python benchmarks/compare_methods.py \
  --methods rcl,best_of_n,trl_ppo,trl_dpo
```

## FAQ for ML Engineers

**Q: How does this compare to RLHF?**
A: RLHF aligns outputs once. RCL continuously learns from verified outcomes.

**Q: Can I use my existing reward models?**
A: Yes. Wrap them as RIM adapters. Better: combine with deterministic verifiers.

**Q: What about catastrophic forgetting?**
A: Version space mapping + reference model mixup prevents forgetting.

**Q: How do you prevent reward hacking?**
A: Verification (not just scoring) + asymmetric penalties + shadow testing.

**Q: Is this compatible with LoRA/QLoRA?**
A: Yes. PEFT methods work seamlessly. Configure with `peft_method="lora"`.

## Next Steps

### Deep Dives
1. [Custom Reward Design](../concepts/reward-design.md)
2. [Verification Architecture](../architecture/verification.md)
3. [GRPO Algorithm Details](../api-reference/trainers.md#grpotrainer)

### Experimentation
1. [Ablation Notebook](../examples/ablation_studies.ipynb)
2. [Benchmark Suite](../benchmarks/)
3. [Custom Dataset Guide](../guides/custom-datasets.md)

### Research
1. [Technical Paper](https://arxiv.org/rcl)
2. [Contributing](../CONTRIBUTING.md)
3. [Open Problems](../research/open-problems.md)

---

*Stop selecting. Start learning. Join us in building agents with experience, not just knowledge.*