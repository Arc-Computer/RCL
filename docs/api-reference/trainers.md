---
title: Trainers API
description: Complete API reference for GRPO trainer classes and reward functions.
---

# Trainers API

## GRPOTrainer

Base trainer class for Group Relative Policy Optimization (GRPO), implementing efficient policy gradient training with reference model KL regularization.

### Class Definition

```python
class GRPOTrainer(Trainer):
    """
    GRPO trainer for language model alignment using rewards and KL regularization.
    
    Inherits from transformers.Trainer and extends it with GRPO-specific functionality.
    """
```

### Constructor

```python
def __init__(
    self,
    model: Union[str, PreTrainedModel],
    reward_funcs: Union[RewardFunc, list[RewardFunc]],
    args: GRPOConfig = None,
    train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
    eval_dataset: Optional[Union[Dataset, IterableDataset, dict]] = None,
    processing_class: Optional[PreTrainedTokenizerBase] = None,
    reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list]] = None,
    callbacks: Optional[list[TrainerCallback]] = None,
    optimizers: tuple[Optional[Optimizer], Optional[LRScheduler]] = (None, None),
    peft_config: Optional["PeftConfig"] = None,
)
```

#### Parameters

- **model** : `Union[str, PreTrainedModel]`
  - The model to train. Can be a model ID string or instantiated model.
  - If string, will load using `AutoModelForCausalLM.from_pretrained()`
  
- **reward_funcs** : `Union[RewardFunc, list[RewardFunc]]`
  - Reward function(s) for scoring completions. Can be:
    - String model ID for reward model
    - PreTrainedModel instance
    - Callable function with signature `(prompts, completions, **kwargs) -> list[float]`
  
- **args** : `GRPOConfig`, optional
  - Training configuration. If None, creates default config.
  - See GRPOConfig section for all parameters.
  
- **train_dataset** : `Dataset`, optional
  - Training dataset with "prompt" column minimum
  
- **processing_class** : `PreTrainedTokenizerBase`, optional
  - Tokenizer for the model. Auto-loaded if not provided.

#### Key Methods

##### train()
Standard training loop inherited from Trainer with GRPO-specific loss computation.

##### _generate_and_score()
```python
def _generate_and_score(
    self, 
    inputs: dict[str, Union[torch.Tensor, Any]]
) -> tuple[list[str], torch.Tensor, torch.Tensor]
```
Generates completions and computes rewards.

**Returns:**
- prompts_text: List of prompt strings
- completion_ids: Tensor of completion token IDs  
- advantages: Normalized advantage scores

##### compute_loss()
```python
def compute_loss(
    self,
    model: PreTrainedModel,
    inputs: dict[str, torch.Tensor]
) -> torch.Tensor
```
Computes GRPO loss with KL regularization.

**Loss Formula:**
```
loss = -advantages * log_probs + beta * kl_divergence
```

### Reference
`trainers/grpo.py:151-800`

---

## TeacherGRPOTrainer

Extends GRPOTrainer with adaptive teaching capabilities using a two-pass protocol.

### Class Definition

```python
class TeacherGRPOTrainer(GRPOTrainer, TeacherTrainer):
    """
    GRPO trainer specialized for adaptive teaching with diagnostic probing.
    """
```

### Constructor

```python
def __init__(
    self,
    *args,
    student_model: Optional[Union[str, PreTrainedModel]] = None,
    use_reference_teacher_model: bool = False,
    student_model_init_kwargs: Optional[dict] = None,
    logging_prob: float = 0.0,
    disable_student_offloading: bool = False,
    **kwargs
)
```

#### Additional Parameters

- **student_model** : `Union[str, PreTrainedModel]`, optional
  - Student model for capability assessment. Defaults to ref_model.
  
- **use_reference_teacher_model** : `bool`, default=False
  - If True, uses reference model as teacher instead of trained model
  
- **student_model_init_kwargs** : `dict`, optional
  - Initialization kwargs for student model loading
  
- **logging_prob** : `float`, default=0.0
  - Probability of logging completions for debugging
  
- **disable_student_offloading** : `bool`, default=False
  - Prevents CPU offloading of student model

#### Key Attributes

- **max_probe_tokens** : `int`, default=500
  - Maximum tokens for diagnostic probe responses
  
- **student_diagnostic_template** : `str`
  - Template for generating student diagnostic probes
  
- **teacher_adaptive_template** : `str`
  - Template for adaptive teaching generation
  
- **student_with_teaching_template** : `str`
  - Template for student solution with teaching
  
- **student_baseline_template** : `str`
  - Template for baseline student solution

#### Key Methods

##### _generate_baseline_solutions()
```python
def _generate_baseline_solutions(
    self, 
    prompts_text: list[str]
) -> list[str]
```
Generates student solutions without teaching for baseline comparison.

**Parameters:**
- prompts_text: List of problem prompts

**Returns:**
- List of extracted solutions from student responses

##### _generate_student_approaches()
```python
def _generate_student_approaches(
    self,
    prompts_text: list[str]
) -> list[str]
```
Generates diagnostic probes to assess student capability (≤max_probe_tokens).

**Returns:**
- List of student approach descriptions

##### _extract_solutions_from_responses()
```python
def _extract_solutions_from_responses(
    self,
    responses: list[str]
) -> list[str]
```
Extracts solutions from responses containing `<solution>` tags.

**Returns:**
- List of extracted solution strings (empty string if no tags found)

### Reference
`trainers/teacher_trainers.py:19-292`

---

## AdaptiveTeachingReward

Reward function implementing asymmetric rewards for adaptive teaching.

### Class Definition

```python
class AdaptiveTeachingReward(TeacherReward):
    """
    Computes rewards that penalize degradation and reward efficient improvement.
    """
```

### Constructor

```python
def __init__(
    self,
    student_model: Optional[PreTrainedModel] = None,
    teacher_model: Optional[PreTrainedModel] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    degradation_penalty_multiplier: float = 2.0,
    efficiency_weight: float = 1.0,
)
```

#### Parameters

- **degradation_penalty_multiplier** : `float`, default=2.0
  - Multiplier for performance degradation penalty
  - Ensures conservative teaching for capable students
  
- **efficiency_weight** : `float`, default=1.0
  - Weight for teaching efficiency bonus
  - Rewards concise, targeted teaching

### Key Methods

##### evaluate_student_performance()
```python
@torch.no_grad()
def evaluate_student_performance(
    self,
    question: str,
    solution: str,
    ground_truth: str
) -> float
```

Evaluates correctness of student solution.

**Parameters:**
- question: Problem statement (currently unused)
- solution: Student's answer to evaluate
- ground_truth: Correct answer

**Returns:**
- 1.0 if exact match (case-insensitive)
- 0.0 otherwise

##### __call__()
```python
def __call__(
    self,
    prompts: list[str],
    completions: list[str],
    **kwargs
) -> list[float]
```

Computes adaptive teaching rewards.

**Required kwargs:**
- ground_truths: List of correct answers
- baseline_solutions: Solutions without teaching
- solutions: Solutions with teaching
- questions: Problem statements (defaults to prompts)

**Returns:**
- List of reward values per example

**Reward Calculation:**
```python
performance_delta = with_teaching - without_teaching

if performance_delta < 0:
    # Penalize degradation
    reward = -degradation_penalty_multiplier * abs(performance_delta)
else:
    # Reward improvement with efficiency bonus
    efficiency_bonus = 100.0 / (100.0 + teaching_tokens)
    reward = performance_delta * (1.0 + efficiency_weight * efficiency_bonus)
```

### Reference
`trainers/teacher_rewards.py:120-206`

---

## Configuration Usage

### Example: Training with Adaptive Teaching

```python
from trainers import TeacherGRPOTrainer, AdaptiveTeachingReward
from trainers.grpo_config import GRPOConfig

# Configure training
config = GRPOConfig(
    output_dir="results/adaptive_teacher",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    num_generations=8,
    max_prompt_length=512,
    max_completion_length=1024,
    max_probe_tokens=50,
    temperature=0.7,
    beta=0.04,
    degradation_penalty_multiplier=2.0,
    efficiency_weight=1.0
)

# Initialize reward
reward_func = AdaptiveTeachingReward(
    degradation_penalty_multiplier=2.0,
    efficiency_weight=1.0
)

# Create trainer
trainer = TeacherGRPOTrainer(
    model="Qwen/Qwen2.5-7B-Instruct",
    student_model="Qwen/Qwen2.5-1.5B-Instruct",
    reward_funcs=[reward_func],
    args=config,
    train_dataset=dataset,
)

# Train
trainer.train()
```

### Example: Custom Reward Function

```python
def custom_teaching_reward(prompts, completions, **kwargs):
    """
    Custom reward function following the expected interface.
    """
    rewards = []
    for prompt, completion in zip(prompts, completions):
        # Custom logic here
        reward = compute_custom_reward(prompt, completion)
        rewards.append(reward)
    return rewards

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[custom_teaching_reward],
    args=config,
)
```

---

## See Also

- [GRPOConfig](configs.md#grpoconfig) - Configuration parameters
- [Data Builders](data-builders.md) - Dataset preparation
- [vLLM Integration](../architecture/vllm-integration.md) - High-throughput generation