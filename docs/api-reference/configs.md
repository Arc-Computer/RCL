
# Configuration API

## GRPOConfig

Main configuration class for GRPO training, inheriting from `transformers.TrainingArguments`.

### Class Definition

```python
@dataclass
class GRPOConfig(TrainingArguments):
    """
    Configuration for Group Relative Policy Optimization training.
    Extends TrainingArguments with GRPO-specific parameters.
    """
```

### Core Training Parameters

#### Model Configuration

- **model_init_kwargs** : `Optional[dict]`, default=None
  - Keyword arguments for `AutoModelForCausalLM.from_pretrained()`
  - Example: `{"torch_dtype": "float16", "use_cache": False}`

- **remove_unused_columns** : `bool`, default=False
  - Keep columns beyond 'prompt' if custom reward functions need them

#### Generation Parameters

- **num_generations** : `int`, default=8
  - Number of completions to sample per prompt
  - Global batch size must be divisible by this value

- **max_prompt_length** : `int`, default=512
  - Maximum prompt length (truncated from left if longer)

- **max_completion_length** : `int`, default=256
  - Maximum tokens to generate per completion

- **generation_aggregation_steps** : `Optional[int]`, default=None
  - Aggregates generations across accumulation steps for efficiency
  - Enables higher vLLM GPU utilization

#### Sampling Parameters

- **temperature** : `float`, default=0.7
  - Sampling temperature (higher = more random)

- **top_p** : `float`, default=1.0
  - Nucleus sampling threshold (0-1)

- **top_k** : `Optional[int]`, default=None
  - Top-k filtering (None = disabled)

- **min_p** : `Optional[float]`, default=None
  - Minimum token probability scaled by most likely token

- **repetition_penalty** : `float`, default=1.0
  - Penalty for repeating tokens (>1 discourages repetition)

### vLLM Configuration

#### Basic vLLM Settings

- **use_vllm** : `bool`, default=False
  - Enable vLLM for generation (requires GPU reservation)

- **vllm_device** : `str`, default="auto"
  - Device for vLLM (e.g., "cuda:1", "auto" selects next available)

- **vllm_gpu_memory_utilization** : `float`, default=0.9
  - GPU memory ratio for model/KV cache (0-1)

- **vllm_dtype** : `str`, default="auto"
  - Data type for vLLM ("auto", "float16", "bfloat16")

- **vllm_max_model_len** : `Optional[int]`, default=None
  - Override model context size for memory efficiency

#### vLLM Server Settings

- **use_vllm_server** : `bool`, default=False
  - Use external vLLM server instead of local instance

- **vllm_host** : `Optional[str]`, default=None
  - vLLM server hostname

- **vllm_port** : `Optional[int]`, default=None
  - vLLM server port

- **vllm_server_timeout** : `float`, default=120.0
  - Timeout waiting for server startup (seconds)

- **num_vllm_clients** : `int`, default=1
  - Number of parallel vLLM clients

### GRPO Algorithm Parameters

#### Optimization

- **learning_rate** : `float`, default=1e-6
  - Initial learning rate for AdamW optimizer

- **beta** : `float`, default=0.04
  - KL divergence coefficient for regularization

- **reward_weights** : `Optional[list[float]]`, default=None
  - Weights for multiple reward functions
  - Must match number of reward functions
  - Default: equal weights (1.0 each)

#### Reference Model Synchronization

- **sync_ref_model** : `bool`, default=False
  - Enable periodic reference model updates

- **ref_model_mixup_alpha** : `float`, default=0.9
  - Mixing coefficient: `π_ref = α * π_θ + (1 - α) * π_ref_prev`

- **ref_model_sync_steps** : `int`, default=64
  - Steps between reference model updates

### Memory Optimization

- **ds3_gather_for_generation** : `bool`, default=True
  - Gather DeepSpeed ZeRO-3 weights for generation
  - Disable for models exceeding single GPU VRAM

- **offload_untrained_models** : `bool`, default=False
  - CPU offload for reference and reward models

- **backprop_accumulation_steps** : `Optional[int]`, default=None
  - Accumulate loss during backprop for memory efficiency

- **backprop_accumulation_micro_batch_size** : `Optional[int]`, default=None
  - Alternative: specify max per-device batch for backprop

### Adaptive Teaching Parameters

- **max_probe_tokens** : `int`, default=50
  - Maximum tokens for student diagnostic probing

- **degradation_penalty_multiplier** : `float`, default=2.0
  - Multiplier for performance degradation penalty

- **efficiency_weight** : `float`, default=1.0
  - Weight for teaching efficiency bonus

- **student_diagnostic_template** : `Optional[str]`
  - Template for generating student probes
  - Variables: `{question}`

- **teacher_adaptive_template** : `Optional[str]`
  - Template for adaptive teaching generation
  - Variables: `{question}`, `{approach}`

- **student_with_teaching_template** : `Optional[str]`
  - Template for student solution with teaching
  - Variables: `{question}`, `{teaching}`

- **student_baseline_template** : `Optional[str]`
  - Template for baseline student solution
  - Variables: `{question}`

### Logging and Debugging

- **activate_debugging_logs** : `bool`, default=False
  - Enable detailed GRPO debugging output

- **log_completions** : `bool`, default=False
  - Log completions to Weights & Biases

- **save_completions_probability** : `Optional[float]`, default=None
  - Probability of saving sample completions to file

- **artificial_epochs** : `int`, default=1
  - Number of epochs for training (workaround for HF GRPO bug)

### Reference
`trainers/grpo_config.py:7-354`

---

## Configuration Usage

### Command Line Overrides

Override any configuration parameter from command line:

```bash
# Override single parameter
scripts/launch.sh 8 configs/run/teacher_sft.yaml learning_rate=5e-6

# Override multiple parameters
scripts/launch_with_server.sh 2 2 configs/run/teacher_rcl.yaml \
  dataset_id_or_path=my/dataset \
  temperature=0.8 \
  num_generations=16

# Use different model
scripts/launch.sh 4 configs/run/teacher_sft.yaml \
  model=llama3_70b \
  per_device_train_batch_size=1
```

### Programmatic Configuration

```python
from hydra import compose, initialize
from trainers.grpo_config import GRPOConfig

# Initialize Hydra
with initialize(config_path="../configs"):
    cfg = compose(config_name="train", 
                  overrides=["trainer=teacher_grpo"])
    
# Create GRPOConfig
config = GRPOConfig(
    output_dir=cfg.output_dir,
    num_train_epochs=cfg.num_train_epochs,
    per_device_train_batch_size=cfg.per_device_train_batch_size,
    num_generations=cfg.num_generations,
    temperature=cfg.temperature,
    beta=cfg.beta,
    learning_rate=cfg.learning_rate,
    max_probe_tokens=cfg.max_probe_tokens,
    degradation_penalty_multiplier=cfg.degradation_penalty_multiplier,
    efficiency_weight=cfg.efficiency_weight
)
```

### Creating Custom Configurations

#### Custom Model Configuration

```yaml
# configs/model/my_model.yaml
name: my_model
model_name_or_path: org/my-model-7b

model_init_kwargs:
  torch_dtype: bfloat16
  use_cache: false
  attn_implementation: flash_attention_2

tokenizer_kwargs:
  padding_side: left
  use_fast: true
```

#### Custom Dataset Configuration

```yaml
# configs/data/my_dataset.yaml
dataset_id_or_path: my-org/my-dataset
train_split: train
val_split: validation

completion_only_training: true
artificial_epochs: 2

keep_columns:
  - text
  - ground_truth
```

---

## Environment Variables

### Required

- **HF_TOKEN**: HuggingFace authentication token
- **WANDB_API_KEY**: Weights & Biases API key (if using W&B)

### Optional

- **HF_HUB_ENABLE_HF_TRANSFER**: Set to "1" for faster downloads
- **CUDA_VISIBLE_DEVICES**: Control GPU visibility
- **TRANSFORMERS_CACHE**: Model cache directory
- **VLLM_WORKER_MULTIPROC_METHOD**: Set to "spawn" for vLLM

### Example Setup

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxx
export WANDB_API_KEY=xxxxxxxxxxxxx
export HF_HUB_ENABLE_HF_TRANSFER=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

---

## DeepSpeed Configuration

Located in `accelerate/` directory:

### zero1.yaml (Stage 1 Optimization)

```yaml
compute_environment: LOCAL_MACHINE
deepspeed_config:
  gradient_accumulation_steps: 1
  gradient_clipping: 1.0
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: false
  zero_stage: 1
distributed_type: DEEPSPEED
```

### zero3.yaml (Stage 3 Optimization)

```yaml
compute_environment: LOCAL_MACHINE
deepspeed_config:
  gradient_accumulation_steps: 1
  gradient_clipping: 1.0
  offload_optimizer_device: cpu
  offload_param_device: cpu
  zero3_init_flag: true
  zero3_save_16bit_model: true
  zero_stage: 3
distributed_type: DEEPSPEED
```

---

## See Also

- [Trainers API](trainers.md) - Trainer classes using these configurations
- [RL Training Guide](../guides/rl-training.md) - GRPO training workflow
- [Distributed Training Guide](../guides/distributed-training.md) - Multi-GPU setup

