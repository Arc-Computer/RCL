
## Configuration Usage

### Command Line Overrides

Override any configuration parameter from command line:

```bash
# Override single parameter
./launch.sh 8 configs/run/teacher_sft.yaml learning_rate=5e-6

# Override multiple parameters
./launch_with_server.sh 2 2 configs/run/teacher_rcl.yaml \
  dataset_id_or_path=my/dataset \
  temperature=0.8 \
  num_generations=16

# Use different model
./launch.sh 4 configs/run/teacher_sft.yaml \
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

Located in `accelerate_configs/` directory:

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
- [Training Pipeline Guide](../guides/training-pipeline.md) - Complete training workflow
- [Distributed Training Guide](../guides/distributed-training.md) - Multi-GPU setup

