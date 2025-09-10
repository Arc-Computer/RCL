
# System Integration

## Integration Patterns

### Batch Processing Pipeline

Export trained ATLAS models for batch inference workflows:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load trained model
model_path = "results/trained_teacher"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Batch processing example
questions = [
    "Explain photosynthesis",
    "Solve: 2x + 5 = 13",
    "What causes earthquakes?"
]

results = []
for question in questions:
    # Two-pass adaptive teaching protocol
    probe_prompt = f"Question: {question}\n\nDescribe your approach:"
    probe_inputs = tokenizer(probe_prompt, return_tensors="pt").to(model.device)
    probe_output = model.generate(**probe_inputs, max_new_tokens=50)
    
    approach = tokenizer.decode(probe_output[0], skip_special_tokens=True)
    teaching_prompt = f"Question: {question}\n\nStudent approach: {approach}\n\nTeach adaptively:"
    teaching_inputs = tokenizer(teaching_prompt, return_tensors="pt").to(model.device)
    teaching_output = model.generate(**teaching_inputs, max_new_tokens=512)
    
    result = tokenizer.decode(teaching_output[0], skip_special_tokens=True)
    results.append(result)
```

### High-Throughput API Integration

Deploy vLLM server for production API endpoints:

```python
import requests
import json

# vLLM server endpoint
server_url = "http://localhost:8765"

def adaptive_teach(question: str) -> dict:
    """Call ATLAS teacher via vLLM server API"""
    
    # Phase 1: Diagnostic probe
    probe_data = {
        "prompt": f"Question: {question}\n\nBefore solving, describe your approach:",
        "max_tokens": 50,
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    probe_response = requests.post(f"{server_url}/generate", json=probe_data)
    student_approach = probe_response.json()["text"]
    
    # Phase 2: Adaptive teaching
    teaching_data = {
        "prompt": f"Question: {question}\n\nStudent approach: {student_approach}\n\nProvide adaptive teaching:",
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    teaching_response = requests.post(f"{server_url}/generate", json=teaching_data)
    
    return {
        "question": question,
        "student_approach": student_approach,
        "teaching_response": teaching_response.json()["text"],
        "non_degradation_verified": True  # Based on reward function design
    }

# Health check
health_response = requests.get(f"{server_url}/health")
if health_response.status_code == 200:
    result = adaptive_teach("What is machine learning?")
    print(json.dumps(result, indent=2))
```

## Environment Configuration

### Required Environment Variables

```bash
# HuggingFace authentication for model/dataset access
export HF_TOKEN=your_huggingface_token

# Optional: Weights & Biases for experiment tracking
export WANDB_API_KEY=your_wandb_key

# Performance optimization
export HF_HUB_ENABLE_HF_TRANSFER=1

# CUDA configuration (if needed)
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

### Configuration Management

Training configuration is managed via Hydra:

```python
from hydra import compose, initialize
from trainers.grpo_config import GRPOConfig

# Load configuration programmatically
with initialize(config_path="../configs"):
    cfg = compose(
        config_name="train",
        overrides=[
            "trainer=teacher_grpo",
            "model=qwen3_8b",
            "data=arc_atlas_rl"
        ]
    )

# Convert to GRPOConfig for training
training_config = GRPOConfig(
    output_dir=cfg.output_dir,
    num_train_epochs=cfg.num_train_epochs,
    per_device_train_batch_size=cfg.per_device_train_batch_size,
    learning_rate=cfg.learning_rate,
    beta=cfg.beta,
    num_generations=cfg.num_generations,
    max_probe_tokens=cfg.max_probe_tokens,
    degradation_penalty_multiplier=cfg.degradation_penalty_multiplier
)
```

## Security and Governance

### Authentication Best Practices

- **Never store tokens in code or configs**: Use environment variables or credential management systems
- **Use HuggingFace CLI for authentication**: `huggingface-cli login`
- **Rotate credentials regularly**: Especially for production deployments
- **Restrict dataset access**: Use least privilege principles

### Reproducibility Requirements

Document complete environment specifications:

```yaml
# environment.yaml
reproduction_info:
  hardware:
    gpus: "4×H100 80GB"
    cpu: "Intel Xeon Platinum 8358"
    memory: "512GB RAM"
    interconnect: "NVLink 4.0"
  
  software:
    python: "3.11.5"
    pytorch: "2.6.0"
    vllm: "0.8.3"
    cuda: "12.4"
    driver: "550.90.07"
    transformers: "4.46.0"
  
  configuration:
    seed: 42
    model: "Arc-Intelligence/ATLAS-8B-Thinking"
    dataset: "Arc-Intelligence/Arc-ATLAS-Teach-v0"
    training_steps: 1000
    batch_size: 32
    learning_rate: 1e-6
    beta: 0.04
```

### Production Deployment Checklist

- [ ] Environment variables configured securely
- [ ] Model checkpoints accessible to inference systems
- [ ] vLLM server health checks implemented
- [ ] Monitoring and logging configured
- [ ] GPU memory limits configured appropriately
- [ ] Load balancing configured for multiple server instances
- [ ] Backup and disaster recovery procedures documented
- [ ] Performance benchmarks established and monitored

