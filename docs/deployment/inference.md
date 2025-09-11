
# Inference Deployment

## Loading Trained Models

### Using Transformers

Load ATLAS teacher models trained with ATLAS:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load from local checkpoint
model_path = "results/teacher_model_output"  # Path from training output_dir
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

# Or load published ATLAS models
model_name = "Arc-Intelligence/ATLAS-8B-Thinking"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

### Two-Pass Inference Protocol

ATLAS teacher models use a two-pass protocol for adaptive learning:

```python
# Pass 1: Diagnostic probing
probe_prompt = "Question: {question}\n\nBefore solving, briefly describe your approach:"
probe_inputs = tokenizer(probe_prompt, return_tensors="pt").to(model.device)
probe_output = model.generate(**probe_inputs, max_new_tokens=50)
student_approach = tokenizer.decode(probe_output[0], skip_special_tokens=True)

# Pass 2: Adaptive guidance based on probe
teaching_prompt = f"Question: {question}\n\nStudent approach: {student_approach}\n\nProvide adaptive guidance:"
teaching_inputs = tokenizer(teaching_prompt, return_tensors="pt").to(model.device)
teaching_output = model.generate(**teaching_inputs, max_new_tokens=512)
final_response = tokenizer.decode(teaching_output[0], skip_special_tokens=True)
```

## Production Considerations

### Model Export
- Training saves complete model and tokenizer automatically
- Output directory specified by `output_dir` parameter in configs
- Models compatible with standard Transformers inference pipeline

### Memory Requirements
- **ATLAS-8B models**: Minimum 16GB GPU memory (bfloat16)
- Use `torch_dtype=torch.bfloat16` for optimal memory usage
- `device_map="auto"` enables multi-GPU inference if needed

### Generation Parameters
- **Probe phase**: `max_new_tokens=50` (diagnostic responses)
- **Guidance phase**: `max_new_tokens=512` (adaptive guidance)
- **Temperature**: 0.7 (matches training configuration)
- **Top-p**: 0.9 (matches training configuration)

