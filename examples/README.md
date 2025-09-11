# ATLAS Examples

Interactive Jupyter notebooks demonstrating ATLAS's two-pass inference protocol with pre-trained teacher models.

## 🔄 Inference Pipeline Overview

The ATLAS inference pipeline wraps your existing agent with a teacher model in three simple steps:

1. **Load Models**: Your agent (student, 4B-8B) + ATLAS teacher (8B) models
2. **Run Two-Pass Protocol**: 
   - Diagnostic probe (~50 tokens): Teacher assesses your agent's capability
   - Adaptive guidance (~200 tokens): Teacher provides targeted help
3. **Generate Response**: Your agent uses guidance to improve output quality

**Total time**: ~30 seconds per problem on T4 GPU
**Improvement**: 15-30% accuracy gain with near-zero degradation

## ⚡ Quick Test (< 1 minute)

Test ATLAS with your agent in under a minute:

```python
from utils.atlas_inference import ATLASInference, load_atlas_models

# Initialize with your agent
atlas, _ = load_atlas_models(
    student_model_name="your-agent-model",  # Your existing agent
    teacher_thinking_name="Arc-Intelligence/ATLAS-8B-Thinking"
)

# Test on single problem
problem = "What is 15% of 80?"
result = atlas.run_full_protocol(problem)

print(f"Your Agent Alone: {result['baseline_response']}")
print(f"With ATLAS Teacher: {result['guided_response']}")
```

## 🚀 Wrap Your Agent with ATLAS

### Step 1: Install Dependencies
```bash
pip install transformers torch accelerate
```

### Step 2: Initialize ATLAS Wrapper
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.atlas_inference import ATLASInference

# Load your existing agent
your_agent = AutoModelForCausalLM.from_pretrained("your-model")
your_tokenizer = AutoTokenizer.from_pretrained("your-model")

# Load ATLAS teacher
teacher = AutoModelForCausalLM.from_pretrained("Arc-Intelligence/ATLAS-8B-Thinking")
teacher_tokenizer = AutoTokenizer.from_pretrained("Arc-Intelligence/ATLAS-8B-Thinking")

# Create ATLAS wrapper
atlas = ATLASInference(
    student_model=your_agent,
    student_tokenizer=your_tokenizer,
    teacher_model=teacher,
    teacher_tokenizer=teacher_tokenizer
)
```

### Step 3: Use in Your Pipeline
```python
def your_agent_with_atlas(user_input):
    # Run ATLAS protocol
    result = atlas.run_full_protocol(user_input)
    
    # Return enhanced response
    return result['guided_response']

# Drop-in replacement for your existing agent
response = your_agent_with_atlas("Solve this problem: ...")
```

### Integration Examples

**LangChain Integration:**
```python
from langchain.llms import HuggingFacePipeline

class ATLASEnhancedLLM(HuggingFacePipeline):
    def _call(self, prompt):
        result = atlas.run_full_protocol(prompt)
        return result['guided_response']
```

**API Server Integration:**
```python
from fastapi import FastAPI

app = FastAPI()

@app.post("/generate")
def generate(prompt: str):
    result = atlas.run_full_protocol(prompt)
    return {
        "baseline": result['baseline_response'],
        "enhanced": result['guided_response'],
        "strategy": result['learning']['strategy']
    }
```

## Quick Start

### Google Colab (Recommended)
Click the badges below to run examples directly in Google Colab with GPU acceleration:

[![Open Math Demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Arc-Intelligence/RCL/blob/main/examples/math_reasoning_demo.ipynb)
[![Open Code Demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Arc-Intelligence/RCL/blob/main/examples/code_generation_demo.ipynb)

**Hardware Requirements:**
- GPU: T4 (free tier) or A100 (Colab Pro/Pro+) recommended
- RAM: 12-16GB
- Runtime: < 10 minutes per notebook

### Local Development
```bash
# Clone repository
git clone https://github.com/Arc-Intelligence/RCL.git
cd RCL/examples

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

## Notebooks

### 1. Math Reasoning Demo (`math_reasoning_demo.ipynb`)
Demonstrates ATLAS improving math problem solving accuracy with:
- **Student Model**: Qwen/Qwen3-4B-Instruct-2507
- **Teacher Model**: ATLAS-8B-Thinking
- **Task**: GSM8K-style math word problems
- **Expected Improvement**: ~15.7% accuracy gain [¹](#performance-metrics)

**Key Features:**
- Two-pass diagnostic probing protocol
- Before/after accuracy comparison
- Token efficiency analysis
- Interactive problem exploration

### 2. Code Generation Demo (`code_generation_demo.ipynb`)
Shows ATLAS enhancing code generation and explanation quality with:
- **Student Model**: Qwen/Qwen3-4B-Instruct-2507  
- **Teacher Model**: ATLAS-8B-Instruct
- **Task**: HumanEval-style coding problems
- **Expected Improvement**: Better code quality and explanations

**Key Features:**
- Adaptive learning based on coding capability
- Code correctness evaluation
- Explanation quality assessment
- Developer productivity metrics

## Architecture Overview

ATLAS uses a two-pass inference protocol:

1. **Diagnostic Probing** (≤50 tokens): Teacher assesses student capability
2. **Adaptive Learning**: Conditional guidance based on diagnosed strength
   - Strong students: Minimal intervention to prevent degradation
   - Weak students: Comprehensive scaffolding and support

## Extending the Examples

### Using Your Own Student Model
```python
# Replace in any notebook
tokenizer = AutoTokenizer.from_pretrained("your/model-name")
model = AutoModelForCausalLM.from_pretrained(
    "your/model-name",
    device_map="auto",
    torch_dtype=torch.float16
)
```

### Custom Problem Sets
```python
# Add your own problems to data/
custom_problems = [
    {"problem": "Your question here", "solution": "Expected answer"},
    # ... more problems
]
```

### Different Tasks
The ATLAS protocol generalizes to any text generation task:
- Question answering
- Creative writing
- Technical documentation
- Code review and debugging

## Support

For issues or questions:
- Check the main [repository documentation](../README.md)
- Review [troubleshooting guide](../docs/)
- Open an issue on GitHub

## Performance Metrics

The performance improvements cited in these examples are based on comprehensive benchmarks documented in [docs/benchmarks/detailed-performance-analysis.md](../docs/benchmarks/detailed-performance-analysis.md):

- **15.7% accuracy improvement**: Average accuracy gain across Arc-ATLAS-Teach-v0 dataset (32 samples per problem)
- **31% completion rate improvement**: From ~69% to ~100% task completion
- **97% non-degradation rate**: Percentage of problems where adaptive learning maintains or improves performance

These metrics were validated on 4×H100 GPUs with controlled evaluation conditions (seed: 42).

## Citation

If you use these examples in your work:

```bibtex
@article{atlas2025,
  title     = {ATLAS: Adaptive Training Methodology for RL},
  author    = {Arc Intelligence},
  journal   = {arXiv preprint},
  year      = {2025}
}
```