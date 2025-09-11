"""
Configuration settings for ATLAS examples.

Centralizes magic numbers and configuration parameters.
"""

# Model Configuration
DEFAULT_STUDENT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_TEACHER_THINKING = "Arc-Intelligence/ATLAS-8B-Thinking"
DEFAULT_TEACHER_INSTRUCT = "Arc-Intelligence/ATLAS-8B-Instruct"

# Token Limits
PROBE_TOKEN_LIMIT = 50  # Maximum tokens for diagnostic probing (per docs)
TEACHING_RESPONSE_LIMIT = 200  # Maximum tokens for teaching guidance
STUDENT_RESPONSE_LIMIT = 300  # Maximum tokens for student responses

# Capability Score Thresholds
CAPABILITY_HIGH_THRESHOLD = 4  # Scores 4-5: Light intervention
CAPABILITY_MEDIUM_THRESHOLD = 2  # Scores 2-3: Medium guidance
# Score 1: Heavy support

# Teaching Strategies (aligned with docs)
STRATEGY_LIGHT = "Light"
STRATEGY_MEDIUM = "Medium" 
STRATEGY_HEAVY = "Heavy"

# Evaluation Settings
DEGRADATION_PENALTY_MULTIPLIER = 2.0  # Per adaptive-teaching.md
IMPROVEMENT_REWARD = 1.0
NO_CHANGE_REWARD = 0.0

# Memory Requirements
MIN_GPU_MEMORY_GB = 12  # Minimum recommended GPU memory
RECOMMENDED_GPU_MEMORY_GB = 16  # Recommended for smooth operation

# Dataset Settings
DEFAULT_NUM_SAMPLES = 20  # Default number of samples to load
DEFAULT_DATASET_SPLIT = "train"

# Model Loading Settings
DEFAULT_DEVICE_MAP = "auto"
DEFAULT_TORCH_DTYPE = "float16"
MODEL_TRUST_REMOTE_CODE = True  # Required for some models

# Timeout Settings (seconds)
MODEL_DOWNLOAD_TIMEOUT = 300  # 5 minutes for model downloads
DATASET_DOWNLOAD_TIMEOUT = 120  # 2 minutes for dataset downloads

# Cache Settings
ENABLE_OFFLINE_MODE = True  # Enable fallback to cached data
CACHE_DIR = "./cache"  # Directory for cached datasets

# Logging Settings
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Visualization Settings
PLOT_FIGSIZE = (14, 10)
PLOT_DPI = 100
COLOR_IMPROVEMENT = "#2ca02c"
COLOR_DEGRADATION = "#ff4444"
COLOR_UNCHANGED = "#d3d3d3"

# Error Messages
ERROR_GPU_OOM = """
❌ GPU Out of Memory Error!
Solutions:
1. Free GPU memory: torch.cuda.empty_cache()
2. Use smaller batch size
3. Enable 8-bit quantization: load_in_8bit=True
4. Use CPU fallback: device_map='cpu'
5. Use Google Colab with larger GPU (A100)
"""

ERROR_MODEL_LOAD = """
❌ Failed to load model: {model_name}
Please check:
1. Internet connection
2. HuggingFace access token (if required)
3. Available disk space
4. Model name spelling
"""

ERROR_DATASET_LOAD = """
⚠️ Failed to load dataset: {dataset_name}
Falling back to sample problems...
"""

# Performance Benchmarks (from docs/benchmarks/detailed-performance-analysis.md)
BENCHMARK_ACCURACY_IMPROVEMENT = 15.7  # Percentage
BENCHMARK_COMPLETION_RATE_IMPROVEMENT = 31  # Percentage
BENCHMARK_NON_DEGRADATION_RATE = 97  # Percentage
BENCHMARK_TOKEN_EFFICIENCY = 0.372

# Documentation References
DOCS_PERFORMANCE_ANALYSIS = "../docs/benchmarks/detailed-performance-analysis.md"
DOCS_ADAPTIVE_TEACHING = "../docs/concepts/adaptive-teaching.md"
DOCS_MAIN_README = "../README.md"