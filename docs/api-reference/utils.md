
## DeepSpeed Utilities

### prepare_deepspeed

Prepares model for DeepSpeed with optional CPU offloading.

```python
def prepare_deepspeed(
    model: PreTrainedModel,
    accelerator: Accelerator,
    offload_to_cpu: bool = False
) -> Union[PreTrainedModel, DeepSpeedEngine]
```

#### Parameters

- **model** : `PreTrainedModel`
  - Model to prepare for DeepSpeed
  
- **accelerator** : `Accelerator`
  - Accelerate instance managing distributed training
  
- **offload_to_cpu** : `bool`, default=False
  - Enable parameter offloading to CPU for memory efficiency

#### Returns

- **model** : `Union[PreTrainedModel, DeepSpeedEngine]`
  - DeepSpeed-wrapped model ready for training

#### Usage

```python
from accelerate import Accelerator
from trainers.utils_trl_15 import prepare_deepspeed

accelerator = Accelerator()
model = prepare_deepspeed(
    model, 
    accelerator, 
    offload_to_cpu=True  # For large models
)
```

### unwrap_model_for_generation

Context manager for unwrapping model during generation.

```python
@contextmanager
def unwrap_model_for_generation(
    model: Union[PreTrainedModel, DeepSpeedEngine, DistributedDataParallel],
    accelerator: Accelerator
) -> PreTrainedModel
```

#### Parameters

- **model** : `Union[PreTrainedModel, DeepSpeedEngine, DistributedDataParallel]`
  - Wrapped model to unwrap for generation
  
- **accelerator** : `Accelerator`
  - Accelerate instance managing the model

#### Yields

- **unwrapped_model** : `PreTrainedModel`
  - Base model suitable for generation

#### Usage

```python
from trainers.utils_trl_15 import unwrap_model_for_generation

with unwrap_model_for_generation(model, accelerator) as unwrapped:
    outputs = unwrapped.generate(
        input_ids,
        generation_config=generation_config
    )
```

### Reference
`trainers/utils_trl_15.py:100-150`

---

## vLLM Client

### VLLMClient

Client for communicating with vLLM generation server.

```python
class VLLMClient:
    """
    HTTP client for vLLM server with weight synchronization support.
    """
    
    def __init__(
        self,
        host: str,
        port: int,
        server_type: str = "http"
    )
```

#### Constructor Parameters

- **host** : `str`
  - vLLM server hostname
  
- **port** : `int`
  - vLLM server port
  
- **server_type** : `str`, default="http"
  - Server protocol type

#### Key Methods

##### check_server()

```python
def check_server(
    self,
    total_timeout: float = 120.0
) -> None
```

Polls `/health/` endpoint until server is ready.

**Parameters:**
- total_timeout: Maximum seconds to wait for server

**Raises:**
- `ConnectionError` if server doesn't respond within timeout

##### generate()

```python
def generate(
    self,
    prompts: list[str],
    sampling_params: dict = None
) -> list[str]
```

Sends generation request to vLLM server.

**Parameters:**
- prompts: List of text prompts
- sampling_params: Generation parameters (temperature, top_p, etc.)

**Returns:**
- List of generated completions

##### init_communicator()

```python
def init_communicator(
    self,
    host: str,
    port: int,
    world_size: int
) -> None
```

Initializes NCCL communicator for weight synchronization.

**Parameters:**
- host: NCCL group hostname
- port: NCCL group port
- world_size: Number of processes in group

##### update_model_params()

```python
def update_model_params(
    self,
    model: PreTrainedModel
) -> None
```

Synchronizes model weights to vLLM server via NCCL.

**Parameters:**
- model: Model with updated weights to sync

#### Example Usage

```python
from trainers.vllm_client import VLLMClient

# Initialize client
client = VLLMClient(host="localhost", port=8000)

# Check server health
client.check_server(total_timeout=60)

# Generate completions
prompts = ["Question: What is 2+2?", "Question: Explain gravity."]
sampling_params = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 256
}

completions = client.generate(prompts, sampling_params)

# Weight synchronization (for training)
client.init_communicator(
    host="localhost",
    port=51216,
    world_size=4
)
client.update_model_params(trained_model)
```

### Reference
`trainers/vllm_client/utils.py:50-250`

---

## Chat Format Utilities

### setup_chat_format

Sets up chat template for models without native chat support.

```python
def setup_chat_format(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    format: Literal["chatml"] = "chatml",
    resize_to_multiple_of: Optional[int] = None
) -> tuple[PreTrainedModel, PreTrainedTokenizer]
```

#### Parameters

- **model** : `PreTrainedModel`
  - Model to configure for chat
  
- **tokenizer** : `PreTrainedTokenizer`
  - Tokenizer to add chat template to
  
- **format** : `Literal["chatml"]`, default="chatml"
  - Chat format to apply
  
- **resize_to_multiple_of** : `Optional[int]`, default=None
  - Resize embeddings to multiple of this value

#### Returns

- **model, tokenizer** : `tuple[PreTrainedModel, PreTrainedTokenizer]`
  - Configured model and tokenizer with chat support

#### ChatML Format

Adds special tokens and template:
- `<|im_start|>system`
- `<|im_start|>user`
- `<|im_start|>assistant`
- `<|im_end|>` (EOS and pad)

#### Example

```python
from trainers.utils_trl_15 import setup_chat_format

model, tokenizer = setup_chat_format(
    model,
    tokenizer,
    format="chatml",
    resize_to_multiple_of=8
)

# Now can use chat template
messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"}
]
text = tokenizer.apply_chat_template(messages)
```

### Reference
`trainers/utils_trl_15.py:68-105`

---

## Data Processing Utilities

### wrap_as_list

Utility to wrap arguments and kwargs as a list.

```python
def wrap_as_list(
    *args,
    **kwargs
) -> list
```

Combines positional and keyword arguments into a single list.

### Reference
`hydra_utils.py:38-44`

---

## See Also

- [Trainers API](trainers.md) - Training classes using these utilities
- [vLLM Integration](../architecture/vllm-integration.md) - vLLM server architecture
- [Distributed Training](../guides/distributed-training.md) - DeepSpeed configuration

