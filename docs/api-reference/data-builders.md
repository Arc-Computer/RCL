---
title: Data Builders API
description: Complete API reference for dataset loading, formatting, and processing functions.
---

# Data Builders API

## Dataset Loading Functions

### load_formatted_sft_dataset

Primary function for loading and formatting datasets for supervised fine-tuning.

```python
def load_formatted_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_id_or_path: str,
    dataset_local_directory: Optional[str] = None,
    train_split: str = 'train',
    val_split: Optional[str] = None,
    process_line_fn: Optional[Callable] = None,
    model_name_or_path: Optional[str] = None,
    completion_only_training: bool = True,
    custom_start_of_response: Optional[str] = None,
    keep_columns: Optional[list[str]] = None,
    add_dataset_indices: bool = False,
    artificial_epochs: Optional[int] = None,
    **dataset_loading_kwargs
) -> tuple[Dataset, Optional[Dataset], DataCollator]
```

#### Parameters

- **tokenizer** : `PreTrainedTokenizerBase`
  - Tokenizer for processing text into model inputs
  
- **dataset_id_or_path** : `str`
  - HuggingFace dataset ID or local path
  - Special handling for "Arc-Intelligence/Arc-ATLAS-Teach-v0"
  
- **dataset_local_directory** : `str`, optional
  - Local directory containing dataset files
  - Used when loading from disk instead of HuggingFace Hub
  
- **train_split** : `str`, default='train'
  - Name of training split to load
  
- **val_split** : `str`, optional
  - Name of validation split. If None, no validation set loaded
  
- **process_line_fn** : `Callable`, optional
  - Custom function to process each dataset line
  - Signature: `(line: dict, tokenizer) -> dict`
  - If None, uses dataset-specific default from DATA_CONFIGS
  
- **model_name_or_path** : `str`, optional
  - Model identifier for special tokenizer configurations
  
- **completion_only_training** : `bool`, default=True
  - If True, masks prompt tokens in loss calculation
  - Only trains on completion/response tokens
  
- **custom_start_of_response** : `str`, optional
  - Custom marker for start of response in completion-only training
  - Default uses assistant role marker from chat template
  
- **keep_columns** : `list[str]`, optional
  - Columns to retain in processed dataset
  - Default keeps only necessary columns for training
  
- **add_dataset_indices** : `bool`, default=False
  - Add "__index" column with unique identifiers
  
- **artificial_epochs** : `int`, optional
  - Repeat dataset this many times for extended training
  
- **dataset_loading_kwargs**
  - Additional arguments passed to `load_dataset()`

#### Returns

- **train_dataset** : `Dataset`
  - Processed training dataset
  
- **val_dataset** : `Optional[Dataset]`
  - Processed validation dataset if val_split provided
  
- **collator** : `DataCollator`
  - Data collator for batching with optional masking

#### Special Dataset Handling

##### Arc-ATLAS-Teach-v0
When loading this dataset, the function:
1. Downloads the `training/sft.jsonl` file from HuggingFace
2. Processes with special formatting for adaptive teaching
3. Uses ADAPTIVE_TEACHING_SYSTEM_PROMPT
4. Formats with student approach, teacher diagnosis, and teaching

#### Example Usage

```python
from transformers import AutoTokenizer
from custom_data.sft_data import load_formatted_sft_dataset

tokenizer = AutoTokenizer.from_pretrained("Arc-Intelligence/ATLAS-8B-Instruct")

# Load standard dataset
train_ds, val_ds, collator = load_formatted_sft_dataset(
    tokenizer=tokenizer,
    dataset_id_or_path="bespokelabs/Bespoke-Stratos-17k",
    train_split="train",
    val_split="validation",
    completion_only_training=True
)

# Load Arc-ATLAS with custom processing
train_ds, _, collator = load_formatted_sft_dataset(
    tokenizer=tokenizer,
    dataset_id_or_path="Arc-Intelligence/Arc-ATLAS-Teach-v0",
)
```

### Reference
`custom_data/sft_data.py:43-150`

---

## Processing Functions

### get_process_line_fn

Returns a dataset-specific processing function based on DATA_CONFIGS.

```python
def get_process_line_fn(
    dataset_id_or_path: str
) -> Callable[[dict, PreTrainedTokenizerBase], dict]
```

#### Parameters

- **dataset_id_or_path** : `str`
  - Dataset identifier to look up in DATA_CONFIGS

#### Returns

- **process_line_fn** : `Callable`
  - Function that processes dataset lines into chat format
  - Applies system prompt and templates from DATA_CONFIGS

#### Processing Flow

1. Looks up dataset configuration in DATA_CONFIGS
2. Extracts question and completion using dataset-specific logic
3. Formats as chat messages with system/user/assistant roles
4. Applies tokenizer chat template

### Reference
`custom_data/sft_data.py:16-40`

---

## Data Collators

### make_masked_sft_collator

Creates a data collator for completion-only training with prompt masking.

```python
def make_masked_sft_collator(
    tokenizer: PreTrainedTokenizerBase,
    custom_start_of_response: Optional[str] = None,
    mask_prompt_tokens: bool = True
) -> DataCollatorForCompletionOnlyLM
```

#### Parameters

- **tokenizer** : `PreTrainedTokenizerBase`
  - Tokenizer for encoding/decoding
  
- **custom_start_of_response** : `str`, optional
  - Custom marker for response start
  - Defaults to assistant role marker
  
- **mask_prompt_tokens** : `bool`, default=True
  - Whether to mask prompt tokens in loss calculation

#### Returns

- **collator** : `DataCollatorForCompletionOnlyLM`
  - Collator that masks specified tokens during training

#### Masking Strategy

The collator identifies response tokens by:
1. Finding the start-of-response marker
2. Masking all tokens before this marker with -100
3. Training only on unmasked (response) tokens

### Reference
`custom_data/utils.py:200-250`

---

## Dataset Configuration

### DATA_CONFIGS

Dictionary mapping dataset IDs to their processing configurations.

```python
DATA_CONFIGS: dict[str, ReasoningData] = {
    "dataset_id": ReasoningData(
        system_prompt="...",
        think_prefix="<think>",
        think_suffix="</think>",
        solution_prefix="<solution>",
        solution_suffix="</solution>",
        extract_question_and_completion_from_line=custom_function
    ),
    ...
}
```

#### ReasoningData Structure

```python
@dataclass
class ReasoningData:
    system_prompt: str
    think_prefix: str = "<think>"
    think_suffix: str = "</think>" 
    solution_prefix: str = "**Solution:**"
    solution_suffix: str = ""
    extract_question_and_completion_from_line: Optional[Callable] = None
```

#### Built-in Configurations

- **CUSTOM_CONFIG_STRATOS_STYLE**: Default for unknown datasets
- **bespokelabs/Bespoke-Stratos-17k**: Reasoning with think tags
- **Arc-Intelligence/Arc-ATLAS-Teach-v0**: Adaptive teaching format
- **openr1/Big-Math-RL-Verified-Processed**: Math problems

### ADAPTIVE_TEACHING_SYSTEM_PROMPT

System prompt used for adaptive teaching datasets:

```python
ADAPTIVE_TEACHING_SYSTEM_PROMPT = """You are an adaptive teaching assistant specialized in mathematics and reasoning. Your role is to:

1. Diagnose student understanding through their initial approach
2. Provide targeted, adaptive teaching based on their specific needs
3. Guide them toward correct solutions without overwhelming them

When analyzing student approaches:
- Identify conceptual gaps or misconceptions
- Recognize partially correct reasoning
- Assess their current capability level

When providing teaching:
- Be concise and focused on what the student specifically needs
- Use appropriate scaffolding based on their demonstrated understanding
- Avoid over-explaining for capable students
- Provide comprehensive support for struggling students

Remember: Your goal is to improve performance without degradation. Be conservative with students who show strong understanding."""
```

### Reference
`custom_data/reasoning_datasets_info.py:50-200`

---

## Helper Functions

### add_indices

Adds unique indices to dataset rows.

```python
def add_indices(ds: Dataset) -> Dataset
```

Adds "__index" column if not present, using row indices.

### wrap_string_between_tag

Wraps content between specified tags.

```python
def wrap_string_between_tag(
    content: str,
    prefix: str,
    suffix: str
) -> str
```

### grab_text_between_tag

Extracts text between specified tags.

```python
def grab_text_between_tag(
    text: str,
    prefix: str,
    suffix: str,
    include_tags: bool = False
) -> Optional[str]
```

### Reference
`custom_data/reasoning_datasets_info.py:10-45`

---

## Example: Custom Dataset Integration

### Adding a New Dataset Configuration

```python
from custom_data.reasoning_datasets_info import DATA_CONFIGS, ReasoningData

# Define extraction function
def extract_my_dataset(line):
    question = line["problem"]
    reasoning = line.get("work", "")
    solution = line["answer"]
    
    completion = f"<think>\n{reasoning}\n</think>\n\nSolution: {solution}"
    return question, completion

# Add to DATA_CONFIGS
DATA_CONFIGS["my-org/my-dataset"] = ReasoningData(
    system_prompt="You are a helpful assistant.",
    think_prefix="<think>",
    think_suffix="</think>",
    solution_prefix="Solution:",
    solution_suffix="",
    extract_question_and_completion_from_line=extract_my_dataset
)

# Now load normally
train_ds, val_ds, collator = load_formatted_sft_dataset(
    tokenizer=tokenizer,
    dataset_id_or_path="my-org/my-dataset"
)
```

### Custom Processing Function

```python
def custom_process_fn(line, tokenizer):
    # Custom processing logic
    messages = [
        {"role": "system", "content": "Custom prompt"},
        {"role": "user", "content": line["input"]},
        {"role": "assistant", "content": line["output"]}
    ]
    
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,
        continue_final_message=False
    )
    
    return {"text": text}

# Use custom function
train_ds, _, collator = load_formatted_sft_dataset(
    tokenizer=tokenizer,
    dataset_id_or_path="any-dataset",
    process_line_fn=custom_process_fn
)
```

---

## See Also

- [Trainers API](trainers.md) - Training classes that use these datasets
- [Configuration](configs.md) - Dataset configuration in Hydra
- [Custom Datasets Guide](../guides/custom-datasets.md) - Tutorial on dataset integration
