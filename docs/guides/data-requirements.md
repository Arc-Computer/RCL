
# Data Requirements

RCL expects problem–solution data with optional reasoning traces. These requirements apply across SFT and RL.

## Columns

- `question` (string): the problem or prompt text
- `solution` (string): the ground‑truth final answer
- `reasoning_trace` (string, optional): detailed steps; used for SFT warmup

During SFT, unknown datasets are formatted via `DATA_CONFIGS` defaults; reasoning is wrapped with `<|begin_of_thought|> ... <|end_of_thought|>` and solutions with `<|begin_of_solution|> ... <|end_of_solution|>`.

## Example JSONL (generic)

```json
{"question": "Compute 27^(2/3)", "solution": "9", "reasoning_trace": "27^(1/3)=3; 3^2=9"}
```

## Arc‑ATLAS‑Teach (SFT special case)

When `dataset_id_or_path=Arc-Intelligence/Arc-ATLAS-Teach-v0`, SFT loads `training/sft.jsonl` and builds messages from fields:

- `problem_text`, `student_approach`, `teacher_diagnosis`, `teacher_teaching`

See `custom_data/sft_data.py` for construction and `ADAPTIVE_TEACHING_SYSTEM_PROMPT` in `custom_data/datasets_info.py`.

## DATA_CONFIGS Customization

`custom_data/datasets_info.py` defines `DATA_CONFIGS`, mapping dataset IDs to tag format and system prompts.

Steps to support a custom dataset:

1) Add an entry to `DATA_CONFIGS` (tags/system prompt)
2) If your schema differs, subclass `CustomReasoningData` to extract `question`/`solution`/`reasoning_trace`
3) Optionally provide a `process_line_fn` to format chat text for SFT (see `custom_data/sft_data.py`)

