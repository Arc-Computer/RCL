# Online Teaching Optimization

The ATLAS online learning workflow layers GEPA's reflective prompt evolution on top of the RL-trained ATLAS-8B-Thinking teacher. In under two hours we reached a 165% student performance gain on Arc-ATLAS-Teach-v0 while spending roughly $10 on inference. The reflection agent (gemini/gemini-flash-2.5) kept prompts concise, pushing the efficiency score to 1.97—97% shorter than the student's baseline attempts.

> Credit: GEPA Reflective Prompt Evolution by Lakshya A Agrawal et al. inspired this pipeline.

## System Overview
- **Teacher**: `Arc-Intelligence/ATLAS-8B-Thinking` (GRPO-finetuned for adaptive teaching)
- **Student**: `Qwen/Qwen3-4B`
- **Reflection LM**: `gemini/gemini-flash-2.5` via LiteLLM
- **Dataset**: `Arc-Intelligence/Arc-ATLAS-Teach-v0` (`training/rl.jsonl` split)
- **Budgets**: 2,048 max tokens per completion, 500-token diagnostic pass, up to 3,500 metric calls

The GEPA loop evaluates baseline student performance, applies reflective mutations to teaching prompts, and keeps the best Pareto-front candidate after each iteration. A plateau between iterations 2–3 indicates the optimizer preserves strong configurations instead of overfitting.

## Quick Start

### Standard Optimization (Teacher + Student Models)
```bash
python optimize_teaching.py --config configs/optimize/default.yaml \
  --teacher-model Arc-Intelligence/ATLAS-8B-Thinking \
  --student-model Qwen/Qwen3-4B \
  --reflection-lm gemini/gemini-flash-2.5
```
- Populate `configs/optimize/default.yaml` with alternate seed prompts, dataset overrides, or LiteLLM worker limits.
- Provide provider credentials (for example `GEMINI_API_KEY`, `OPENAI_API_KEY`, Azure keys) in your environment so LiteLLM can route calls.
- Results are written to `optimized_prompts.json`, with traces in `traces/optimize_traces/`.

### Compatibility Mode (Wrap an Existing Agent)
```bash
scripts/openai_agent_atlas.sh configs/wrappers/openai_existing_agent.yaml
```
- Set `user_agent.integration_type` to `http_api`, `python_function`, or `cli_command` and adjust headers or module paths.
- The wrapper invokes your agent twice: once for the baseline answer and once with `<teaching>` guidance injected.
- Teaching prompts evolve through the same GEPA pipeline, and outputs persist under `compatibility_results.json` plus `traces/compatibility_traces/`.

## Configuration Anatomy
- `seed_prompts`: Templates the optimizer may mutate. In compatibility mode only the teacher template is active.
- `fixed_prompts`: Prompts that stay constant (baseline student prompt, etc.).
- `generation_config`: Controls token budgets, temperature, and LiteLLM request timeouts.
- `max_metric_calls`: Total evaluation budget—raise for broader searches or lower for quick smoke tests.
- `module_selector`: `round_robin` rotates prompt components; `single` locks to one component (used for compatibility mode).
- `max_litellm_workers`: Parallel LiteLLM worker count; adjust to match provider rate limits.

## Wrapper Types
| Type | Config Keys | Notes |
|------|-------------|-------|
| `http_api` | `endpoint`, `prompt_field`, `response_field`, optional `headers`, `timeout` | Place API keys in `.env` (e.g., `OPENAI_API_KEY`) or inline headers. Payload is built from `request_template`. |
| `python_function` | `module_path`, `function_name` | Function must accept a prompt string and return a string. Useful for local agents. |
| `cli_command` | `command`, optional `timeout` | `{prompt}` placeholder is swapped into the command. Use for shell tools or scripts. |
| `openai_sdk` | `name`, `instructions`, `model`, optional `tools`, `handoffs` | Uses OpenAI Agents SDK; rely on `OPENAI_API_KEY`. Configure tools via module paths. |
| `openai_assistant` | `api_key`, optional `assistant_id`, `name`, `instructions`, `model`, optional `response_format`, `output_extraction` | Creates or reuses an Assistant; `api_key` should be provided via env or config. Supports JSON-formatted outputs. |

Wrappers in `configs/wrappers/` introduce `teacher_wrapper` and `student_wrapper` blocks. Supply `type` (`openai_sdk`, `openai_assistant`, `custom`) and `config` to let GEPA call hosted agents or OpenAI Assistants. See `wrappers/` for implementation details.

## Outputs and Iteration
- `optimized_prompts.json`: Contains `best_candidate`, `best_score`, and Pareto frontier snapshots.
- `traces/*/prompt_evolution/`: JSON files recording prompts per evaluation for auditability.
- `traces/*/eval_XXXX.jsonl`: Rollout-level trajectories (baseline, teaching, enhanced responses, rewards).
- `results.best_score`: Direct comparison across iterations; monitor for plateaus to understand convergence.

For quick scripts you can reach out to any LiteLLM-compatible backend (OpenAI, Azure, local vLLM, etc.) by adjusting the model names and providing credentials via environment variables.

```python
import json
import litellm

with open('optimized_prompts.json') as f:
    prompts = json.load(f)['best_candidate']

teacher_template = prompts['teacher_adaptive_template']
student_diag = prompts.get('student_diagnostic_template')
student_with_teaching = prompts.get('student_with_teaching_template')

def call_llm(model_name, prompt, max_tokens=512, temperature=0.2):
    response = litellm.completion(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature
    )
    return response.choices[0].message.content

def run_with_best_prompts(question):
    if student_diag:
        approach = call_llm(
            'Qwen/Qwen3-4B',
            student_diag.format(question=question),
            max_tokens=400
        )
        teaching_prompt = teacher_template.format(question=question, approach=approach)
    else:
        baseline = call_llm('Qwen/Qwen3-4B', question, max_tokens=512)
        teaching_prompt = teacher_template.format(
            question=question,
            baseline_response=baseline
        )
        approach = baseline
    teaching = call_llm(
        'Arc-Intelligence/ATLAS-8B-Thinking',
        teaching_prompt,
        max_tokens=600,
        temperature=0.1
    )
    if student_with_teaching:
        final_prompt = student_with_teaching.format(question=question, teaching=teaching)
    else:
        final_prompt = f"Context: {teaching}

Question: {question}"
    return call_llm('Qwen/Qwen3-4B', final_prompt, max_tokens=700, temperature=0.2)

print(run_with_best_prompts('Solve 12 * 7?'))
```


Replay prompts in notebooks or production agents by loading `best_candidate` templates and feeding them into the ATLAS inference utilities under `examples/utils/`.

## Best Practices
- Choose a reflection LM that supports long outputs (≥2k tokens) and low latency; Gemini Flash 2.5 and GPT-4o-mini work well.
- Keep diagnostic prompts short (≤500 tokens) to control cost and speed up iterations.
- Use validation splits (`--valset`) to verify gains on held-out data before adopting new prompts.
- When wrapping existing agents, normalize outputs (e.g., enforce `<solution>` tags) so the reward extractor can compare baselines reliably.
- Monitor cost and latency via LiteLLM logging or provider dashboards—GEPA is parallel but bounded by `max_metric_calls`.

## Next Steps
- Feed optimized prompts into SFT or GRPO runs to create new teacher checkpoints.
- Integrate evolved prompts into API-facing inference services for immediate gains.
- Share observations or issues via the repository issue tracker; include trace files for faster debugging.
