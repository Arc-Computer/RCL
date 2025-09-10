
# Inference

## Load with Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

ckpt = "results/rcl_teacher/<group>/<run>/<ts>"  # or a pushed Hub repo
tok = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype="auto", device_map="auto")

prompt = "Solve: ..."
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=256)
print(tok.decode(out[0], skip_special_tokens=True))
```

## Export Considerations

- Save tokenizer with the model (`trainer.save_model` + `tokenizer.save_pretrained`).
- Ensure pad token is set (Qwen/Llama) via `fix_pad_token` at training time.

