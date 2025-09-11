#!/bin/bash

set -euo pipefail

python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install vllm==0.8.3 tensorboard
python -m pip install flash-attn==2.7.3 --no-build-isolation --no-cache-dir
python -m pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.6/

python -m pip install --upgrade -r requirements-py312.txt
python -m pip check
python - <<'PY'
import importlib
for m in ("vllm","torch","accelerate","ray"):
    try:
        importlib.import_module(m)
    except Exception as e:
        raise SystemExit(f"Import failed: {m}: {e}")
print("Imports OK (py312)")
PY
