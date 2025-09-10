
# Testing Your Installation

Run these minimal commands to verify your environment before full runs.

## Smoke Test (1 minute)

```bash
./launch.sh 1 configs/run/teacher_sft.yaml \
  num_train_epochs=1 \
  train_batch_size=16 \
  per_device_train_batch_size=1 \
  save_final_model=false \
  report_to=null
```

Expected: initialization logs and a short training loop without errors.

## Memory‑Constrained Test

For limited GPU memory, enable CPU offload:

```bash
./launch.sh 1 configs/run/teacher_sft.yaml offload \
  num_train_epochs=1 \
  train_batch_size=16 \
  per_device_train_batch_size=1 \
  save_final_model=false \
  report_to=null
```

If still OOM, reduce `per_device_train_batch_size`, decrease `max_seq_length`, or switch to `zero1`.

