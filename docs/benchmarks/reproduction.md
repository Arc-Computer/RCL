
# Reproduction

The paper is in progress. To reproduce core results:

1) SFT warmup (3 epochs) on Bespoke-Stratos-17k
2) GRPO RL on Arc‑ATLAS‑Teach with vLLM servers (4×H100)

```bash
./launch.sh 4 configs/run/teacher_sft.yaml \
  dataset_id_or_path=bespokelabs/Bespoke-Stratos-17k \
  output_dir=results/pre_rl_model num_train_epochs=3

./launch_with_server.sh 1 3 configs/run/teacher_rcl.yaml \
  model_name_or_path=results/pre_rl_model \
  dataset_id_or_path=Arc-Intelligence/Arc-ATLAS-Teach-v0 \
  num_generations=64 generation_aggregation_steps=256 beta=0.04
```

Document exact package versions and seeds; attach small qualitative evals.

