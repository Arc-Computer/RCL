#!/bin/bash

# gpus for vllm server
num_gpus=$1
# gpus for backprop
num_gpus_2=$2

# resolve yaml path
if [[ "$3" == configs/run/* ]]; then
  yaml_file="$3"
else
  yaml_file="configs/run/$3"
fi

# collect extra args, grabbing an opt model override if present
custom_model=""
extra_args=()
for arg in "${@:4}"; do
  if [[ "$arg" == model_name_or_path=* ]]; then
    custom_model="${arg#model_name_or_path=}"
  else
    extra_args+=("$arg")
  fi
done

echo "Running launch script..."

# cleanup helpers
cleanup() {
  echo "cleaning up background processes"
  kill $(jobs -p) 2>/dev/null || true
  unset CUDA_VISIBLE_DEVICES
  nvidia-smi --query-compute-apps=pid \
    --format=csv,noheader | xargs -r -n1 kill -9
  pkill -9 -f vllm_server || true
}
trap 'cleanup; exit 1' SIGINT SIGTERM
trap 'cleanup' EXIT

# read values from yaml
model_name=$(grep '^model_name_or_path:' "$yaml_file" | awk '{print $2}')
base_port=$(grep '^vllm_port:' "$yaml_file" | awk '{print $2}')

# apply optional model override
if [[ -n "$custom_model" ]]; then
  model_name="$custom_model"
  if grep -q '^model_name_or_path:' "$yaml_file"; then
    sed -i "s|^model_name_or_path:.*|model_name_or_path: $model_name|" \
      "$yaml_file"
  else
    echo "model_name_or_path: $model_name" >> "$yaml_file"
  fi
fi

echo "Extracting prefix cache..."
# prepare prefix caching flag
if [[ -n "$enable_prefix_caching" ]]; then
  prefix_arg="--enable_prefix_caching $enable_prefix_caching"
else
  prefix_arg=""
fi

# seed handling
if grep -q '^seed:' "$yaml_file"; then
  base_seed=$(grep '^seed:' "$yaml_file" | awk '{print $2}')
elif grep -q '^seed:' configs/train.yaml; then
  base_seed=$(grep '^seed:' configs/train.yaml | awk '{print $2}')
else
  echo "Error: seed not found in $yaml_file or configs/train.yaml"
  exit 1
fi

# sanity checks
[[ -z "$model_name" ]] && {
  echo "Error: include 'model_name_or_path' in $yaml_file"
  exit 1
}
[[ -z "$base_port" ]] && {
  echo "Error: include 'vllm_port' in $yaml_file"
  exit 1
}

master_addr=$(hostname)
echo "MODEL_NAME=$model_name, MASTER_ADDR=$master_addr, BASE_PORT=$base_port, \
NUM_GPUS=$num_gpus" > vllm_server_sessions.txt

# launch one vllm server per gpu
for ((i=0; i<num_gpus; i++)); do
  seed=$((base_seed + i))
  seed_arg="--seed $seed"
  cmd="CUDA_VISIBLE_DEVICES=$i python trainers/vllm_server.py \
--model=$model_name --port=$((base_port + i)) $prefix_arg $seed_arg"
  bash -c "$cmd" 2>&1 | tee -a job_${PBS_JOBID}.log &
done

echo "Servers initialized!"
# ensure yaml ends with newline
sed -i -e '$a\' "$yaml_file"

# record host and client details in yaml
if grep -q '^vllm_host:' "$yaml_file"; then
  sed -i "s/^vllm_host:.*/vllm_host: $master_addr/" "$yaml_file"
else
  echo "vllm_host: $master_addr" >> "$yaml_file"
fi
if grep -q '^num_vllm_clients:' "$yaml_file"; then
  sed -i "s/^num_vllm_clients:.*/num_vllm_clients: $num_gpus/" "$yaml_file"
else
  echo "num_vllm_clients: $num_gpus" >> "$yaml_file"
fi

sleep 120

# wait for log silence (30-sec checks)
log_file="job_${PBS_JOBID}.log"
if [[ -f "$log_file" ]]; then
  last_size=$(stat -c%s "$log_file")
  while true; do
    sleep 30
    cur_size=$(stat -c%s "$log_file")
    [[ "$cur_size" -eq "$last_size" ]] && break
    last_size=$cur_size
  done
fi

# choose cuda devices for backprop jobs
start_dev=$num_gpus
end_dev=$((num_gpus + num_gpus_2 - 1))
dev_list=$(seq -s, $start_dev $end_dev)
echo "cuda visible devices: $dev_list"

# launch downstream job
CUDA_VISIBLE_DEVICES=$dev_list bash launch.sh "$num_gpus_2" \
  "$yaml_file" "${extra_args[@]}"

unset CUDA_VISIBLE_DEVICES

