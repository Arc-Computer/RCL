
# vLLM Server Deployment

## Production vLLM Serving

### Training-Time Server Integration

During RL training, vLLM servers provide high-throughput generation:

```bash
# Launch training with integrated vLLM server
# Syntax: <vllm_gpus> <training_gpus> <config> [overrides]
scripts/launch_with_server.sh 1 3 configs/run/teacher_rcl.yaml \
  model_name_or_path=results/pre_rl_model
```

**Resource allocation**:
- First N GPUs: Run vLLM servers (`trainers/vllm_server.py`)
- Remaining GPUs: Run training via Accelerate/DeepSpeed
- Automatic host and port configuration in training scripts

### Standalone vLLM Server

Deploy trained ATLAS models for production inference:

```bash
python trainers/vllm_server.py \
  --model Arc-Intelligence/ATLAS-8B-Thinking \
  --port 8765 \
  --gpu_memory_utilization 0.9 \
  --dtype bfloat16
```

### Client Usage

Connect to vLLM server using the integrated client:

```python
from trainers.vllm_client.utils import VLLMClient

# Initialize client
client = VLLMClient(
    host="localhost", 
    server_port=8765, 
    group_port=51216, 
    connection_timeout=120
)

# Check server health
health = client.health_check()

# Generate responses
prompts = ["Explain quantum computing in simple terms."]
responses = client.generate(
    prompts, 
    n=1, 
    max_tokens=512, 
    temperature=0.7,
    top_p=0.9
)
```

## Server Endpoints

The vLLM server provides FastAPI endpoints:

- **`/health`**: Server health check
- **`/generate`**: Text generation endpoint  
- **`/init_communicator`**: Initialize NCCL communication for training
- **`/update_named_param`**: Update model parameters during training
- **`/reset_prefix_cache`**: Reset prefix cache between evaluation phases
- **`/close_communicator`**: Close NCCL communication

## Operational Guidelines

### GPU Resource Management
- Reserve one GPU exclusively for vLLM server during training
- Use remaining GPUs for training processes
- Ensure sufficient GPU memory for both model weights and KV cache

### Performance Optimization
- **Prefix caching**: Enable with `enable_prefix_caching=true` for repeated prompt prefixes
- **GPU memory**: Set `gpu_memory_utilization=0.9` to maximize throughput
- **Batch size**: Adjust based on available memory and latency requirements

### Monitoring and Health Checks
```bash
# Check server status
curl http://localhost:8765/health

# Monitor GPU utilization
nvidia-smi -l 1
```

### Production Deployment Considerations
- Use load balancers for multiple vLLM server instances
- Monitor memory usage and adjust `gpu_memory_utilization` as needed
- Implement proper logging and error handling
- Set up health check endpoints for orchestration systems

