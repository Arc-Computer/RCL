#!/bin/bash

# RCL Quickstart Script - 5 minutes to first result
# Validates setup and runs minimal training loops

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Timing
START_TIME=$(date +%s)

echo "================================================"
echo "      RCL Quickstart - 5 Minute Validation     "
echo "================================================"
echo ""

# Function to print colored status
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Function to check elapsed time
print_elapsed() {
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    MINUTES=$((ELAPSED / 60))
    SECONDS=$((ELAPSED % 60))
    echo -e "${GREEN}⏱️  Elapsed: ${MINUTES}m ${SECONDS}s${NC}"
}

# Step 1: Environment Validation
echo "Step 1: Environment Validation"
echo "-------------------------------"

# Check Python
if ! command -v python &> /dev/null; then
    print_error "Python not found. Please install Python 3.11+"
    exit 1
fi

# Check PyTorch and CUDA
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null
if [ $? -eq 0 ]; then
    print_status "PyTorch with CUDA verified"
else
    print_error "PyTorch not installed or CUDA not available"
    echo "Run: bash scripts/install_08.sh"
    exit 1
fi

# Check HuggingFace login
if [ ! -f ~/.cache/huggingface/token ]; then
    print_warning "HuggingFace not authenticated"
    echo "Please run: huggingface-cli login"
    exit 1
else
    print_status "HuggingFace authenticated"
fi

# Count available GPUs
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
if [ "$GPU_COUNT" -lt 4 ]; then
    print_warning "Only $GPU_COUNT GPUs detected. Recommended: 4+"
    echo "Continuing with available GPUs..."
    if [ "$GPU_COUNT" -lt 2 ]; then
        print_error "Minimum 2 GPUs required for quickstart"
        exit 1
    fi
else
    print_status "$GPU_COUNT GPUs available"
fi

print_elapsed
echo ""

# Step 2: SFT Warmup Training
echo "Step 2: Minimal SFT Training (2 minutes)"
echo "-----------------------------------------"

SFT_OUTPUT="results/quickstart_sft"

# Clean previous runs
if [ -d "$SFT_OUTPUT" ]; then
    print_warning "Removing previous quickstart SFT output"
    rm -rf "$SFT_OUTPUT"
fi

# Determine GPU allocation for SFT
if [ "$GPU_COUNT" -ge 4 ]; then
    SFT_GPUS=4
else
    SFT_GPUS=$GPU_COUNT
fi

echo "Running SFT with $SFT_GPUS GPUs..."
./launch.sh $SFT_GPUS configs/run/quickstart_sft.yaml \
    output_dir=$SFT_OUTPUT 2>&1 | tee quickstart_sft.log &

SFT_PID=$!

# Monitor SFT progress
SFT_COMPLETE=false
for i in {1..120}; do
    sleep 1
    if ! kill -0 $SFT_PID 2>/dev/null; then
        SFT_COMPLETE=true
        break
    fi
    
    # Check for training progress in log
    if [ -f quickstart_sft.log ]; then
        if grep -q "Step 10/10" quickstart_sft.log 2>/dev/null; then
            sleep 2  # Let it finish cleanly
            SFT_COMPLETE=true
            break
        fi
        
        # Show progress
        CURRENT_STEP=$(grep -oE "Step [0-9]+/" quickstart_sft.log | tail -1 | grep -oE "[0-9]+" | head -1)
        if [ -n "$CURRENT_STEP" ]; then
            echo -ne "\rProgress: Step $CURRENT_STEP/10"
        fi
    fi
done

echo ""

if [ "$SFT_COMPLETE" = true ]; then
    # Check for success
    if grep -q "Training completed" quickstart_sft.log 2>/dev/null || \
       grep -q "Step 10/10" quickstart_sft.log 2>/dev/null; then
        print_status "SFT training completed successfully"
    else
        # Check if process exited successfully
        wait $SFT_PID
        if [ $? -eq 0 ]; then
            print_status "SFT training completed"
        else
            print_error "SFT training failed. Check quickstart_sft.log"
            exit 1
        fi
    fi
else
    print_error "SFT training timeout. Killing process..."
    kill $SFT_PID 2>/dev/null
    exit 1
fi

print_elapsed
echo ""

# Step 3: RL Training with vLLM
echo "Step 3: Minimal RL Training (2.5 minutes)"
echo "------------------------------------------"

RCL_OUTPUT="results/quickstart_rcl"

# Clean previous runs
if [ -d "$RCL_OUTPUT" ]; then
    print_warning "Removing previous quickstart RCL output"
    rm -rf "$RCL_OUTPUT"
fi

# Determine GPU allocation for RL
if [ "$GPU_COUNT" -ge 4 ]; then
    VLLM_GPUS=1
    TRAIN_GPUS=3
else
    VLLM_GPUS=1
    TRAIN_GPUS=$((GPU_COUNT - 1))
fi

echo "Running RL with $VLLM_GPUS vLLM GPU(s) and $TRAIN_GPUS training GPU(s)..."

# Update the config to use quickstart SFT model
sed -i.bak "s|model_name_or_path:.*|model_name_or_path: $SFT_OUTPUT|" configs/run/quickstart_rcl.yaml

./launch_with_server.sh $VLLM_GPUS $TRAIN_GPUS configs/run/quickstart_rcl.yaml \
    output_dir=$RCL_OUTPUT 2>&1 | tee quickstart_rcl.log &

RCL_PID=$!

# Monitor RL progress
RCL_COMPLETE=false
for i in {1..180}; do
    sleep 1
    if ! kill -0 $RCL_PID 2>/dev/null; then
        RCL_COMPLETE=true
        break
    fi
    
    # Check for training progress
    if [ -f quickstart_rcl.log ]; then
        if grep -q "Step 4/4" quickstart_rcl.log 2>/dev/null; then
            sleep 2  # Let it finish cleanly
            RCL_COMPLETE=true
            break
        fi
        
        # Show progress
        if grep -q "Servers initialized" quickstart_rcl.log 2>/dev/null; then
            CURRENT_STEP=$(grep -oE "Step [0-9]+/" quickstart_rcl.log | tail -1 | grep -oE "[0-9]+" | head -1)
            if [ -n "$CURRENT_STEP" ]; then
                echo -ne "\rProgress: Step $CURRENT_STEP/4"
            else
                echo -ne "\rWaiting for training to start..."
            fi
        else
            echo -ne "\rInitializing vLLM servers..."
        fi
    fi
done

echo ""

# Cleanup vLLM processes
pkill -f vllm_server 2>/dev/null || true

if [ "$RCL_COMPLETE" = true ]; then
    # Check for success
    if grep -q "Training completed" quickstart_rcl.log 2>/dev/null || \
       grep -q "Step 4/4" quickstart_rcl.log 2>/dev/null; then
        print_status "RL training completed successfully"
    else
        wait $RCL_PID
        if [ $? -eq 0 ]; then
            print_status "RL training completed"
        else
            print_warning "RL training may have issues. Check quickstart_rcl.log"
        fi
    fi
else
    print_error "RL training timeout. Killing processes..."
    kill $RCL_PID 2>/dev/null
    pkill -f vllm_server 2>/dev/null || true
    exit 1
fi

print_elapsed
echo ""

# Final Summary
echo "================================================"
echo "           Quickstart Complete!                 "
echo "================================================"

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
TOTAL_MINUTES=$((TOTAL_TIME / 60))
TOTAL_SECONDS=$((TOTAL_TIME % 60))

echo ""
print_status "Environment validated"
print_status "SFT training successful"
print_status "RL training successful"
echo ""
echo -e "${GREEN}Total time: ${TOTAL_MINUTES} min ${TOTAL_SECONDS} sec${NC}"

echo ""
echo "Next steps:"
echo "1. Review logs: quickstart_sft.log and quickstart_rcl.log"
echo "2. For full validation, increase max_steps and num_generations"
echo "3. See docs/getting-started/training-pipeline.md for production setup"

# Restore config backup
if [ -f configs/run/quickstart_rcl.yaml.bak ]; then
    mv configs/run/quickstart_rcl.yaml.bak configs/run/quickstart_rcl.yaml
fi

echo ""
print_status "Quickstart validation successful!"