#!/usr/bin/env bash

###############################################################################
# Usage:
#    bash run_gsm8k_single_layer.sh <MODEL_NAME> [<GPU_IDs>]
#
# Examples:
#    bash run_gsm8k_single_layer.sh deepseek-qwen-14b "4 5 6"
#    bash run_gsm8k_single_layer.sh deepseek-qwen-14b          # uses all GPUs
###############################################################################

# 1) Parse script arguments
MODEL="$1"
if [ -z "$2" ]; then
    GPUS=(0 1 2 3 4 5 6 7)
else
    GPUS=($2)
fi

if [ -z "$MODEL" ]; then
    echo "Usage: $0 <MODEL_NAME> [<GPU_IDs>]"
    echo "Example: $0 deepseek-qwen-14b \"4 5 6\""
    exit 1
fi

###############################################################################
# Global Configuration
###############################################################################
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

PIDS=()
start_time=$(date +%s)

get_free_gpu() {
    while true; do
        for gpu in "${GPUS[@]}"; do
            process_count=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader |
                grep -c "$(nvidia-smi --id="$gpu" --query-gpu=uuid --format=csv,noheader)")
            free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits |
                awk "NR==$((gpu+1))")

            if [ "$free_mem" -gt "$MEMORY_THRESHOLD" ] && [ "$process_count" -lt "$MAX_PROCESSES_PER_GPU" ]; then
                echo "${gpu}:${free_mem}"
                return
            fi
        done
        echo "All GPUs [${GPUS[*]}] are busy or below memory threshold ($MEMORY_THRESHOLD MB). Retrying in 5s..."
        sleep 5
    done
}

cleanup() {
    echo "Terminating all running processes..."
    kill ${PIDS[*]} 2>/dev/null
    exit
}

trap cleanup SIGINT SIGTERM

###############################################################################
# Model-Specific Settings
###############################################################################
case "$MODEL" in
    "deepseek-llama3-8b")
        MEMORY_THRESHOLD=80000
        MAX_PROCESSES_PER_GPU=1
        LAYER_SEQ=$(seq 0 31)
        ;;
    "deepseek-qwen-14b")
        MEMORY_THRESHOLD=80000
        MAX_PROCESSES_PER_GPU=1
        LAYER_SEQ=$(seq 0 63)
        ;;
    "deepseek-qwen-1.5b")
        MEMORY_THRESHOLD=40000
        MAX_PROCESSES_PER_GPU=2
        LAYER_SEQ=$(seq 0 27)
        ;;
    *)
        echo "Unknown model '$MODEL'. Aborting..."
        exit 1
        ;;
esac

CONTROL="thinking_length_mlp"
DIRECTION_WEIGHTS=(-1 1)

###############################################################################
# Main: Run only thinking_length_mlp across layers
###############################################################################
for weight in "${DIRECTION_WEIGHTS[@]}"; do
    for layer in $LAYER_SEQ; do

        res=$(get_free_gpu)
        free_gpu=$(echo "$res" | cut -d':' -f1)
        free_mem=$(echo "$res" | cut -d':' -f2)

        echo "MODEL=$MODEL | CONTROL=$CONTROL | LAYER=$layer | WEIGHT=$weight"
        echo "Assigning GPU $free_gpu (Free Memory: ${free_mem}MB)"

        CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$free_gpu \
        nohup python evaluate_gsm8k_single_layer.py \
            --model "$MODEL" \
            --control "$CONTROL" \
            --layer "$layer" \
            --direction_weight "$weight" \
            > "${LOG_DIR}/${MODEL}_${CONTROL}_layer${layer}_weight${weight}.log" 2>&1 &

        PIDS+=($!)
        sleep 20

        total_jobs_allowed=$(( ${#GPUS[@]} * MAX_PROCESSES_PER_GPU ))
        while [ "$(jobs -p | wc -l)" -ge "$total_jobs_allowed" ]; do
            sleep 10
        done
    done

done

###############################################################################
# Wait for all background jobs to finish
###############################################################################
wait

echo "All experiments completed!"

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))

hours=$(( elapsed / 3600 ))
minutes=$(( (elapsed % 3600) / 60 ))
seconds=$(( elapsed % 60 ))

echo "Total run time: ${hours}h ${minutes}m ${seconds}s"
