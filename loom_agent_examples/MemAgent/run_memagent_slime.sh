#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export NO_VCS_VERSION=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
: "${CONDA_PREFIX:?Activate the slime conda environment before training}"

# The HF checkpoint, converted weights and model arguments must describe the same model.
: "${HF_CHECKPOINT:?Set HF_CHECKPOINT to the local model directory}"
: "${MEGATRON_CHECKPOINT:?Set MEGATRON_CHECKPOINT to its converted torch_dist checkpoint}"
: "${TRAIN_DATA:?Set TRAIN_DATA to the JSONL produced by slime.prepare_data}"
source "${MODEL_CONFIG:-.backend-src/slime/scripts/models/qwen3-4B.sh}"

exec "$CONDA_PREFIX/bin/python" -m loom.integrations.slime.memory_main \
    --loom-task "${TASK_YAML:-examples/MemAgent/memagent_slime.yaml}" \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node "${NUM_GPUS:-4}" \
    --colocate \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "$HF_CHECKPOINT" \
    --load "$MEGATRON_CHECKPOINT" \
    --finetune \
    --no-load-optim \
    --no-load-rng \
    --prompt-data "$TRAIN_DATA" \
    --input-key prompt \
    --rollout-shuffle \
    --rollout-batch-size 1 \
    --start-rollout-id 0 \
    --num-rollout "${NUM_ROLLOUT:-1}" \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --expert-model-parallel-size 1 \
    --rollout-num-gpus-per-engine 1 \
    --sglang-mem-fraction-static 0.5 \
    --use-dynamic-batch-size \
    --max-tokens-per-gpu 8192 \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --bf16 \
    --optimizer adam \
    --lr 1e-6 \
    --lr-decay-style constant \
    --attention-dropout 0 \
    --hidden-dropout 0 \
    --attention-backend flash \
    --pg-loss-type reinforce \
    --calculate-per-token-loss \
    --entropy-coef 0 \
    "$@"
