#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"
DATA_PATH="./datas/memory_bank/train.parquet"
OUTPUT_DIR="./output/Memory-R1"
NUM_GPUS=1

exec python -m loom.integrations.skyrl_train.memory_main \
    "trainer.policy.model.path=$MODEL_PATH" \
    "data.train_data=['$DATA_PATH']" \
    data.val_data=[] \
    generator.task=examples/Memory-R1/memory_r1.yaml \
    trainer.strategy=fsdp \
    trainer.placement.colocate_all=true \
    "trainer.placement.policy_num_gpus_per_node=$NUM_GPUS" \
    "trainer.placement.ref_num_gpus_per_node=$NUM_GPUS" \
    "generator.inference_engine.num_engines=$NUM_GPUS" \
    generator.inference_engine.tensor_parallel_size=1 \
    generator.inference_engine.gpu_memory_utilization=0.6 \
    trainer.train_batch_size=1 \
    trainer.policy_mini_batch_size=1 \
    trainer.micro_forward_batch_size_per_gpu=1 \
    trainer.micro_train_batch_size_per_gpu=1 \
    generator.step_wise_trajectories=true \
    generator.merge_stepwise_output=false \
    environment.env_class=memory_bank \
    trainer.algorithm.advantage_estimator=grpo \
    trainer.algorithm.grpo_norm_by_std=true \
    trainer.algorithm.advantage_batch_normalize=false \
    trainer.algorithm.policy_loss_type=regular \
    trainer.algorithm.loss_reduction=sequence_mean \
    trainer.algorithm.eps_clip_low=0.2 \
    trainer.algorithm.eps_clip_high=0.2 \
    trainer.algorithm.use_kl_loss=true \
    trainer.algorithm.use_kl_in_reward=false \
    trainer.algorithm.kl_loss_coef=0.01 \
    trainer.algorithm.use_entropy_loss=false \
    trainer.policy.optimizer_config.lr=1e-6 \
    trainer.policy.optimizer_config.num_warmup_steps=0 \
    trainer.epochs=5 \
    trainer.eval_interval=-1 \
    trainer.logger=console \
    trainer.project_name=loom \
    trainer.run_name=loom-memory-r1 \
    trainer.ckpt_interval=1000 \
    "trainer.ckpt_path=$OUTPUT_DIR/checkpoints" \
    "trainer.export_path=$OUTPUT_DIR/exports" \
    "$@"
