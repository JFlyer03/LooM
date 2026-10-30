#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=XFORMERS

MODEL_PATH="/public/home/jiangfeibing/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master"
DATA_PATH="./datas/memory/memagent_train_verl.parquet"
VAL_DATA_PATH="./datas/memory/memagent_dev.parquet"
OUTPUT_DIR="./output/MemAgent-verl"
NUM_GPUS=1

exec python -m loom.integrations.verl.memory_main \
    +skyrl_agent.task_yaml=examples/MemAgent/memagent_verl.yaml \
    "actor_rollout_ref.model.path=$MODEL_PATH" \
    "data.train_files=['$DATA_PATH']" \
    "data.val_files=['$VAL_DATA_PATH']" \
    data.train_batch_size=1 \
    data.val_batch_size=1 \
    data.dataloader_num_workers=0 \
    data.return_raw_chat=true \
    data.filter_overlong_prompts=false \
    data.truncation=error \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_torch_compile=false \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=true \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=fp32 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=fp32 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype=fp32 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer_dtype=fp32 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.model.use_remove_padding=false \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    +actor_rollout_ref.model.override_config._attn_implementation=sdpa \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.rollout.enable_chunked_prefill=false \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    +actor_rollout_ref.ref.fsdp_config.model_dtype=fp32 \
    +actor_rollout_ref.ref.fsdp_config.mixed_precision.param_dtype=fp32 \
    +actor_rollout_ref.ref.fsdp_config.mixed_precision.reduce_dtype=fp32 \
    +actor_rollout_ref.ref.fsdp_config.mixed_precision.buffer_dtype=fp32 \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=false \
    algorithm.use_kl_in_reward=false \
    "trainer.n_gpus_per_node=$NUM_GPUS" \
    trainer.nnodes=1 \
    trainer.project_name=loom \
    trainer.experiment_name=memagent-verl \
    'trainer.logger=[console]' \
    trainer.val_before_train=false \
    trainer.test_freq=10 \
    trainer.save_freq=10 \
    trainer.total_epochs=2 \
    "trainer.default_local_dir=$OUTPUT_DIR" \
    "$@"
