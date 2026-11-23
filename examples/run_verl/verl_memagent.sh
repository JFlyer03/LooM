#!/usr/bin/env bash
set -x

export CUDA_VISIBLE_DEVICES=6,7

train_data="datas/converted_hotpotqa_2000_verl.parquet"
output_dir="outputs/memagent_qwen3-1.7b_verl"
WANDB_NAME="memagent_longcontext_Qwen3-1.7B"

MODEL=Qwen3-1.7B
NNODES=1
SP_SIZE=1
TP_SIZE=1

uv run --isolated --locked --extra verl --env-file .env -m loom.integrations.verl.verl_main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=false \
    "data.train_files=[\"$train_data\"]" \
    "data.val_files=$val_files" \
    data.train_batch_size=24 \
    data.shuffle=false \
    data.val_batch_size=4 \
    data.max_prompt_length=6000 \
    data.max_response_length=2500 \
    "actor_rollout_ref.model.path=\"$MODEL\"" \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=24 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    "actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP_SIZE" \
    "actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    'trainer.logger=[console]' \
    trainer.project_name=loom \
    "trainer.experiment_name=\"$WANDB_NAME\"" \
    trainer.n_gpus_per_node=1 \
    "trainer.nnodes=$NNODES" \
    "trainer.validation_data_dir=\"$output_dir/validation\"" \
    "trainer.default_local_dir=\"$output_dir/checkpoints\"" \
    trainer.save_freq=250 \
    trainer.test_freq=10 \
    trainer.total_epochs=2 \
    +loom.task_yaml=examples/run_verl/verl_memagent.yaml \
    +loom.num_trajectories=16 \
    "$@"
