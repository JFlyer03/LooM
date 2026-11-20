#!/usr/bin/env bash
set -x

cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1

train_data="datas/memagent_train.parquet"
test_data="datas/memagent_dev.parquet"
output_dir="outputs/memagent-verl"

MODEL=Qwen/Qwen3-1.7B
NNODES=1
SP_SIZE=1
TP_SIZE=1

uv run --isolated --locked --extra verl --env-file .env  -m loom.integrations.verl.verl_main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=false \
    "data.train_files=[\"$train_data\"]" \
    "data.val_files=[\"$test_data\"]" \
    data.train_batch_size=1 \
    data.val_batch_size=4 \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    "actor_rollout_ref.model.path=\"$MODEL\"" \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.model.use_remove_padding=false \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    "actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP_SIZE" \
    "actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    'trainer.logger=[console]' \
    trainer.project_name=loom \
    trainer.experiment_name=memagent-verl \
    trainer.n_gpus_per_node=1 \
    "trainer.nnodes=$NNODES" \
    "trainer.validation_data_dir=\"$output_dir/validation\"" \
    "trainer.default_local_dir=\"$output_dir/checkpoints\"" \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    +loom.task_yaml=examples/run_verl/verl_memagent.yaml \
    +loom.num_trajectories=4 \
    "$@"
