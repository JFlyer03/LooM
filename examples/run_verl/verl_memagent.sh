#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
# Expand the launch arguments after uv loads .env.
exec uv run --no-project --python "$(command -v python)" --env-file .env bash -s -- "$@" <<'LAUNCH'
set -euo pipefail

model_path=${MODEL_PATH:-Qwen/Qwen3-1.7B}
train_data=${DATA_PATH:-datas/memagent_train.parquet}
val_data=${VAL_DATA_PATH:-datas/memagent_dev.parquet}
output_dir=${OUTPUT_DIR:-outputs/memagent-verl}

exec python -m loom.integrations.verl.verl_main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=false \
    "data.train_files=['$train_data']" \
    "data.val_files=['$val_data']" \
    data.train_batch_size=1 \
    data.val_batch_size=4 \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    "actor_rollout_ref.model.path=$model_path" \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.model.use_remove_padding=false \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    'trainer.logger=[console]' \
    trainer.project_name=loom \
    trainer.experiment_name=memagent-verl \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    "trainer.validation_data_dir=$output_dir/validation" \
    "trainer.default_local_dir=$output_dir/checkpoints" \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    +loom.task_yaml=examples/run_verl/verl_memagent.yaml \
    +loom.num_trajectories=4 \
    "$@"
LAUNCH
