#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${MODEL_PATH:?Set MODEL_PATH to a local model directory or Hugging Face model ID}"
: "${DATA_PATH:?Set DATA_PATH to the converted SkyRL training Parquet file}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python -m loom.integrations.skyrl_train.memory_main \
  data.train_data="['${DATA_PATH}']" \
  trainer.policy.model.path="$MODEL_PATH" \
  generator.task=examples/memory/memagent.yaml \
  trainer.strategy=fsdp \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_gpus_per_node=1 \
  trainer.placement.ref_num_gpus_per_node=1 \
  generator.inference_engine.num_engines=1 \
  generator.inference_engine.tensor_parallel_size=1 \
  trainer.train_batch_size=1 \
  trainer.policy_mini_batch_size=1 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.max_prompt_length=6000 \
  generator.max_input_length=6000 \
  generator.sampling_params.max_generate_length=2500 \
  generator.n_samples_per_prompt=16 \
  generator.step_wise_trajectories=true \
  generator.eval_n_samples_per_prompt=1 \
  environment.env_class=memory \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.grpo_norm_by_std=true \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.001 \
  trainer.policy.optimizer_config.lr=5e-6 \
  trainer.epochs=2 \
  trainer.eval_interval=-1 \
  trainer.logger=console \
  trainer.project_name=loom \
  trainer.run_name=loom-memagent \
  trainer.ckpt_path=output/skyrl_memagent/checkpoints \
  trainer.export_path=output/skyrl_memagent/exports \
  "$@"
