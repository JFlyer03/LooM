# MemAgent 长文本问答

这个示例复用 LooM 的 VeRL 训练入口。记忆更新与最终回答使用同一套策略参数，每条候选独立更新文本记忆。Task 用回答里的 `\boxed{}` 计算终局奖励。训练按轨迹计算 GRPO 优势，验证每条轨迹保留一个最终回答。

使用仓库的 [VeRL 0.5.0 / CUDA 12.4 环境](../../requirements/verl-cu124.md)。从仓库根目录分别转换训练和验证数据，输入支持包含 `context`、`question` 和 `answers` 的文件，也接受原始 MemAgent Parquet 的 `extra_info.question` 和 `reward_model.ground_truth` 字段。

```bash
python -m loom.tasks.prepare_memory_data \
  --input /path/to/hotpotqa_train.parquet --output datas/memagent_train.parquet
python -m loom.tasks.prepare_memory_data \
  --input /path/to/hotpotqa_dev.parquet --output datas/memagent_dev.parquet
MODEL_PATH=/path/to/model bash examples/run_verl/verl_memagent.sh
```

[任务配置](verl_memagent.yaml)默认每个输入采样 4 条训练轨迹和 1 条验证轨迹，每块 2048 个输入 token。[启动脚本](verl_memagent.sh)展示单卡 FSDP + vLLM，使用 VeRL 的 `token-mean` 损失聚合。启动脚本接受 Hydra 覆盖参数，可加 `--cfg job --resolve` 检查配置或加 `trainer.total_training_steps=1` 进行单步训练。
