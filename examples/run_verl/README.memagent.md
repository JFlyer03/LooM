# MemAgent 长文本问答

这个示例复用 LooM 的 VeRL 训练入口。记忆更新与最终回答使用同一套策略参数，每条候选独立更新文本记忆。Task 用回答里的 `\boxed{}` 计算终局奖励。训练按轨迹计算 GRPO 优势，验证每条轨迹保留一个最终回答。

启动脚本首次运行时，将 MemFactory 的 `datasets/MemFactory/converted_hotpotqa_2000.json` 转为 VeRL 所需的 Parquet。默认只加载训练集。需要验证时，通过 `TEST_DATA` 传入已转换的验证 Parquet。模型路径、输出目录和并行参数可在脚本顶部设置。

```bash
bash examples/run_verl/verl_memagent.sh
# 需要验证时
TEST_DATA=datas/memagent_dev.parquet bash examples/run_verl/verl_memagent.sh
```

[任务配置](verl_memagent.yaml)每题采样 16 条训练轨迹，每块 2500 个正文 token，最多 6 块。训练提示词上限为 6000 token，每次生成上限为 2500 token。[启动脚本](verl_memagent.sh)使用单卡 FSDP + vLLM，学习率为 `5e-6`，KL 系数为 `0.001`，训练两轮，每 250 步保存一次检查点。VeRL 每批采样 24 道题，actor 按展开的训练片段分批更新。可加 `--cfg job --resolve` 检查配置，或加 `trainer.total_training_steps=1` 进行单步训练。
