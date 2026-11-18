# Memory-R1 on LooM

[memory_r1.yaml](memory_r1.yaml) 组合 `NaiveExtractor` 和 `NaiveUpdater`，每条轨迹依次抽取候选记忆、生成更新计划。两个步骤共享策略模型，并保留各自的原始奖励。

## 准备数据与模型

使用 Python 3.12+，安装匹配的 SkyRL 训练栈后，从 LooM 根目录执行：

```bash
python -m pip install -e '.[data,memory-bank]'
python -m loom.tasks.prepare_memory_data \
  --task memory_bank \
  --input datas/train.jsonl \
  --output datas/memory_bank/train.parquet
```

输入支持 JSON 数组、JSONL 或 Parquet，使用 `M/f/q/a/context_memory` 或 `memory/fact/query/answer/context_memory` 字段。已有记忆条目须包含 `id/key/value/memory_type/tags`。它与 MemAgent 的 HotpotQA 长文本输入不同。

## 启动与阶段选择

编辑 [run_memory_r1.sh](run_memory_r1.sh) 顶部的模型、数据、输出目录和 GPU 数量，然后运行：

```bash
bash examples/Memory-R1/run_memory_r1.sh
```

默认模型 `Qwen/Qwen2.5-3B-Instruct`，单卡、每批 1 个问题、每题 8 条轨迹、输入上限 3072 token、生成上限 2048 token、训练 5 个 epoch，LR=1e-6、KL=0.01。验证默认关闭。

当前 YAML 选择更新阶段：

```yaml
memory:
  trainable_stages: [update]
```

训练抽取阶段时，将该项改为 `[extraction]`。两个步骤都会生成，但只有所选阶段的输出进入 loss，组内优势使用该阶段自己的奖励。当前 SkyRL 原生 step-wise 接口不支持 `[extraction, update]` 分别计算优势的联合训练，会对该设置报错。

调整训练参数可直接编辑 bash，或追加原生 SkyRL 参数：

```bash
bash examples/Memory-R1/run_memory_r1.sh \
  trainer.policy.model.path=/path/to/model \
  "data.val_data=['/path/to/converted_dev.parquet']" \
  trainer.eval_interval=5
```

YAML 控制模块、训练阶段、采样和并发；采样参数自动映射到后端，不需要在 bash 中重复填写。更换训练阶段后，使用不同的 `trainer.ckpt_path` 和 `trainer.export_path` 保存对应实验。

## 当前评分与验证边界

配置沿用旧 LooM 的 GRPO 标准差归一化和按序列平均 loss。旧实现的抽取/更新奖励主要检查 JSON 格式和更新操作有效性，未启用下游 QA 准确性奖励；默认流程无需外部裁判 API。组内所有分数相同时，GRPO 策略优势为零。这里提供已有模块实现的启动配置，不声称完整复现 Memory-R1 论文的训练目标。

记忆库使用每条轨迹独立的内存存储及旧版 hash mock embedding。离线测试覆盖模块组合、阶段奖励、mask 和原生采样组；没有执行 GPU 训练，运行环境仍需满足 SkyRL/vLLM 版本要求。
