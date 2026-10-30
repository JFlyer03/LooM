# loom 的记忆 Agent 运行包

本目录在 SkyRL-Agent 的基础上改造，Python 包名为 **`loom`**。
许可证见 [LICENSE](LICENSE)。

`loom_agent/` 保存旧 loom 实现作为参考；新的 MemAgent 路径不调用旧 Trainer。

## 当前实现范围

- 所有本包内部 import 和类注册路径已改为 `loom`，避免导入另一份已安装的 `skyrl_agent`。
- `modules/memory_agent.py`：旧 LooM 记忆模块迁入新包，包含独立文本状态、token 分块、记忆覆盖、提示词及完整循环；`memory/recurrent.py` 保留新包内的公开导出。
- `agents/memory/`：固定编排的 MemAgent、Trajectory 和复用 Dispatcher 的 Runner。
- `tasks/memory.py`：长文档 QA 字段适配、MemAgent 官方规则评分，以及可选的 LLM judge / exact-match 评分。
- `integrations/skyrl_train/memory_generator.py`：连接原生 SkyRL 的采样 ID 与 step-wise 输出，不重复展开采样组。
- `integrations/skyrl_train/memory_main.py`：复用 SkyRL 原生 Trainer 的新入口，不使用旧的 `SkyRLAgentPPOTrainer` 训练循环。

抽取、更新、检索、记忆库以及 Memory-R1 / RMM 的组合编排也已迁入新包，详见[源码迁移与接口说明](../docs/memory_modules_migration.md)。Memory-R1 的双阶段独立优势联合训练需要额外后端适配；当前 SkyRL 入口要求每次选择一个训练阶段。旧 SWE、ReAct、VERL、Tinker 集成已保留并修改命名空间，但没有验证其完整训练。

## 安装边界

从 **LooM 项目根目录**运行。使用独立 Python 3.12 环境，不在原 Python 3.10 `loom` 环境中安装新的训练栈。

仅安装记忆流程和测试依赖：

```bash
python -m pip install -e '.[test,data]'
python -m pytest tests -q
```

安装训练后端时，按所选 SkyRL commit 的文档安装该版本及其匹配的 PyTorch/vLLM/CUDA 依赖，再将本项目安装到同一个环境。若明确使用根目录的本地 SkyRL checkout，可将其作为本地源码安装：

```bash
python -m pip install -e './SkyRL[fsdp]'
python -m pip install -e '.[data]'
```

这些命令不代表当前 GPU 已满足所选 vLLM 和注意力内核要求。完整 GPU 训练尚未执行，不将通过无 GPU 测试视为后端运行兼容性证明。

不要使用 `skyrl-agent/pyproject.toml` 安装本包：它安装的是另一个保留的原始副本。这里使用根目录的新 `pyproject.toml`。

## 数据转换

保留所有样本和参考答案，只转换训练输入结构，不进行清洗或划分：

```bash
python -m loom.tasks.prepare_memory_data \
  --input datas/converted_hotpotqa_2000.json \
  --output datas/memory/memagent_train.parquet
```

输出包含 `prompt`、`instance`、`data_source`、`original_index`。其中 `instance` 保存文档、问题和答案，供自定义 Generator 使用。原 JSON 不会被覆盖。

如果输入已经含有标准化的 `context`、`question`、`answers`，也可以转换。原始 Parquet 的 `extra_info.question`、`reward_model.ground_truth` 同样可读取。文档必须是文本，不支持将超长 eval 文件中的文档索引数组误当成文本。

## 配置与启动

SkyRL 入口配置：[examples/MemAgent/memagent.yaml](../examples/MemAgent/memagent.yaml)。VeRL 对应入口为 [run_memagent_verl.sh](../examples/MemAgent/run_memagent_verl.sh)，使用 [memagent_verl.yaml](../examples/MemAgent/memagent_verl.yaml)，为 V100 提供单卡 FSDP + V0 rollout 适配。完整 GPU 训练尚未验证；具体版本和精度限制见 [MemAgent 示例](../examples/MemAgent/README.md)。

记忆库算法的对应配置和简洁启动脚本在 [RMM](../examples/RMM/README.md) 与 [Memory-R1](../examples/Memory-R1/README.md)。RMM 默认只训练重排，Memory-R1 默认训练更新阶段。

记忆编排代码从旧实现迁入新 `loom/modules/`，运行时不导入 `loom_agent`。默认采用 MemAgent 的分块与生成预算：

```text
5000 token一块，最多7块；记忆和回答各最多1024 token
每块：问题 + 当前记忆 + 新块 → 更新记忆
最后：问题 + 最终记忆 → boxed答案
```

每次生成使用新的输入上下文。每条轨迹独立持有记忆；不依赖旧实现中的全局 MemoryStore。

默认 `reward_mode: memagent` 使用本包内的官方 MemAgent 训练 verifier，不需要 API key。原始终局分数交给 SkyRL，使用组内均值中心化优势、关闭标准差归一化、token 平均 loss，优化各轮记忆更新和最终回答。

显式选择旧 loom 的 `llm_judge` 时，需安装 `.[judge]` 并设置：

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...
export LLM_MODEL=...
```

不会自动加载或打印根目录的 `.env`。裁判连接失败、返回空值或格式不合法会报错，不会偷偷变成负奖励。`exact_match` 可用于离线调试，但不是原 LLM judge 的等价替代。

```bash
bash examples/MemAgent/run_memagent.sh \
  trainer.policy.model.path=/path/to/model \
  "data.train_data=['$PWD/datas/memory/memagent_train.parquet']"
```

启动脚本顶部只保留模型、数据、输出和 GPU 数量四个变量；其余训练参数直接列在命令中。默认单卡、每个问题16条轨迹；并发由 YAML 控制。多卡时修改脚本内的 `NUM_GPUS`；完整运行参数见示例 README。

这是一条最小接通示例，**不是原 MemAgent 论文或旧 loom 实验的严格复现脚本**：模型、训练批次、累积更新和后端执行差异必须单独记录，旧 `GRAD_ACC_STEPS=24` 并没有在新脚本中自动等价迁移。

训练时需要 `generator.step_wise_trajectories=true`。更新记忆与最终回答的多个上下文保留同一 `(instance_id, repetition_id)`，由原生 SkyRL 使用终局奖励计算组内优势，再分配给该轨迹的步骤。

Python 入口从 YAML 自动映射采样组大小、采样参数和长度限制，并在启动 Ray 前拒绝冲突的命令行设置。不支持以重复展开输入的方式伪造额外采样。缺少采样概率时输出 `None`，不制造全零 logprob；需要 rollout 概率的校正算法应确认推理端实际返回了这些信息。

## 验证范围

测试使用可控的假推理后端，不连接外部裁判、不加载模型，检查：

- 提示词与旧 loom 一致。
- 并发采样记忆隔离、上下文重置和固定流程。
- 原始奖励未被提前转为优势。
- 原生采样只展开一次，步骤与轨迹归属正确。
- 截断轨迹的 mask、缺失 logprob、非法 token/probability 对齐。
- 数据字段转换与包内 import 来源。
- 抽取、更新、检索提示词与旧版一致；临时 ID、操作执行、记忆库隔离。
- Memory-R1 阶段奖励和训练 mask；RMM 共享固定候选集且只训练重排。

完整 GPU 采样、反向传播和权重同步仍需在安装匹配的 SkyRL 环境后验证。
