# loom 的记忆 Agent 运行包

本目录在 SkyRL-Agent 的基础上改造，Python 包名为 **`loom`**。
来源、许可证和改动范围见 [UPSTREAM.md](UPSTREAM.md)。

`loom_agent/` 保存旧 loom 实现作为参考；新的 MemAgent 路径不调用旧 Trainer。

## 当前实现范围

- 所有本包内部 import 和类注册路径已改为 `loom`，避免导入另一份已安装的 `skyrl_agent`。
- `memory/recurrent.py`：独立文本记忆状态、token 分块、记忆覆盖和最终回答提示词。
- `agents/memory/`：固定编排的 MemAgent、Trajectory 和复用 Dispatcher 的 Runner。
- `tasks/memory.py`：长文档 QA 字段适配、boxed 答案提取、LLM judge 或显式 exact-match 评分。
- `integrations/skyrl_train/memory_generator.py`：连接原生 SkyRL 的采样 ID 与 step-wise 输出，不重复展开采样组。
- `integrations/skyrl_train/memory_main.py`：复用 SkyRL 原生 Trainer 的新入口，不使用旧的 `SkyRLAgentPPOTrainer` 训练循环。

Memory-R1 / RMM 的原始实现仍在 `loom_agent/`，尚未迁移到新运行路径。旧 SWE、ReAct、VERL、Tinker 集成已保留并修改命名空间，但没有验证其完整训练。

## 安装边界

从 **LooM 项目根目录**运行。使用独立 Python 3.12 环境，不在原 Python 3.10 `loom` 环境中安装新的训练栈。

仅安装记忆流程和测试依赖：

```bash
python -m pip install -e '.[test,data]'
python -m pytest tests/test_memory_runtime.py -q
```

安装训练后端时，按所选 SkyRL commit 的文档安装该版本及其匹配的 PyTorch/vLLM/CUDA 依赖，再将本项目安装到同一个环境。若明确使用根目录的本地 SkyRL checkout，可将其作为本地源码安装：

```bash
python -m pip install -e './SkyRL[fsdp]'
python -m pip install -e '.[data,judge]'
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

入口配置：[examples/memory/memagent.yaml](../examples/memory/memagent.yaml)。

默认保持旧 loom 的分块参数和固定流程：

```text
2500 token一块，最多6块
每块：问题 + 当前记忆 + 新块 → 更新记忆
最后：问题 + 最终记忆 → boxed答案
```

每次生成使用新的输入上下文。每条轨迹独立持有记忆；不依赖旧实现中的全局 MemoryStore。

配置中的默认奖励 `llm_judge` 对齐旧 loom 的 boxed 答案 + 外部裁判模式，需要在进程环境中显式设置：

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...
export LLM_MODEL=...
```

不会自动加载或打印根目录的 `.env`。裁判连接失败、返回空值或格式不合法会报错，不会偷偷变成负奖励。`exact_match` 可用于离线调试，但不是原 LLM judge 的等价替代。

```bash
MODEL_PATH=/path/to/model \
DATA_PATH="$PWD/datas/memory/memagent_train.parquet" \
bash examples/memory/run_memagent_skyrl.sh
```

启动脚本默认一张可见 GPU、每个问题16条轨迹；并发由 YAML 控制，不是一次把16条长序列全部送入单卡。

这是一条最小接通示例，**不是原 MemAgent 论文或旧 loom 实验的严格复现脚本**：模型、训练批次、累积更新和后端执行差异必须单独记录，旧 `GRAD_ACC_STEPS=24` 并没有在新脚本中自动等价迁移。

训练时需要 `generator.step_wise_trajectories=true`。更新记忆与最终回答的多个上下文保留同一 `(instance_id, repetition_id)`，由原生 SkyRL 使用终局奖励计算组内优势，再分配给该轨迹的步骤。

采样组大小须与 YAML `num_trajectories` 一致，验证配置同理。不支持以重复展开输入的方式伪造额外采样。缺少采样概率时输出 `None`，不制造全零 logprob；需要 rollout 概率的校正算法应确认推理端实际返回了这些信息。

## 验证范围

测试使用可控的假推理后端，不连接外部裁判、不加载模型，检查：

- 提示词与旧 loom 一致。
- 并发采样记忆隔离、上下文重置和固定流程。
- 原始奖励未被提前转为优势。
- 原生采样只展开一次，步骤与轨迹归属正确。
- 截断轨迹的 mask、缺失 logprob、非法 token/probability 对齐。
- 数据字段转换与包内 import 来源。

完整 GPU 采样、反向传播和权重同步仍需在安装匹配的 SkyRL 环境后验证。
