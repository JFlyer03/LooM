# 旧 LooM 记忆模块接入新运行框架

旧 `loom_agent` 中的记忆模块、数据结构、存储、任务评分和算法编排已迁入新 `loom` 源码。新运行路径不导入旧包；旧目录保留供对照。模块负责记忆操作，Agent 组合模块，Runner 组织采样，SkyRL 处理优化和分布式执行。

## 源码对应

| 原实现 | 新实现 | 保留的职责 |
| --- | --- | --- |
| `loom_agent/common/registry.py` | `loom/common/registry.py` | 模块、Agent、环境注册表 |
| `loom_agent/common/utils.py` | `loom/common/utils.py` | 提示词、JSON/boxed 解析、LLM 客户端与评分辅助 |
| `loom_agent/modules/base.py` | `loom/modules/base.py` | 统一模块接口，改接异步生成回调 |
| `loom_agent/modules/memory_agent.py` | `loom/modules/memory_agent.py` | token 分块、记忆覆盖、逐段更新和最终回答 |
| `loom_agent/modules/memory_extractor.py` | `loom/modules/memory_extractor.py` | 对话格式化、抽取提示词、候选记忆 |
| `loom_agent/modules/memory_updater.py` | `loom/modules/memory_updater.py` | 临时 ID 映射、更新提示词、更新计划执行 |
| `loom_agent/modules/memory_retriever.py` | `loom/modules/memory_retriever.py` | 相似检索、候选记忆、重排提示词与选中 ID |
| `loom_agent/modules/placeholders.py` | `loom/modules/placeholders.py` | 原有抽取/更新/检索扩展类名，提供默认实现 |
| `loom_agent/agents/memory_agent.py` | `loom/agents/memory/agent.py` | MemAgent：委托循环记忆模块 |
| `loom_agent/agents/memory_r1_agent.py` | `loom/agents/memory/memory_r1_agent.py` | Memory-R1：抽取 → 更新 |
| `loom_agent/agents/rmm_agent.py` | `loom/agents/memory/rmm_agent.py` | RMM：固定抽取/更新 → 策略重排 |
| `loom_agent/envs/` | `loom/envs/` | 长文本 QA、记忆库、重排评分，以及 MemoryItem 和存储客户端 |
| 原 Env 接口 | `loom/tasks/memory.py`、`loom/tasks/memory_bank.py` | 连接新框架的 Task 生命周期 |

迁移保留任务编排，调整了执行接口：生成改为异步推理后端调用；`num_generations` 由 Runner 统一展开；每次生成记录后端实际返回的 token 和 logprob。原 `Samples` 的张量打包、PyTorch 本地采样、组内优势计算与优化器职责由训练后端承担。

## 模块组合

保留旧模块的注册名，可以直接构建并替换：

```python
from loom.modules import build_module

extractor = build_module("naive_extractor", tokenizer)
updater = build_module("naive_updater", tokenizer)
retriever = build_module("rerank_retriever", tokenizer)
```

`build_module` 也接受新包中的完整类路径。模块通过 `await generate(prompt, stage)` 使用当前策略；它们不持有 Trainer、不计算 GRPO 优势。

在任务 YAML 中选择 Agent 和模块。例如 Memory-R1 的运行配置可以采用：

```yaml
agent_cls: loom.agents.memory.MemoryR1Agent
task: loom.tasks.memory_bank.MemoryBankTask
memory:
  extractor: naive_extractor
  updater: naive_updater
  trainable_stages: [extraction]
```

其余 generator、data、dispatcher 字段与 MemAgent 示例相同。`trainable_stages` 控制哪些策略输出参与 loss；采样组数量统一由 generator 设置。RMM 对应 `loom.agents.memory.MemoryRMMAgent`、`loom.tasks.memory_bank.RerankBankTask` 和 `retriever: rerank_retriever`。

## 各算法的轨迹与奖励

- MemAgent：各次记忆更新和最终回答属于一条完整轨迹。最终 boxed 答案得分产生一个组内优势，覆盖全部生成步骤。
- Memory-R1：每条轨迹依次生成抽取文本和更新计划，保留两个独立的阶段分数。原环境的默认奖励主要是 JSON 格式及更新操作有效性，准确性奖励在旧代码中关闭；本次迁移保留这一点。默认评分跳过不影响分数的额外 QA 请求。
- RMM：每个输入先进行一次固定抽取/更新，得到相同候选集，再采样 G 次策略重排。固定步骤使用外部 LLM，不记为策略训练 token；只有 rerank 输出参与优化。评分保留格式、QA 正确性和引用奖励。

当前 SkyRL 原生 step-wise 路径使用每条轨迹的一个终局优势。因此 MemAgent 和单个重排目标可以直接适配；Memory-R1 在该入口每次须选择一个 `trainable_stages`。同时训练抽取和更新、分别计算两组优势，需要额外的多目标适配器；现有入口对该配置直接报错，避免用更新奖励覆盖抽取奖励。Runner 本身保留两个阶段的原始奖励，供后续适配使用。

## 记忆存储与数据

`MemoryItem`、`ConversationMessage`、枚举、Neo4j/Milvus 客户端和统一 `MemoryStore` 都已迁入新包。每个训练轨迹使用独立的内存存储；不复用旧的进程级全局 store。反序列化不修改源数据，mock embedding 使用局部随机数生成器，不修改训练的 NumPy 随机状态。

默认 mock embedding 保留旧实现的确定性 hash 向量行为，它不具有语义检索能力。真实 embedding/数据库客户端源码已保留，真实服务连接未验证；显式请求持久化存储而连接失败时会报错。

MemAgent 只使用文本记忆，不依赖这些数据库和 embedding。记忆库算法可安装 `.[memory-bank]`；真实数据库接口另有 `.[memory-db]`。导入新模块不会自动读取 `.env` 或建立外部连接。

数据转换入口支持原来的短字段 `M/f/q/a` 和规范字段 `memory/fact/query/answer/context_memory`：

```bash
python -m loom.tasks.prepare_memory_data \
  --task memory_bank --input datas/train.jsonl --output datas/memory_bank/train.parquet
```

RMM 可选择 `--task rerank_bank`。长文本 QA 仍使用默认的 `--task longcontext`。

## 本次可用入口与验证范围

完整的 MemAgent 训练 YAML 和启动脚本在 [examples/MemAgent](../examples/MemAgent/README.md)。旧五个训练脚本继续保留原有实验入口；新入口使用上述模块体系，不调用旧 Trainer。

MemAgent 同时提供 `run_memagent_verl.sh` / `memagent_verl.yaml`。VeRL 适配保留独立的记忆上下文行，按真实步骤数对齐 DataProto，只把每条轨迹的终局交给原生 GRPO，再将优势广播到步骤；验证仅保留最终答案。默认使用 `memory_v0.py` 的单卡 FSDP 同步 V0 适配，记忆循环在原生 worker 的权重同步上下文内执行，复用原生 sharding manager；当前限制 TP=1、全参数训练。`rollout_data.py` 为 V0 和异步路径共享张量打包；异步 V1 仍要求算力至少 8.0。完整 GPU 训练尚未验证。

对应的记忆库启动配置在 [examples/RMM](../examples/RMM/README.md) 和 [examples/Memory-R1](../examples/Memory-R1/README.md)，分别选择重排目标和更新目标；Memory-R1 可在 YAML 中切换为抽取目标。

离线测试覆盖全部迁移模块的提示词、注册、临时 ID、存储操作与隔离、MemAgent 循环、Memory-R1 阶段奖励及 mask、RMM 固定准备与重排分工。测试使用受控推理和裁判；完整 GPU 训练、真实数据库和外部模型效果尚未验证。
