# RMM on LooM

[rmm.yaml](rmm.yaml) 组合 `NaiveExtractor`、`NaiveUpdater` 和 `RerankRetriever`。每个问题先通过外部 LLM 完成一次固定抽取、更新，构造候选记忆，再用策略模型采样 8 次重排。只有重排输出参与训练。

## 准备数据与模型

使用 Python 3.12+，安装匹配的 SkyRL 训练栈后，从 LooM 根目录安装本项目依赖：

```bash
python -m pip install -e '.[data,memory-bank]'
python -m loom.tasks.prepare_memory_data \
  --task rerank_bank \
  --input datas/train.jsonl \
  --output datas/rerank_bank/train.parquet
```

输入支持 JSON 数组、JSONL 或 Parquet。字段为 `M/f/q/a/context_memory`，或规范名称 `memory/fact/query/answer/context_memory`。它们对应已有记忆、最新对话、问题、参考答案及待更新的上下文记忆；不使用 MemAgent 的 HotpotQA 长文本数据。已有记忆条目须包含 `id/key/value/memory_type/tags`。

固定抽取、更新和奖励中的 QA/裁判使用外部 LLM。启动前设置：

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://your-endpoint/v1
export LLM_MODEL=your-model
```

这些变量需要在运行 Agent 的 Ray 进程中可见，不要把 API key 写进 YAML；本项目不会自动加载 `.env`。

## 启动

编辑 [run_rmm.sh](run_rmm.sh) 顶部的模型、数据、输出目录和 GPU 数量，再运行：

```bash
bash examples/RMM/run_rmm.sh
```

默认策略模型 `Qwen/Qwen3-1.7B`，单卡、每批 1 个问题、每题 8 条轨迹、输入上限 3072 token、生成上限 3072 token、训练 5 个 epoch，LR=1e-6、KL=0.01。默认关闭验证，可在脚本末尾覆盖原生参数：

```bash
bash examples/RMM/run_rmm.sh \
  trainer.policy.model.path=/path/to/model \
  "data.val_data=['/path/to/converted_dev.parquet']" \
  trainer.eval_interval=5
```

YAML 保存模块和采样设置；Python 入口将采样组大小、长度、温度映射到 SkyRL。训练参数直接列在 bash 中。多卡时修改顶部 `NUM_GPUS`，相同 GPU 上的 FSDP 和 TP=1 vLLM 引擎由 SkyRL 调度。

## 当前评分与验证边界

沿用旧 LooM 的 GRPO 标准差归一化、按序列平均 loss，以及格式、答案正确性和引用奖励。内存存储默认使用旧实现的 hash mock embedding，它没有语义检索能力；实际实验中应明确记录或替换检索实现。这里提供迁移后的启动配置，不声称复现原 RMM 论文结果。

离线测试使用固定生成后端与裁判，检查一次固定准备、多次重排、奖励及策略 token 的归属。没有执行 GPU 训练或外部 API 调用；运行环境须满足所选 SkyRL/vLLM 版本要求。
