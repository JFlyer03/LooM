# slime 后端

`loom-slime` 使用独立的 Python 3.12 环境。LooM 负责 MemAgent 的记忆流程、数据转换和
slime custom-generate hook；采样组、GRPO、优化器、checkpoint 和权重同步由原生 slime
与 Megatron 执行。本环境不安装 VERL、SkyRL 或 vLLM。

## 安装

在 LooM 根目录执行。已有 `loom-slime` 时跳过创建命令；不要在旧 `loom` 中安装。

```bash
conda create -n loom-slime python=3.12 pip -y
conda activate loom-slime
cd /home/sunjiankang/projects/LooM
"$CONDA_PREFIX/bin/python" -m pip install "uv==0.13.7" packaging
"$CONDA_PREFIX/bin/python" scripts/prepare_unified_sources.py --backend slime
"$CONDA_PREFIX/bin/python" -m uv pip install --python "$CONDA_PREFIX/bin/python" \
  -r requirements/slime-cuda.txt
"$CONDA_PREFIX/bin/python" scripts/configure_unified_runtime.py
conda deactivate
conda activate loom-slime
"$CONDA_PREFIX/bin/python" scripts/build_unified_wheels.py --backend slime
NO_VCS_VERSION=1 "$CONDA_PREFIX/bin/python" -m uv pip install \
  --python "$CONDA_PREFIX/bin/python" -r requirements/slime.lock.txt
"$CONDA_PREFIX/bin/python" scripts/apply_slime_patches.py
"$CONDA_PREFIX/bin/python" scripts/configure_unified_runtime.py
conda deactivate
conda activate loom-slime
"$CONDA_PREFIX/bin/python" -m pip check
"$CONDA_PREFIX/bin/python" scripts/check_unified_runtime.py --backend slime --gpu
"$CONDA_PREFIX/bin/python" -m pytest tests -q
```

`slime.txt` 是直接依赖清单，`slime.lock.txt` 固定本次解析的传递依赖。源码位于
`.backend-src/`；需要保留，以使用模型配置、checkpoint 转换工具及原生 `train.py`。
名称含 `unified` 的准备工具沿用已有文件名；`--backend slime` 只准备 slime 所需产物。
缓存目录中的其他框架 wheel 不会被此清单安装。内存管理库使用 slime 上游指定的
固定提交，在当前环境的 CUDA 13 toolkit 上构建，不替换为 PyPI 的其他版本。

| 组件 | 固定版本 |
| --- | --- |
| Python / Torch / CUDA | 3.12 / 2.13.0+cu130 / 13.0 |
| slime | 79aa41a884ab0bf65222c94418f6357d3c31192b |
| Megatron | ae76c60e6c35d913867d6dd7bad0f30b128b54e7 |
| SGLang / Transformers / OpenAI | 0.5.21 / 5.12.1 / 2.6.1 |
| Ray / Transformer Engine | 2.57.0 / 2.19.0 |
| FlashInfer / FlashAttention 2 / FlashAttention 4 | 0.6.18 / 2.8.3+loom1 / 4.0.0b18 |
| nvidia-resiliency-ext | 0.7.0，固定源码本机构建 |
| torch-memory-saver | 0.0.9.post1+slime，bdab8c55f490a00a746ec34e301391617f5ca878 |

SGLang 使用原版依赖声明。FA2 的 `+loom1` 只移除与 FA4 重叠的旧 `flash_attn/cute/`
文件，保留原生二进制；slime 配套源码补丁按上游顺序应用。NVRx 官方预编译包要求的
系统 glibc 高于本机，因此从固定版本构建 checkpoint 所需部分，关闭可选 CUPTI 性能
分析扩展。补丁及产物校验记录在 `~/.cache/loom-unified/wheels/loom-patches.json`。

运行配置会为当前 conda 环境设置 CUDA_HOME、CUDA 库路径和本地 RPC 的 NO_PROXY，
不替换系统 CUDA 或 NVIDIA 驱动。迁移环境路径后需要重新执行配置工具。

## 准备模型与数据

以下使用 Qwen3-4B，模型配置必须与 HF checkpoint 匹配，不能混用 Instruct-2507 配置。
转换输出目录应为新目录。转换会使用 GPU。

```bash
"$CONDA_PREFIX/bin/hf" download Qwen/Qwen3-4B --local-dir /path/to/Qwen3-4B
source .backend-src/slime/scripts/models/qwen3-4B.sh
"$CONDA_PREFIX/bin/python" .backend-src/slime/tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" --hf-checkpoint /path/to/Qwen3-4B \
  --save /path/to/Qwen3-4B_torch_dist --bf16

"$CONDA_PREFIX/bin/python" -m loom.integrations.slime.prepare_data \
  --input /path/to/memory_tasks.parquet --output /path/to/memory_tasks_slime.jsonl
```

JSONL 每条记录包含 `prompt` 和 `metadata.instance`；后者保存完整的 context、question、
answers。转换器拒绝覆盖已有文件。

## 训练

```bash
HF_CHECKPOINT=/path/to/Qwen3-4B \
MEGATRON_CHECKPOINT=/path/to/Qwen3-4B_torch_dist \
TRAIN_DATA=/path/to/memory_tasks_slime.jsonl \
NUM_GPUS=4 NUM_ROLLOUT=2 \
  bash examples/MemAgent/run_memagent_slime.sh \
    --save /path/to/new-training-output --save-interval 1
```

默认任务为 `examples/MemAgent/memagent_slime.yaml`。通过 `TASK_YAML` 选择其他 MemAgent
任务，通过 `MODEL_CONFIG` 选择匹配的原生模型配置，例如
`.backend-src/slime/scripts/models/qwen3-0.6B.sh`。采样数、温度和 token 限制由任务 YAML
映射到原生参数；冲突会在启动 worker 前报错。

脚本用于从转换后的模型开始新训练，显式使用 `--start-rollout-id 0`。`NUM_ROLLOUT`
是总轮数，默认 1。恢复已有训练时直接调用 `loom.integrations.slime.memory_main`，
使用原来的 `--load`、`--save` 和模型/数据参数，移除 `--finetune`、`--no-load-optim`、
`--no-load-rng` 和 `--start-rollout-id 0`，让原生 checkpoint 逻辑恢复进度与优化器。
如果增加了训练总轮数，需明确学习率调度策略；本次扩展恢复验证使用原生
`--use-checkpoint-opt-param-scheduler` 保留 checkpoint 中的调度计划。

评估可追加 `--eval-prompt-data dev /path/to/dev.jsonl --eval-interval 1`。
训练会保留每个独立记忆上下文的原生 Sample；同一轨迹共享 rollout_id，终局奖励只按
一条轨迹参与组内归一化。评估只返回最终回答，避免把记忆更新步骤重复计入准确率。

当前训练入口支持 MemAgent 的单一终局目标。Memory-R1 双阶段独立优势的联合训练需要
另行适配，不能直接复用此 GRPO 入口。

## 验证范围

已在本机 H100、驱动 580.178.04 上通过原生导入、BF16 注意力与 Transformer Engine
反向、SGLang RMSNorm 检查。Qwen3-0.6B 的启动脚本完成了两轮真实训练，出现非零
策略损失与梯度，初始权重一致性检查、训练后权重同步、每轮保存和评估均成功；还验证了恢复模型、
优化器与 RNG 后继续训练并保存下一轮。

验证使用一条独立的长上下文 QA 样本、每题两条轨迹和短 token 限制，检查训练链路，
不用于评价模型效果。原始日志保存在本机 `/tmp/loom-slime-e2e/`；训练前后 QKV 参数
比较确认约 328 万个元素发生变化。

项目测试：70 项通过，1 项跳过（独立环境未安装 VERL）。多机、MoE、量化及 CUPTI
性能分析未验证或未启用；小样本验证不代表复现论文训练效果。
