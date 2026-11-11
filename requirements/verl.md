# VERL MemAgent 独立环境

使用 `loom-verl`，Python 3.12，H100；不在旧 `loom` 或 `loom-slime` 中安装。
当前环境已通过依赖检查、原生模块导入和项目测试，并完成 100 步真实训练与完整验证。

## 安装

从 LooM 根目录执行：

```bash
conda create -n loom-verl python=3.12 pip -y
conda activate loom-verl
python scripts/prepare_unified_sources.py
# flash-attn 2.8.3 的 wheel 必须匹配 CPython 3.12、torch 2.13 和 CUDA 13。
# 此索引提供匹配的预编译 wheel，避免从 PyPI 编译。
uv pip install -r requirements/verl.lock.txt \
  --index https://download.pytorch.org/whl/cu130 \
  --index https://verl-project.github.io/verl-wheelhouse/cu130/torch2.13/simple/ \
  --index-strategy unsafe-best-match
python scripts/apply_verl_patches.py
python -m pip check
python -m pytest tests -q
```

`uv` 可通过 `python -m pip install uv` 安装。锁文件的两个 editable 路径相对 LooM 根目录解析。
本机已缓存匹配的 FlashAttention wheel，可额外传入
`--find-links /home/sunjiankang/.cache/loom-unified/originals`。

该环境使用 torch 2.13.0+cu130、vLLM 0.32.0、transformers 5.16.1，VERL 源码固定在
`scripts/prepare_unified_sources.py` 指定的提交。与上游通用 `uv.lock` 相比，安装清单保留
NumPy 2.3.5、tilelang 0.1.12、nvidia-cusparselt-cu13 0.8.1，满足实际安装包的依赖声明。
不要再次直接执行上游 `uv sync` 覆盖这些版本。

本机 `/dev/shm` 的 tmpfs 不限容量，`disk_usage` 返回 total=0/free=0。
`apply_verl_patches.py` 修复 vLLM 对这种挂载的误判，同时保留有限容量挂载的空间检查。
补丁只应用于当前环境中的 vLLM 0.32.0，重新安装 vLLM 后需重新应用。

actor 使用 FP32 主参数和优化器状态，FSDP 前反向使用原生 BF16 混合精度；rollout 和
reference 使用 BF16。不能为了节省显存把 actor 主参数也载为 BF16，否则小学习率的更新
可能被参数舍入吞掉。

如果本机代理导致包下载超时，按实际网络设置 `NO_PROXY`；本次安装通过直连
`pypi.org,files.pythonhosted.org,pypi.nvidia.com,.pytorch.org` 完成。

## 数据与运行

本机已下载并校验的 Qwen3-1.7B 位于
`/home/sunjiankang/.cache/loom-models/Qwen3-1.7B`，可直接用作下面的 `MODEL_PATH`。

```bash
python -m loom.tasks.prepare_memory_data \
  --input datasets/MemFactory/converted_hotpotqa_2000.json \
  --output datas/memory/memagent_train_verl.parquet
python -m loom.tasks.prepare_memory_data \
  --input datasets/hotpotqa/eval_100.json \
  --output datas/memory/memagent_dev.parquet

CUDA_VISIBLE_DEVICES=0 MODEL_PATH=/path/to/Qwen3-1.7B \
OUTPUT_DIR="$PWD/output/MemAgent-verl/run" \
bash examples/MemAgent/run_memagent_verl.sh \
  trainer.total_training_steps=100 trainer.test_freq=25 trainer.save_freq=25 \
  data.val_batch_size=128 actor_rollout_ref.rollout.max_num_seqs=48 \
  ray_kwargs.ray_init.num_cpus=16
```

数据转换器保留原始样本且拒绝覆盖已有输出。训练 1990 条、验证 128 条，问题及完整上下文
均无重叠；来源与 SHA256 记录在 `output/MemAgent-verl/data_audit.json`。
完整任务使用原 YAML 的 16 条训练采样、4 条验证采样、5000 token 分块和 1024 token 输出上限。
训练前评测及后续定期评测均保存逐条生成结果到 `$OUTPUT_DIR/validation/`。
最终指标必须按每条轨迹的最终回答统计；同一轨迹的中间记忆输出不能重复计入准确率。

完整评测文件落盘后，可重新评分，并与训练前的同一验证集比较：

```bash
python scripts/evaluate_memagent_verl.py \
  --validation-data datas/memory/memagent_dev.parquet \
  --generations "$OUTPUT_DIR/validation/100.jsonl" \
  --baseline "$OUTPUT_DIR/validation/0.jsonl" \
  --samples-per-question 4 --output "$OUTPUT_DIR/validation/comparison.json"
```

脚本检查验证问题覆盖、每题采样数、参考答案及原始奖励与重新评分的一致性，并输出按问题
配对 bootstrap 的差值置信区间。置信区间覆盖零时，不应据此宣称有统计显著的准确率提升。

本次后台任务的命令、PID 和日志位于 `output/MemAgent-verl/run-20261110/`。
启动成功和通过单元测试不代表模型在验证集上已有提升；需要比较训练前后的实际评测记录。

2026-11-10 的训练前验证已完成：128 题、每题 4 次采样，最终回答 39/512 正确，
准确率 7.6172%；至少一次答对的题目为 19/128（14.8438%）。按问题 bootstrap 的准确率
95% 区间为 [4.2969%, 11.5234%]。`validation/baseline.json` 已复核全部奖励与答案，
完整原始记录为 `validation/0.jsonl`（包含中间记忆步骤）。

FP32 主参数的训练已连续完成完整 GRPO 步骤，包含非零梯度和权重同步。
完整训练日志为 `train-fp32-attempt2.log`，配置为 `resolved_config.yaml`；
训练前基线配置另存为 `resolved_config-baseline.yaml`。正式训练复用同一初始模型和已保存
的训练前基线，不重复评测 step 0；后续评测改为一次提交全部 128 题、最多同时执行
48 个生成序列，每题仍为 4 次采样。100 步训练已完成，四个 checkpoint 的完整评测均已复核。
预先固定的第 100 步主比较为 107/512（20.90%），同配置初始模型为 39/512（7.62%），
提高 13.28 个百分点，95% 差值区间为 [+7.42, +19.34] 个百分点。
第 75 步为本轮最高观测值 21.88%；完整结果及证据见[验证报告](../docs/memagent_verl_validation.md)。
