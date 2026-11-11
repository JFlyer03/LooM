# MemAgent / VERL 训练与验证结果

任务：`examples/MemAgent/memagent_verl.yaml`。已完成预定的 100 步真实 GPU 训练、
四次完整验证、同配置初始模型对照及最终复核。训练日志无 traceback，进程已结束。

## 运行设置

- 环境：`loom-verl`，Python 3.12，torch 2.13.0+cu130，vLLM 0.32.0，transformers 5.16.1。
- 硬件与模型：NVIDIA H100 80GB，Qwen3-1.7B；原始模型文件大小和 SHA256 已重新校验。
- 训练数据池 1990 条，独立验证集 128 条。原始 JSON 与转换后 Parquet 逐条一致；
  问题文本及完整上下文的精确重叠均为 0。本轮为 100 个训练批次，训练批次大小为 1。
- 保留原任务的 5000 token 分块、最多 7 块、每阶段最多 1024 个输出 token、16 条训练采样
  和 4 条验证采样。完整数据通过分块和输入长度检查，无静默截断。
- 原生 VERL V1/TransferQueue、FSDP、异步 vLLM、GRPO；actor 主参数与优化器状态为 FP32，
  前反向采用 BF16 混合精度，rollout/reference 为 BF16。
- 初始模型对照和训练后评测均为验证批次 128、`max_num_seqs=48`，采样与长度设置一致。
  对照使用 `trainer.val_only=true`，不执行优化器更新。

## 完整验证结果

| 模型 | 正确最终回答 | 官方规则准确率 | 至少一次答对的题目 |
| --- | ---: | ---: | ---: |
| 同配置初始模型 | 39 / 512 | 7.6172% | 21 / 128 |
| 第 25 步 | 64 / 512 | 12.5000% | 31 / 128 |
| 第 50 步 | 95 / 512 | 18.5547% | 35 / 128 |
| 第 75 步 | 112 / 512 | 21.8750% | 35 / 128 |
| **第 100 步，预设最终比较点** | **107 / 512** | **20.8984%** | **37 / 128** |

第 100 步相对同配置初始模型提高 **13.2813 个百分点**。以 128 个问题为单位配对
bootstrap 10000 次，随机种子 42，差值的 95% 区间为 **[+7.4219, +19.3359] 个百分点**。
这支持本次 MemAgent / VERL 训练在该验证集上的准确率提升。

第 75 步是本轮最高观测值，第 100 步略低；最终主结论按预先固定的第 100 步报告。
第 25、50、75 步相对基线的差值及 95% 区间分别为：

| 步数 | 差值（百分点） | 95% 差值区间（百分点） |
| --- | ---: | ---: |
| 25 | +4.8828 | [+1.9531, +8.0078] |
| 50 | +10.9375 | [+5.8594, +16.4063] |
| 75 | +14.2578 | [+8.5938, +20.1172] |
| 100 | +13.2813 | [+7.4219, +19.3359] |

评分保持 `reward_mode: memagent`：回答末尾 300 字符、boxed 提取和官方规则归一化。
每份完整评测包含 128 题、每题 4 次采样。原始文件有 2100 个输出步骤，报告只统计
512 个最终回答。所有问题覆盖、采样数、参考答案和原生奖励均重新核对，全部通过。

早期使用验证批次 8 的基线同样为 39/512，但至少一次答对为 19/128。
该早期结果保留供审计；上述比较全部使用重新完成的同配置对照。

## 训练与保存检查

第 25、50、75、100 步的模型、优化器、随机状态、调度器和数据加载器状态均已保存，
最终 checkpoint tracker 为 100。

抽查第 0 层 Q 投影、输入 LayerNorm 和第 27 层 MLP 输出投影，均为 FP32 主参数。
第 100 步相对初始模型有 16774790 个抽查元素发生变化；转为推理用 BF16 后，仍有
1687018 个元素变化。第 25、50 步的同组 BF16 变化元素分别为 560884、1155780 个。

TensorBoard 检查确认 1～100 步记录连续；所检验的损失、梯度、学习率与耗时均为有限值，
100 步梯度范数均非零。每步采样的策略版本跨度均为 1，staleness 的均值、最小值、
最大值及 worst-case 指标均为 0。

488 个代码及配置文件的哈希与运行快照一致，请求的任务 YAML 与运行快照一致。
项目测试为 75 项通过、1 项跳过；最终 `pip check` 无冲突。

## 本机证据与复核

运行目录：`/home/sunjiankang/projects/LooM/output/MemAgent-verl/run-20261110/`。

| 文件 | 用途 |
| --- | --- |
| `completion_audit.json` | 最终完成条件的检查结果 |
| `launch.json`、`resolved_config.yaml`、`task.yaml` | 训练命令、配置和任务 |
| `evaluation_plan.json` | 预设最终比较点与对照设置 |
| `train-fp32-attempt2.log` | 完整训练日志，末尾为 100/100 |
| `global_step_{25,50,75,100}/actor/` | 模型、优化器和附加状态 |
| `weight_audit_{25,50,75,100}.json`、`audit_weights.py` | 权重变化及复核脚本 |
| `training_audit.json`、`audit_training_metrics.py` | 100 步指标检查及复核脚本 |
| `code_manifest.json`、`model_files.json` | 源码和模型校验清单 |
| `matched-baseline/launch.json`、`matched-baseline/validation.log` | 初始模型只评测对照 |
| `matched-baseline/validation/0.jsonl` | 同配置初始模型原始输出 |
| `validation/{25,50,75,100}.jsonl` | 各 checkpoint 的原始输出 |
| `validation/comparison-{25,50,75,100}.json` | 重新评分、差值和置信区间 |
| `evaluation-monitor.log` | 自动复核记录，末尾为正常结束 |

从 LooM 根目录复核最终结果：

```bash
conda activate loom-verl
RUN_DIR="$PWD/output/MemAgent-verl/run-20261110"
python scripts/evaluate_memagent_verl.py \
  --validation-data datas/memory/memagent_dev.parquet \
  --generations "$RUN_DIR/validation/100.jsonl" \
  --baseline "$RUN_DIR/matched-baseline/validation/0.jsonl" \
  --samples-per-question 4 --output "$RUN_DIR/validation/comparison-100.json"
CUDA_VISIBLE_DEVICES='' python "$RUN_DIR/audit_weights.py" --step 100
CUDA_VISIBLE_DEVICES='' python "$RUN_DIR/audit_training_metrics.py" --expected-steps 100
```

安装及重新训练命令见[VERL 独立环境说明](../requirements/verl.md)。
