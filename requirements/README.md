# 按后端隔离 conda 环境

本机 A100 / 驱动 550 / CUDA 12.4 / glibc 2.31 使用
[CUDA 12.4 适配清单与步骤](verl-cu124.md)。下面的 H100 / CUDA 13 记录属于另一套运行栈。

当前采用三个独立的 Python 3.12 环境，三个后端使用同一份 LooM 代码、模型文件和数据集。
旧 `loom` 仍被 Memory-TTT 的运行任务和派生虚拟环境使用，暂不能删除；清理前需先迁移这些依赖。
先前统一环境实验 `loom-rebuild` 已于 2026-11-10 删除，包清单备份位于本机
`~/.cache/loom-env-backups/20261110/`。

| 环境 | 训练 / 推理组合 | 状态 |
| --- | --- | --- |
| `loom-verl` | VERL 原生 FSDP + vLLM | 已完成 100 步训练及完整验证，准确率 7.62% → 20.90%；75 项测试通过 |
| `loom-slime` | slime 原生 Megatron + SGLang | 已安装；H100 单卡 MemAgent 连续两轮训练、保存和评估通过 |
| `loom-skyrl` | SkyRL 原生 FSDP + vLLM | Python 3.12 基础环境已创建，后端安装与训练验证待完成 |

## 为什么分开

选定版本的依赖有直接冲突：SGLang 0.5.21 固定 tokenizers 0.22.2、OpenAI 2.6.1、
compressed-tensors 0.18.0、Numba 0.65.1；SkyRL 0.4.0 要求 tokenizers >=0.23.1、
vLLM 0.30.0，而后者固定 compressed-tensors 0.17.0、Numba 0.65.0。
H100 满足相应 GPU 能力要求，但不会消除 Python 包版本冲突。

依据：[SGLang 依赖声明](https://pypi.org/pypi/sglang/0.5.21/json)、
[SkyRL 依赖声明](https://pypi.org/pypi/skyrl/0.4.0/json)、
[vLLM 依赖声明](https://pypi.org/pypi/vllm/0.30.0/json)。

## 安装边界

根目录 `requirements.txt` 只放公共应用依赖。每个环境先安装各自的原生 GPU 运行栈，
再从 LooM 根目录安装公共代码：

```bash
conda activate loom-verl  # 或 loom-slime / loom-skyrl
cd /home/sunjiankang/projects/LooM
python -m pip install -r requirements.txt -e '.[data,test]'
python -m pip check
```

这段命令只安装 LooM 公共部分，不等于已经安装训练后端。GPU 依赖分别按对应上游的
锁文件和构建流程安装，不将三个后端的 extras 合并。

VERL 使用本项目的[独立环境安装与运行步骤](verl.md)及 `verl.lock.txt`。
该清单在原生后端锁文件基础上修正了三处依赖声明冲突，并包含本机需要的共享内存补丁。
训练与评测命令也见该文档。不要直接执行上游 `uv sync` 覆盖已修正的版本。

SkyRL 使用 [FSDP 安装流程](https://docs.skyrl.ai/docs/getting-started/installation)；
slime 使用本项目已验证的[独立环境安装与运行步骤](slime.md)，包含匹配的 Megatron、
SGLang、CUDA 编译工具和上游补丁，无需执行会写入 `/root` 的原始构建脚本。

## 验证状态

先前统一环境已通过 71 项测试、原生模块导入、H100 BF16 内核检查、vLLM/SGLang
真实生成和 Megatron checkpoint 转换；三个后端的单步训练尚未全部通过。
这些结果不能当成新建的三个环境已经可用的证据。

slime 已在独立环境验证真实采样、GRPO、优化器更新、权重同步、checkpoint 保存及评估；
安装清单见 `slime.txt`，传递依赖锁见 `slime.lock.txt`。VERL 的独立环境训练和完整验证集
评测已有[实测记录](../docs/memagent_verl_validation.md)；SkyRL 仍待完成训练验证。
旧 V0 rollout 已删除。

`experimental-unified.txt` 仅保留统一环境实验记录；`build_unified_wheels.py --backend slime`
用于准备独立 slime 环境的二进制产物，不会安装其他框架。`requirements-full.txt` 是旧环境
快照，不用于新环境安装。
