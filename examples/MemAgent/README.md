# MemAgent on LooM

这里使用新 `loom` 包运行固定流程的 MemAgent：按顺序读取文档块、覆盖文本记忆，读完后根据最终记忆回答。记忆更新和回答共用同一个可训练策略模型；阶段切换由程序控制。

- [memagent.yaml](memagent.yaml)：Agent、记忆模块、任务、采样和并发配置。
- [run_memagent.sh](run_memagent.sh)：SkyRL 原生 FSDP Trainer + vLLM 启动脚本。
- [memagent_verl.yaml](memagent_verl.yaml) / [run_memagent_verl.sh](run_memagent_verl.sh)：VeRL FSDP + 同步 vLLM V0 的单卡入口。
- [记忆模块](../../loom/modules/memory_agent.py)：从旧 `loom_agent/modules/memory_agent.py` 迁入新包的提示词、分块、记忆覆盖与循环。
- [Agent](../../loom/agents/memory/agent.py)：调用推理后端并记录每次真实输入、生成 token 和采样概率。
- [任务](../../loom/tasks/memory.py)：数据适配和最终答案评分。

新路径没有导入 `loom_agent`。旧代码中的 PyTorch 本地生成、padding 和组内优势计算已接到 SkyRL 对应接口。

本次也将旧框架的抽取、更新、检索、记忆库及 Memory-R1/RMM 编排迁入新源码；MemAgent 使用其中的循环记忆模块。完整对应关系和当前后端支持边界见[模块迁移说明](../../docs/memory_modules_migration.md)。

## 环境

在独立 Python 3.12+ 环境中，先按所选 SkyRL 版本安装匹配的训练与推理依赖，再从 LooM 根目录安装本包：

```bash
python -m pip install -e '.[data,test]'
```

若采用项目中的 SkyRL 源码，按 `SkyRL/README.md` 的 FSDP 安装说明准备该版本。原 Python 3.10 的 `loom` 环境不能直接用于这条新训练入口。当前 GPU/CUDA 是否满足所选 SkyRL/vLLM 的要求，需要单独检查；本例没有验证 V100 上的完整训练。

## 数据

所有命令从 LooM 根目录执行。支持旧 LooM 的 `context/input/answers` JSON 和作者发布的 `context/extra_info/reward_model` Parquet。先转换成新 Generator 所需的 `prompt + instance` 结构：

```bash
python -m loom.tasks.prepare_memory_data \
  --input datas/converted_hotpotqa_2000.json \
  --output datas/memory/memagent_train.parquet
```

该 JSON 在本地实际有 1990 条。转换保留全部样本和参考答案，不清洗、不划分数据、不覆盖已有输出文件。

使用作者发布的完整训练数据时，换一个输出路径：

```bash
python -m loom.tasks.prepare_memory_data \
  --input MemAgent/hotpotqa/hotpotqa_train_32k.parquet \
  --output datas/MemAgent/train_32k.parquet
```

验证数据同样单独转换 `MemAgent/hotpotqa/hotpotqa_dev.parquet`；不要将训练文件用作验证文件。`eval_*.json` 的文档索引需要先还原为实际文本，本转换器不做这一步。

## 启动

脚本采用直接列出实验参数的写法。先在顶部设置 `MODEL_PATH`、`DATA_PATH`、`OUTPUT_DIR`、`NUM_GPUS`，再启动：

```bash
bash examples/MemAgent/run_memagent.sh
```

默认模型为 `Qwen/Qwen2.5-7B-Instruct`，单卡、每个问题采样 16 条完整轨迹、每批 1 个问题、训练 2 个 epoch；YAML 中同时运行的 Agent 上限为 2。模型变量可改为本地模型路径；训练批次、学习率、epoch 和保存间隔直接在启动命令中修改。

也可在脚本末尾追加原生 SkyRL 参数，覆盖脚本中的同名设置：

```bash
bash examples/MemAgent/run_memagent.sh \
  trainer.policy.model.path=/path/to/model \
  "data.train_data=['$PWD/datas/MemAgent/train_32k.parquet']" \
  trainer.train_batch_size=128 \
  trainer.policy_mini_batch_size=8 \
  trainer.epochs=30
```

多卡时将脚本顶部 `NUM_GPUS` 改为所需卡数，FSDP policy/ref 和同数量的 TP=1 vLLM 引擎由 SkyRL 在这些 GPU 上调度。仅设置 `CUDA_VISIBLE_DEVICES` 不会改变脚本使用的 GPU 数量。

默认关闭验证。启用时设置原生参数，例如 `"data.val_data=['/path/to/converted_dev.parquet']" trainer.eval_interval=5`。采样组大小、温度和 token 限制由 Python 入口从任务 YAML 自动映射到 SkyRL；修改这些共享参数时只需编辑 YAML，命令行与 YAML 冲突会在启动 Ray 前报错。其他训练参数仍由 bash 直接列出。

## 使用 VeRL 后端

VeRL 入口复用同一套 MemAgent、记忆模块、数据和 verifier。检查所用 VeRL/vLLM 的版本组合后，在独立环境中安装 LooM：

```bash
conda activate loom-verl
cd /public/home/jiangfeibing/projects/LooM
python -m pip install -e '.[data]'
```

本次检查的环境为 Python 3.12、VeRL 0.5.0、vLLM 0.8.5、PyTorch 2.6.0+cu124。默认脚本使用同步 V0：`VLLM_USE_V1=0`、XFormers 推理注意力和 FP16 rollout。在原生 `ActorRolloutRefWorker` 中增加 [V0 MemAgent rollout](../../loom/integrations/verl/memory_v0.py)，记忆循环在 VeRL 原有的权重同步、offload 和 sleep/wake 上下文内执行；不修改已安装的 VeRL 源码。初版只支持**单卡、FSDP、TP=1、全参数训练**，其他组合会在启动 Ray 前报错。异步 V1 路径仍要求算力至少 8.0，不能用于当前 V100。

训练和参考模型参数、FSDP mixed-precision 参数均设为 FP32，模型注意力使用 SDPA，rollout 使用 FP16。VeRL 0.5 的原生 actor 前向内部仍固定使用 BF16 autocast；当前 PyTorch/V100 组合可软件模拟，已通过小矩阵前向和反向的有限值检查，但没有 BF16 硬件加速，也没有验证完整模型训练的吞吐和显存。此处不声称与 A100 上的 BF16 训练等价。

脚本使用 `datas/memory/memagent_train_verl.parquet`。本机原有训练 Parquet 由较新 PyArrow 导出，PyArrow 19 读取时出现 `Repetition level histogram size mismatch`，因此从同一 JSON 在 `loom-verl` 环境重新导出一份，保留全部 1990 条和原文件。VeRL 0.5 的原生 Trainer 要求验证 DataLoader 非空，因此另用作者 dev 数据的 128 条验证记录。若对应输出不存在，再执行转换：

```bash
python -m loom.tasks.prepare_memory_data \
  --input datas/converted_hotpotqa_2000.json \
  --output datas/memory/memagent_train_verl.parquet

python -m loom.tasks.prepare_memory_data \
  --input MemAgent/hotpotqa/hotpotqa_dev.parquet \
  --output datas/memory/memagent_dev.parquet
```

先检查最终配置；这个命令不启动 Ray、不加载 GPU 模型：

```bash
bash examples/MemAgent/run_memagent_verl.sh --cfg job --resolve
```

脚本已设置本机缓存的 Qwen3-1.7B 路径。选择空闲 GPU，先限制为 1 次训练迭代：

```bash
CUDA_VISIBLE_DEVICES=0 bash examples/MemAgent/run_memagent_verl.sh \
  trainer.total_training_steps=1
```

移除 `trainer.total_training_steps=1` 后默认训练 2 个 epoch。脚本顶部设置模型、训练文件、验证文件和输出目录，GPU 数保留为 1；也可在末尾追加原生 VeRL 参数，如 `actor_rollout_ref.actor.optim.lr=2e-6`。默认 Qwen3-1.7B、16 条训练轨迹、4 条验证轨迹，每 10 步验证和保存，关闭训练前验证。V0 engine 在 worker 线程内串行处理模型调用，YAML 的并发数不代表同一引擎并行生成。未启用旧框架的本地 Trainer。

共享的采样数、温度和输入输出预算由 Python 入口从 `memagent_verl.yaml` 映射到 VeRL；与显式命令行参数冲突会报错。`skyrl_agent.*` 是沿用现有 VeRL 集成的配置键，不会调用 SkyRL 的训练后端。

训练时每次记忆更新和回答仍是独立 `(prompt, response)` 行。适配器按实际步骤数展开输入元数据，使用 VeRL 原生 GRPO 只对每条轨迹的终局计算一次优势，再广播给该轨迹全部步骤，避免长轨迹在组内均值中被重复计数。验证只返回每条完整轨迹的最终答案行。PPO loss、旧策略概率重算、优化器、反向传播和权重同步由 VeRL 负责。

## 算法设置

默认 YAML 使用 5000-token 文档块，记忆和最终回答各最多生成 1024 token；单次输入最多 7168 token，加输出构成 8192-token 窗口。最多 7 个文档块，另加一次回答；超限直接报错。更长文档需同步调大 `memory.max_chunk_number` 与 `generator.max_iterations`。

每次更新都创建新上下文，仅输入问题、当前记忆和新块。记忆达到输出 token 上限时仍继续读下一块；最终回答若被截断，则该轨迹得分为 0、全部步骤的训练 mask 为 0，不伪造 EOS。默认关闭 Qwen3 thinking；可以显式修改，但模型和推理模式的变化应作为不同实验记录。

评分默认 `reward_mode: memagent`，使用本包内复制的[官方训练 verifier](https://github.com/BytedTsinghua-SIA/MemAgent/blob/main/verl/utils/reward_score/hotpotqa.py)：只检查回答末尾 300 个字符、提取 boxed 答案并按原规则归一化，与任意参考答案匹配得 1，否则得 0，无需外部裁判 API。它会转为小写并去掉空格等特定字符，但不会像宽松 HotpotQA EM 那样去掉所有标点和冠词。

训练保留独立的 `(prompt, response)` 步骤，同一完整轨迹共享 `(instance_id, repetition_id)`。原生 SkyRL 仅用每条轨迹的最后一步计算组内优势，再广播到该轨迹所有步骤：

```text
R_i = verifier(final_answer_i, reference_answers)
A_i = R_i - mean(R_1, ..., R_G)
```

启动脚本显式关闭标准差归一化和额外 batch 优势归一化，采用 token 平均的 PPO clipped loss、KL=0.001、LR=1e-6，无 entropy loss。这对应论文 Multi-Conv 目标的主要设置；优化器、反向传播和权重同步复用 SkyRL。[论文 §3.2–3.3](https://arxiv.org/html/2507.02259v1#S3.SS2)

默认小批次配置用于先接通训练；使用 Qwen3-1.7B、1990 条数据或单卡小批次不会自动复现原论文的 7B/14B 实验。旧 LooM 的 `GRAD_ACC_STEPS=24` 也没有对应为这里的默认更新批次。

旧 LooM 的 `llm_judge` 和宽松 `exact_match` 仍可显式选择，二者均不同于此处默认的官方规则评分。使用 `llm_judge` 时需额外安装 `.[judge]` 并设置 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`LLM_MODEL`。

## 验证

```bash
python -m pytest tests -q
bash -n examples/MemAgent/run_memagent.sh
bash -n examples/MemAgent/run_memagent_verl.sh
```

这些检查覆盖迁移后的固定流程、记忆隔离、奖励和 mask、生成上限、原生采样组映射、启动参数及包导入边界，使用可控的假推理后端。VeRL 专项测试在安装 VeRL 的环境中执行，覆盖真实 DataProto 和原生 GRPO；未安装 VeRL 时会跳过该测试文件。完整 GPU 采样、反向传播和权重同步尚未运行。
