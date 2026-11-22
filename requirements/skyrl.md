# SkyRL-Train 环境依赖调研

## 本机独立环境安装进度（2026-11-22）

已创建 `loom-skyrl`，路径为 `/disk0/home/jiangfeibing/miniconda3/envs/loom-skyrl`。
已安装 Python 3.12.16、SkyRL 0.3.0、PyTorch 2.11.0+cu128、vLLM 0.23.0+cu129、
Ray 2.56.0、Transformers 5.8.0；依赖一致性检查和 GPU 0 的 BF16 矩阵运算通过。
FlashAttention 2.8.3 已使用 CUDA 12.8.93 / GCC 9 为 A100 编译安装，GPU 前向、反向
及与 PyTorch SDPA 的数值对比通过；LooM SkyRL 入口导入通过。单步训练验证进行中。

SkyRL 0.3.0 的 router 状态日志访问了 vllm-router 0.1.17 不存在的可选字段，
本环境应用了 [一行兼容补丁](skyrl-router-0.1.17.patch)，使缺失字段按 False 处理。
该改动只影响状态日志分支。重新安装 SkyRL 后需重新应用此补丁：

```bash
patch -d "$CONDA_PREFIX/lib/python3.12/site-packages" -p1 < requirements/skyrl-router-0.1.17.patch
```

依赖输入见 [skyrl-runtime.in](skyrl-runtime.in)，精确版本见
[skyrl-runtime.lock.txt](skyrl-runtime.lock.txt)。该配置显式补充 dense 模型需要的
FSDP/vLLM GPU 栈，使用 `skyrl[skyrl-train]`，不安装 `skyrl[fsdp]` 中额外的 GDN/Mamba 内核。

激活命令：`conda activate loom-skyrl`。激活脚本自动设置本环境的 `CUDA_HOME` 和
`LD_LIBRARY_PATH`、CUDA 头文件/链接路径、GCC 9 编译参数及本机地址的代理绕过列表，
退出时恢复原值；LooM 源码通过环境内的 `.pth` 文件加载。
CUDA 12.9 forward-compat 用户态库位于本项目 `.backend-src/cuda-compat-12-9/`，
不会替换系统 550 驱动。此目录需要保留。

以下为安装前的依赖调研记录。

采集日期：2026-11-19。目标是 LooM 的 FSDP/FSDP2 训练 + vLLM 推理，
本机驱动 550.90.07、CUDA Toolkit 12.4.131、glibc 2.31、Python 3.12。
本轮只收集依赖和静态核对接口，没有安装 SkyRL，没有修改 VERL 环境或根项目配置。

## 包名与 extra 的区别

- 旧发行包叫 `skyrl-train`，使用 `skyrl_train.*` 命名空间。
- 新的统一发行包叫 `skyrl`；本项目当前使用 `skyrl.train.*` 和
  `skyrl.backends.skyrl_train.*` 接口，应以这个包为主要适配对象。
- LooM 的 `--extra skyrl-train` 是本项目自定义的依赖组名称，不等于必须安装旧的
  `skyrl-train` 包。对于完整 FSDP + vLLM 栈，应收集 `skyrl[fsdp]`；
  仅选择上游 `skyrl[skyrl-train]` 不包含完整的 FSDP/vLLM GPU 依赖。

依据：[SkyRL 包说明](https://pypi.org/project/skyrl/)、
[旧 SkyRL-Train 包说明](https://pypi.org/project/skyrl-train/)、
[作为依赖库使用](https://docs.skyrl.ai/docs/getting-started/library)。

## 已核对的版本组合

以下是 PyPI 发布元数据中的声明，不是已经安装成功的环境。
Torch 与 torchvision 的配对同时核对了相应 vLLM 发布版的元数据。
新包行按 `skyrl[fsdp]`，旧包行按 `skyrl-train[vllm]` 采集。

| 包 / 版本 | Python | Ray | PyTorch | vLLM | torchvision | FlashAttention |
| --- | --- | --- | --- | --- | --- | --- |
| skyrl-train 0.1.0 | ==3.12.* | 2.48.0 | 2.7.0 | 0.9.2 | 0.22.0 | 未固定 |
| skyrl-train 0.2.0 | ==3.12.* | 2.48.0 | 2.7.1 | 0.10.1.1 | 0.22.1 | 未固定 |
| skyrl-train 0.3.1 | ==3.12.* | 2.51.1 | 2.8.0 | 0.11.0 | 0.23.0 | 2.8.3 |
| skyrl 0.1.0 | >=3.11 | 2.51.1 | 2.9.1 | 0.16.0 | 0.24.1 | 2.8.3 |
| skyrl 0.2.0 | >=3.11 | 2.51.1 | 2.10.0 | 0.19.0 | 0.25.0 | 2.8.3 |
| skyrl 0.3.0 | >=3.11 | 2.56.0 | 2.11.0 | 0.23.0 | 0.26.0 | 2.8.3 |
| skyrl 0.4.0 | >=3.12,<3.15 | 2.57.0 | 2.13.0 | 0.30.0 | 0.28.0 | 2.8.3 |
| skyrl 0.5.0 | >=3.12,<3.15 | 2.57.0 | 2.14.0 | 0.33.0 | 0.29.0 | 2.8.3.post1 |

每个版本的官方元数据地址、wheel SHA256、所选 extra 的完整直接依赖、
vLLM 传递依赖和接口检查结果保存在 [JSON 快照](skyrl-dependency-audit.json)。
8 个 SkyRL wheel 均已下载并对照官方 SHA256 校验；未下载 GPU 大包。

## 当前 LooM 接口与版本的关系

静态检查覆盖本项目 `skyrl_train_backend.py`、`skyrl_train_main.py`、`trainer.py`
中的 18 处 SkyRL 导入，并追踪 `__init__.py` 的相对通配符重导出。

| 版本范围 | 静态检查结果 |
| --- | --- |
| skyrl-train 0.1.0 / 0.2.0 / 0.3.1 | 当前 18 处导入所指的新命名空间模块均不存在，需要迁移接口 |
| skyrl 0.1.0 / 0.2.0 | 缺少 3 项所需接口，不能原样接入 |
| skyrl 0.3.0 / 0.4.0 / 0.5.0 | 本次检查的导入名称均可在 wheel 中找到 |

0.1.0/0.2.0 缺少的名称为：

```text
skyrl.backends.skyrl_train.inference_servers.utils.resolve_policy_model_name
skyrl.backends.skyrl_train.inference_servers.base.InferenceEngineInterface
skyrl.backends.skyrl_train.inference_servers.engine_utils.get_sampling_params_for_backend
```

这只说明名称存在，不证明函数签名、配置、生成结果格式、权重同步和训练行为兼容。
因此，0.3.0 是本次检查到的最早接口候选，不是本机已验证可用的安装版本。

## 需要收集并锁定的依赖层

1. GPU 栈：`torch`、`torchvision`、与其 CUDA/C++ ABI 匹配的 `vllm`、
   `flash-attn`、`flashinfer-python` 及所选版本需要的 JIT/cache、CUDA 库。
2. 训练运行时：`ray`、`torchdata`、`tensordict`、`accelerate`、`peft`、
   `transformers`、`tokenizers`、`hydra-core`、`omegaconf`、`skyrl-gym`。
3. 数据与服务：`datasets`、`pyarrow`、`polars`、`s3fs`、`gcsfs`、
   `fastapi`、`uvicorn`、`orjson`、`pybase64`、`vllm-router`。
4. 跟踪与辅助：`loguru`、`tqdm`、`tensorboard`、`wandb`、`func_timeout`、
   `jaxtyping`、`hf_transfer`、`debugpy`；需要编译时还有 `ninja`、`pybind11`、`setuptools`。

以上按版本取舍，不能把所有版本的依赖拼成一张安装表。例如 0.5.0 还声明
`nixl>=1.5.0`、`flash-linear-attention==0.6.0`、`causal-conv1d==1.7.0`、
`flashinfer-python==0.7.2.post1`，以及分架构的 FlashInfer JIT 包。
其基础依赖包括 `transformers>=5.6.1,<=5.16.1`、`tokenizers>=0.23.1`、
`peft==0.21.0`；不能沿用 VERL 环境的 Transformers 4.51.3 / PEFT 0.15.2。
精确条件见 JSON 快照中的 `requires_dist_for_linux_py312`。

## 本机约束与已有脚本

- 当前上游推荐 CUDA 13 / R580+ 驱动，并推荐 Python 3.12、Ray 2.57.0。
  当前 550 驱动不满足这条推荐安装路线，不能直接套用最新版 CUDA 13 环境。
- 上游不建议 Ray 2.47.0/2.47.1，原因是 uv + Ray 集成问题。
  VERL 环境的 Ray 2.47.1 不能作为 SkyRL 的默认选择。
- 需要 `build-essential` 和 NUMA 支持。本机存在 `libnuma.so.1`，未找到
  `/usr/include/numa.h`；若需要本机编译，仍需检查开发文件。
- uv + Ray 通常需要设置
  `RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook`，
  并保证 driver 与 worker 的 Python、Ray、依赖来源一致。

依据：[上游安装与 Ray 集成说明](https://docs.skyrl.ai/docs/getting-started/installation)。

本仓库 `run_skyrl_swe.sh` 仍覆盖为 vLLM 0.9.2 / Torch 2.7.0 /
Transformers 4.53.0，并指定 torch2.7、CXX11 ABI true 的 FlashAttention wheel。
vLLM 0.9.2 的 PyTorch 声明确实是 2.7.0，但这套旧 GPU 组合与根配置当前的
新 SkyRL / CUDA 13 依赖不是同一代，不能仅凭 `--with` 认定已经兼容。

PyTorch 2.7 的官方安装表列出 cu118/cu126/cu128，没有 cu124；本机已验证的
Torch 2.6.0+cu124 不能直接满足上表中的 Torch 固定版本。
见 [PyTorch 历史版本](https://pytorch.org/get-started/previous-versions/)。

CUDA 12.6/12.8 也不能仅凭 `nvidia-smi` 显示 12.4 就断言完全不能运行：
CUDA 12.x 有小版本兼容机制，但新 PTX/JIT 和驱动功能有额外限制。
这类路线需要单独验证 vLLM/Triton 内核，不能当作已经可用。
见 [NVIDIA 小版本兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。

## 后续适配范围

保留当前 LooM 接口时，先从 `skyrl>=0.3.0` 的确定发布版选择 GPU 栈，
再验证本机驱动、wheel 的 glibc/ABI、NUMA、Ray worker、权重同步和单步训练。
如果坚持完全复用 Torch 2.6 / cu124 / vLLM 0.8.5，需另查更早的 Git 提交并回移
LooM 接口；本次检查的正式发行版中没有确认到这一现成组合。

后续 LooM 的 `skyrl-train` extra 应依赖选定版本的 `skyrl[fsdp]`，与 VERL 分环境。
若通过 PyPI 使用 SkyRL，自定义 wheel 源和 uv override 需要在 LooM 项目中显式配置，
不能假设上游的 uv 配置会自动继承，详见上游作为依赖库使用的说明。

本轮不提供安装锁文件：仅完成发布元数据、wheel 校验和静态 API 调研；
没有运行依赖求解、原生导入、GPU 内核、Ray 或训练验证。
