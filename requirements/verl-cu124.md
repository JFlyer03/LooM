# 本机 CUDA 12.4 的 VERL 环境

适用于本机 NVIDIA A100、550.90.07 驱动、CUDA Toolkit 12.4.131、glibc 2.31，
环境名为 `loom-verl`，Python 3.12。此组合独立于原来的 H100 / CUDA 13 环境。

| 组件 | 版本 |
| --- | --- |
| PyTorch | 2.6.0+cu124 |
| torchvision / torchaudio | 0.21.0+cu124 / 2.6.0+cu124 |
| VERL | 0.5.0 |
| vLLM / xFormers | 0.8.5 / 0.0.29.post2 |
| FlashAttention | 2.7.4.post1，cu12 / torch2.6 / CPython 3.12 / CXX11 ABI false |
| Transformers / PEFT | 4.51.3 / 0.15.2 |
| Ray / TensorDict | 2.47.1 / 0.8.3 |
| NumPy | 1.26.4 |
| PyMilvus | 2.5.10 |

vLLM 0.8.5 的 CUDA 12.4 二进制版本见[上游安装文档](https://docs.vllm.ai/en/v0.8.5/getting_started/installation/gpu.html)。
VERL 的相应 CUDA 12.4 / PyTorch 2.6 / FlashAttention 2.7.4 组合见
[上游环境说明](https://verl.readthedocs.io/en/v0.5.x/start/install.html)。

## 安装

VeRL 启动脚本使用根目录的 `pyproject.toml` 和 `uv.lock`。从 LooM 根目录可直接检查 MemAgent 配置：

```bash
bash examples/run_verl/verl_memagent.sh --cfg job --resolve
```

如需在固定的 Conda 环境中执行调试或数据转换，可以用以下步骤安装。已有 `loom-verl` 时跳过创建步骤：

```bash
conda create -n loom-verl python=3.12 pip -y
conda activate loom-verl
python -m pip install uv
UV_HTTP_TIMEOUT=1800 UV_CONCURRENT_DOWNLOADS=4 \
  uv pip sync requirements/verl-cu124.lock.txt --python "$CONDA_PREFIX/bin/python"
python -m pip check
python -m pytest tests/test_verl_backend_loading.py -q
```

`uv pip sync` 会将所选环境同步到清单并移除多余包，只能对专用的 `loom-verl` 环境使用。
PyTorch 的三个包使用官方 cu124 wheel URL，其余包使用 PyPI。
FlashAttention 使用官方发布的匹配 wheel，避免安装要求更新 glibc 的 CUDA 13 wheel。

此清单包含 LooM 公共依赖及兼容的存储客户端。不要再叠加原来的 `requirements.txt`、
`verl.lock.txt` 或其他训练后端的 extras：vLLM 0.8.5 的 OpenTelemetry 依赖要求
`protobuf<5`，而 `pymilvus>=2.6.6` 要求 `protobuf>=5.27.2`，因此单独固定为 2.5.10。
也不要运行只针对 vLLM 0.32.0 的 `scripts/apply_verl_patches.py`。

调整版本时修改 `verl-cu124.in`，再重新解析：

```bash
UV_HTTP_TIMEOUT=1800 uv pip compile requirements/verl-cu124.in \
  --python "$CONDA_PREFIX/bin/python" -o requirements/verl-cu124.lock.txt
```

## 检查 GPU

GPU 检查必须在能够访问 `/dev/nvidia*` 的宿主环境执行；沙箱内的失败不能用于判断驱动故障。

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=0 python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); x=torch.randn(128,128,device="cuda",dtype=torch.bfloat16); assert torch.isfinite(x@x).all()'
CUDA_VISIBLE_DEVICES=0 python -c 'import loom, verl, vllm._C, flash_attn_2_cuda; print("native imports OK")'
```

## 本次验证

2026-11-19 在 `loom-verl` 中完成：

- 安装完整锁文件，217 项包版本核对无差异，`pip check` 无冲突；后端加载测试 1 项通过。
- 识别 8 张 A100；GPU BF16 矩阵运算通过。
- FlashAttention 因果注意力的输出与 PyTorch SDPA 对比通过。
- LooM 的 VERL 适配器、训练器、异步管理器及原生扩展导入通过；
  自定义 Ray 服务的生成接口检查通过。
- NCCL 单进程 FSDP 前向、反向及 AdamW 参数更新通过；world size 为 1，
  PyTorch 使用 NO_SHARD，因此不能视为多卡分片训练验证。
- 独立进程使用 `CUDA_VISIBLE_DEVICES=0`：vLLM V1 引擎、FlashAttention 后端，
  本地随机初始化的小型 Llama 成功生成 4 个 token：`[70, 70, 37, 11]`。
- `python -m loom.integrations.verl.verl_main_ppo --help` 成功加载训练配置。
- 同一进程使用 `CUDA_VISIBLE_DEVICES=0`，连续执行 BF16、FlashAttention、
  LooM 适配器、NCCL 单进程 FSDP 更新和 vLLM 生成，最终状态为 `passed`。

检查时其他 GPU 上有任务运行，实际任务应通过 `CUDA_VISIBLE_DEVICES` 选择分配到的 GPU。
未进行多卡训练、实际业务模型评测或长时间训练。

原环境包清单：`/tmp/loom-verl-before-cu124.txt`；安装日志：`/tmp/loom-cu124-progress.log`；
完整 GPU 0 检查成功日志：`/tmp/loom-cu124-validation-gpu0.log`；独立生成成功日志：
`/tmp/loom-cu124-vllm.log`。最初未限定可见 GPU 范围的合并检查在生成初始化时失败；
指定空闲 GPU 0 后，完整合并检查及独立生成均已成功。
