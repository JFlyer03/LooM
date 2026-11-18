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

从 LooM 根目录执行。已有 `loom-verl` 时跳过创建步骤：

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
python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); x=torch.randn(128,128,device="cuda",dtype=torch.bfloat16); assert torch.isfinite(x@x).all()'
python -c 'import loom, verl, vllm._C, flash_attn_2_cuda; print("native imports OK")'
```

## 本次验证

版本冲突解析已通过，后端加载测试已通过；完整安装和 GPU 执行验证仍在进行。
原环境包清单备份：`/tmp/loom-verl-before-cu124.txt`。
