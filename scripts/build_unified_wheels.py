"""Build explicitly versioned local wheels for the H100 unified environment.

These dependency changes are integration patches, not upstream support claims.
GPU import/generation/training checks must pass before replacing an environment.
No installed package or global pip configuration is modified by this script.
"""

import argparse
import base64
import csv
import concurrent.futures
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import urllib.request
import zipfile
from email import policy
from email.parser import BytesParser
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


PATCHES = {
    "vllm": {
        "version": "0.29.0",
        "requires": {"compressed-tensors": ("==0.17.0", "==0.18.0"), "numba": ("==0.65.0", "==0.65.1")},
        "replacements": {
            "vllm/distributed/device_communicators/shm_broadcast.py": [[
                "    free_bytes = shutil.disk_usage(shm_path).free\n",
                "    usage = shutil.disk_usage(shm_path)\n"
                "    # Linux tmpfs size=0 is unlimited, not a full filesystem.\n"
                "    if usage.total == 0:\n"
                "        return\n"
                "    free_bytes = usage.free\n",
            ]],
        },
    },
    "skyrl": {
        "version": "0.4.0",
        "replacements": {
            "skyrl/backends/skyrl_train/inference_servers/vllm_router.py": [[
                "self._router_args.vllm_pd_disaggregation or self._router_args.pd_disaggregation",
                'self._router_args.vllm_pd_disaggregation or getattr(self._router_args, "pd_disaggregation", False)',
            ]],
        },
        "requires": {
            "tokenizers": (">=0.23.1", ">=0.22.2"),
            "vllm": ("==0.30.0", "==0.29.0+loom1"),
            "flashinfer-python": ("==0.6.18.post1", "==0.6.18"),
            "flashinfer-jit-cache": ("==0.6.18.post1", "==0.6.18"),
            "flashinfer-cubin": ("==0.6.18.post1", "==0.6.18"),
        },
    },
    "flash-attn": {
        "version": "2.8.3",
        "requires": {},
        "url": "https://github.com/NovaSky-AI/skyrl-wheels/releases/download/cu13torch2.13/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl",
        "remove_prefix": "flash_attn/cute/",
    },
    "sglang": {
        "version": "0.5.21",
        # Legacy outlines is imported lazily and requires an incompatible
        # outlines-core. The unified profile uses the native XGrammar backend.
        "requires": {"outlines": ("==0.1.11", None), "openai": ("==2.6.1", "==2.54.0")},
    },
}

PREBUILT = {
    "transformer-engine-torch": "https://github.com/NovaSky-AI/skyrl-wheels/releases/download/cu13torch2.13-te2.19/transformer_engine_torch-2.19.0-cp312-cp312-linux_x86_64.whl",
    "causal-conv1d": "https://github.com/NovaSky-AI/skyrl-wheels/releases/download/cu13torch2.13/causal_conv1d-1.6.2.post1-cp312-cp312-linux_x86_64.whl",
    "sglang-router": "https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-9daabcd/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl",
    "flashinfer-cubin": "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.18/flashinfer_cubin-0.6.18-py3-none-any.whl",
    "flashinfer-jit-cache": "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.18/flashinfer_jit_cache-0.6.18+cu130-cp39-abi3-manylinux_2_28_x86_64.whl",
    "apex": "https://github.com/verl-project/verl-wheelhouse/releases/download/apex-master-py3.12-torch2.13.0-cu130/apex-0.1-cp312-cp312-linux_x86_64.whl",
}

PREBUILT_HASHES = {
    "flashinfer-cubin": "2dd65c0fcfc6bc44c67f148530de5372979c2e3d260e47935730f94156d4d873",
    "flashinfer-jit-cache": "428a47a554ade93c30a818e142b781df58582bf056bab94611fb4c906cc366bf",
    "apex": "90c555507ea775e947b2d2e77dcec6ddcd8cccd0331cdb6a476691c092eee94c",
}


def download(url, path):
    for attempt in range(3):
        try:
            return download_once(url, path)
        except OSError:
            if attempt == 2:
                raise
            time.sleep(1)


def download_once(url, path):
    if path.exists():
        return
    print(f"Downloading {path.name}", flush=True)
    part = path.with_suffix(path.suffix + ".part")
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30) as response:
            size = int(response.headers.get("Content-Length", 0))
            ranges = response.headers.get("Accept-Ranges") == "bytes"
        if size > 32 * 1024 * 1024 and ranges:
            with part.open("wb") as output:
                output.truncate(size)

            def fetch_range(bounds):
                start, end = bounds
                request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
                for attempt in range(3):
                    try:
                        with urllib.request.urlopen(request, timeout=60) as response, part.open("r+b") as output:
                            if response.status != 206:
                                raise ValueError("Server did not honor the requested wheel range")
                            output.seek(start)
                            count = 0
                            while chunk := response.read(1024 * 1024):
                                count += len(chunk)
                                if count > end - start + 1:
                                    raise ValueError("Oversized wheel range")
                                output.write(chunk)
                            if count != end - start + 1:
                                raise OSError("Incomplete wheel range")
                        print(f"Downloaded range {start}-{end}", flush=True)
                        return
                    except OSError:
                        if attempt == 2:
                            raise

            # Small ranges retry independently on this server's slow long-lived
            # transfers, instead of restarting a multi-hundred-MB download.
            width = 4 * 1024 * 1024
            with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
                list(pool.map(fetch_range, [(start, min(start + width - 1, size - 1)) for start in range(0, size, width)]))
        else:
            with urllib.request.urlopen(url, timeout=60) as response, part.open("wb") as output:
                while chunk := response.read(4 * 1024 * 1024):
                    output.write(chunk)
        part.replace(path)
    finally:
        part.unlink(missing_ok=True)


def wheel_url(name, version):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=30) as response:
                files = json.load(response)["urls"]
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(1)
    candidates = [f for f in files if f["filename"].endswith(".whl") and (
        f["filename"].endswith("-py3-none-any.whl") or
        "x86_64" in f["filename"] and ("cp312" in f["filename"] or "abi3" in f["filename"])
    )]
    if len(candidates) != 1:
        raise ValueError(f"Select one Python 3.12/x86_64 wheel for {name}: {[f['filename'] for f in candidates]}")
    return candidates[0]["url"], candidates[0]["digests"]["sha256"]


def patch_wheel(source, destination, config):
    version = config["version"]
    new_version = version + "+loom1"
    seen = set()
    replaced = set()
    removed = []
    records = []
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as output:
        metadata_name = next(n for n in original.namelist() if n.endswith(".dist-info/METADATA"))
        old_dist = metadata_name.split("/")[0]
        new_dist = old_dist.replace("-" + version + ".dist-info", "-" + new_version + ".dist-info")
        for info in original.infolist():
            name = info.filename
            if name.endswith((".dist-info/RECORD", ".dist-info/RECORD.jws", ".dist-info/RECORD.p7s")):
                continue
            if config.get("remove_prefix") and name.startswith(config["remove_prefix"]):
                removed.append(name)
                continue
            data = original.read(name)
            if name in config.get("replacements", {}):
                for before, after in config["replacements"][name]:
                    if data.count(before.encode()) != 1:
                        raise ValueError(f"Upstream changed source patch context: {name}")
                    data = data.replace(before.encode(), after.encode(), 1)
                replaced.add(name)
            name = name.replace(old_dist + "/", new_dist + "/", 1)
            if info.filename == metadata_name:
                message = BytesParser(policy=policy.compat32).parsebytes(data)
                if message["Version"] != version:
                    raise ValueError("Unexpected upstream wheel version")
                message.replace_header("Version", new_version)
                requirements = message.get_all("Requires-Dist", [])
                del message["Requires-Dist"]
                for text in requirements:
                    req = Requirement(text)
                    key = canonicalize_name(req.name)
                    if key in config["requires"]:
                        before, after = config["requires"][key]
                        if str(req.specifier) != before:
                            raise ValueError(f"Upstream changed {key}: {req.specifier}, expected {before}")
                        seen.add(key)
                        if after is None:
                            continue
                        text = text.replace(before, after, 1)
                    message["Requires-Dist"] = text
                data = message.as_bytes(policy=policy.compat32.clone(max_line_length=0))
            output.writestr(name, data)
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            records.append((name, "sha256=" + digest, len(data)))
        if seen != set(config["requires"]):
            raise ValueError(f"Missing upstream requirements: {set(config['requires']) - seen}")
        if replaced != set(config.get("replacements", {})):
            raise ValueError("Missing source patch target in upstream wheel")
        record_name = new_dist + "/RECORD"
        content = io.StringIO()
        csv.writer(content, lineterminator="\n").writerows([*records, (record_name, "", "")])
        output.writestr(record_name, content.getvalue())
    return removed


def main():
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Build backend wheels with the target Python 3.12 interpreter")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["all", "slime"], default="all")
    parser.add_argument("--output-dir", type=Path, default=Path.home() / ".cache/loom-unified/wheels")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    originals = args.output_dir.parent / "originals"
    originals.mkdir(exist_ok=True)
    manifest_path = args.output_dir / "loom-patches.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest = dict(previous)
    for name, config in PATCHES.items():
        if args.backend == "slime" and name != "flash-attn":
            continue
        cached = previous.get(name, {})
        same_patch = cached.get("wheel_format") == 2 and cached.get("patch") == json.loads(json.dumps(config))
        if same_patch:
            url, expected = cached["url"], cached["sha256"]
        else:
            url, expected = (config["url"], None) if "url" in config else wheel_url(name, config["version"])
        source = originals / url.rsplit("/", 1)[1]
        download(url, source)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if expected and digest != expected:
            raise ValueError(f"Upstream wheel checksum mismatch: {name}")
        destination = args.output_dir / source.name.replace("-" + config["version"] + "-", "-" + config["version"] + "+loom1-", 1)
        if same_patch and destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() == cached.get("wheel_sha256"):
            removed = cached.get("removed", [])
        else:
            temporary = destination.with_suffix(".whl.building")
            removed = patch_wheel(source, temporary, config)
            temporary.replace(destination)
        manifest[name] = {"wheel_format": 2, "url": url, "sha256": digest, "patch": config, "removed": removed,
                          "wheel": destination.name, "wheel_sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}
        print(f"Built {destination.name}", flush=True)
    def prepare_prebuilt(item):
        name, url = item
        destination = args.output_dir / url.rsplit("/", 1)[1]
        download(url, destination)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if name in PREBUILT_HASHES and digest != PREBUILT_HASHES[name]:
            raise ValueError(f"Prebuilt wheel checksum mismatch: {name}")
        return name, {"url": url, "wheel": destination.name,
                      "wheel_sha256": digest}

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for name, info in pool.map(prepare_prebuilt, PREBUILT.items()):
            manifest[name] = info
    # Official NVRx wheels require glibc 2.39. Its checkpoint reader is Python;
    # build it without the optional CUPTI profiler on this glibc 2.35 host.
    source = Path(__file__).resolve().parents[1] / ".backend-src/nvidia-resiliency-ext"
    wheel = args.output_dir / "nvidia_resiliency_ext-0.7.0-cp312-cp312-linux_x86_64.whl"
    if not wheel.exists():
        subprocess.run([sys.executable, "-m", "uv", "build", "--python", sys.executable,
                        "--wheel", "--out-dir", str(args.output_dir), str(source)],
                       env={**os.environ, "UV_HTTP_TIMEOUT": os.environ.get("UV_HTTP_TIMEOUT", "300"),
                            "POETRY_DYNAMIC_VERSIONING_BYPASS": "0.7.0",
                            "STRAGGLER_DET_SKIP_CUPTI_EXT_BUILD": "1"}, check=True)
    manifest["nvidia-resiliency-ext"] = {
        "source": json.loads((source / ".loom-source.json").read_text()),
        "optional_cupti_profiler": False, "wheel": wheel.name,
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
    }
    if args.backend == "slime":
        source = Path(__file__).resolve().parents[1] / ".backend-src/torch-memory-saver"
        cuda = Path(sys.prefix) / "lib/python3.12/site-packages/nvidia/cu13"
        if not (cuda / "include/cuda.h").exists() or not (cuda / "lib64/libcudart.so").exists():
            raise RuntimeError("Install requirements/slime-cuda.txt and run configure_unified_runtime.py first")
        wheel = args.output_dir / "torch_memory_saver-0.0.9.post1+slime-cp312-cp312-linux_x86_64.whl"
        if not wheel.exists():
            subprocess.run([sys.executable, "-m", "uv", "build", "--python", sys.executable,
                            "--wheel", "--out-dir", str(args.output_dir), str(source)],
                           env={**os.environ, "CUDA_HOME": str(cuda), "TMS_CUDA_MAJOR": "13",
                                "UV_HTTP_TIMEOUT": os.environ.get("UV_HTTP_TIMEOUT", "300")}, check=True)
        manifest["torch-memory-saver"] = {
            "source": json.loads((source / ".loom-source.json").read_text()), "wheel": wheel.name,
            "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        }
    (args.output_dir / "loom-patches.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
