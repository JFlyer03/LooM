"""Apply the pinned slime release's upstream patches without replacing its trainer."""

import argparse
import fnmatch
import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path


def apply_patch(root, patch, strip, *, installed_runtime=False):
    command = ["git", "apply", f"-p{strip}"]
    if installed_runtime:
        command += ["--exclude=sglang/test/*", "--exclude=sglang/**/tests/*", "--include=sglang/*"]
    env = {**os.environ, "GIT_CEILING_DIRECTORIES": str(root.parent)}
    kwargs = {"cwd": root, "env": env, "capture_output": True, "text": True}
    check = subprocess.run([*command, "--check", str(patch)], **kwargs)
    if check.returncode:
        reverse = subprocess.run([*command, "--check", "--reverse", str(patch)], **kwargs)
        if reverse.returncode:
            raise RuntimeError(f"Pinned patch no longer applies: {patch}\n{check.stderr}")
        return
    subprocess.run([*command, str(patch)], **kwargs, check=True)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=root / ".backend-src")
    args = parser.parse_args()
    patches = args.sources / "slime/docker/patch/v0.5.21"
    source = args.sources / "megatron"
    for name in ("megatron.patch", "megatron-sglang-aligned.patch"):
        patch = (patches / name).resolve()
        apply_patch(source, patch, 1)
        print(f"Applied {name}", flush=True)
    spec = importlib.util.find_spec("sglang")
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError("Install requirements/slime.txt before applying SGLang patches")
    site_packages = Path(spec.origin).parent.parent
    # Follow slime's Dockerfile: later patches depend on earlier changes.
    names = ["sglang.patch", "sglang-top_p.patch", "sglang-release_hicache.patch",
             "sglang-pull_weights.patch", "sglang-deterministic.patch",
             "sglang-radio_layerscale.patch", "sglang-nemotron-multi-image.patch"]
    ordered = [(patches / name).resolve() for name in names]
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ordered}
    marker = site_packages / "sglang/.loom-patches.json"
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous.get("patches") == manifest and all(
            (hashlib.sha256((site_packages / name).read_bytes()).hexdigest()
             if (site_packages / name).exists() else None) == digest
            for name, digest in previous.get("files", {}).items()
        ):
            print("SGLang patches already applied and verified", flush=True)
            return
    files = set()
    for patch in ordered:
        # Upstream diffs are rooted at a/python/sglang; installed files at sglang/.
        apply_patch(site_packages, patch, 2, installed_runtime=True)
        for line in patch.read_text().splitlines():
            if line.startswith(("+++ b/python/sglang/", "--- a/python/sglang/")):
                name = line.split("/", 2)[2]
                if not name.startswith("sglang/test/") and not fnmatch.fnmatch(name, "sglang/**/tests/*"):
                    files.add(name)
        print(f"Applied {patch.name}", flush=True)
    hashes = {name: hashlib.sha256((site_packages / name).read_bytes()).hexdigest()
              if (site_packages / name).exists() else None for name in files}
    marker.write_text(json.dumps({"patches": manifest, "files": hashes}, indent=2) + "\n")


if __name__ == "__main__":
    main()
