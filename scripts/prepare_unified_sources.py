"""Fetch pinned backend sources needed for native entrypoints and matching patches."""

import argparse
import io
import json
import tarfile
import time
import urllib.request
from pathlib import Path


SOURCES = {
    "verl": ("verl-project/verl", "b17218ba58a6fbcdfb0de32772e3c2a79f88529d"),
    "slime": ("THUDM/slime", "79aa41a884ab0bf65222c94418f6357d3c31192b"),
    "megatron": ("NVIDIA/Megatron-LM", "ae76c60e6c35d913867d6dd7bad0f30b128b54e7"),
    "nvidia-resiliency-ext": ("NVIDIA/nvidia-resiliency-ext", "ebf0352a82ca5699148ea10bec2b8eedd121e382"),
    "torch-memory-saver": ("zhuzilin/torch_memory_saver", "bdab8c55f490a00a746ec34e301391617f5ca878"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["all", "slime"], default="all")
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parents[1] / ".backend-src")
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    for name, (repo, commit) in SOURCES.items():
        if args.backend == "slime" and name == "verl":
            continue
        target = args.directory / name
        marker = target / ".loom-source.json"
        expected = {"repo": repo, "commit": commit}
        if target.exists():
            if not marker.exists() or json.loads(marker.read_text()) != expected:
                raise ValueError(f"Refusing to replace unrecognized sources in {target}")
            print(f"Using {name}@{commit}", flush=True)
            continue
        print(f"Fetching {name}@{commit}", flush=True)
        for attempt in range(3):
            try:
                with urllib.request.urlopen(f"https://codeload.github.com/{repo}/tar.gz/{commit}", timeout=60) as response:
                    data = response.read()
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(1)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            prefix = archive.getmembers()[0].name.split("/")[0]
            archive.extractall(args.directory, filter="data")
        (args.directory / prefix).rename(target)
        marker.write_text(json.dumps(expected, indent=2) + "\n")


if __name__ == "__main__":
    main()
