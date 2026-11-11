"""Apply the vLLM 0.32 fix for Linux tmpfs mounts with no size limit."""

from importlib.metadata import distribution


def main():
    dist = distribution("vllm")
    if dist.version != "0.32.0":
        raise RuntimeError(f"Review the shared-memory patch for vLLM {dist.version}")
    path = dist.locate_file("vllm/distributed/device_communicators/shm_broadcast.py")
    source = path.read_text()
    before = (
        "        free_bytes = shutil.disk_usage(shm_path).free\n"
        "        if required_bytes > free_bytes:\n"
    )
    after = (
        "        usage = shutil.disk_usage(shm_path)\n"
        "        free_bytes = usage.free\n"
        "        # Linux tmpfs size=0 means unlimited capacity.\n"
        "        if usage.total > 0 and required_bytes > free_bytes:\n"
    )
    if after in source:
        print("vLLM shared-memory patch already applied")
        return
    if source.count(before) != 1:
        raise RuntimeError(f"Unrecognized vLLM shared-memory check in {path}")
    path.write_text(source.replace(before, after))
    print(f"Patched {path}")


if __name__ == "__main__":
    main()
