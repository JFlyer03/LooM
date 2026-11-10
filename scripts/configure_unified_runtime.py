"""Configure this conda environment's CUDA toolkit and local backend RPCs."""

import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path


def main():
    prefix = Path(sys.prefix)
    if not (prefix / "conda-meta").is_dir():
        raise RuntimeError("Activate the target conda environment first")
    cuda = prefix / "lib/python3.12/site-packages/nvidia/cu13"
    if not (cuda / "bin/nvcc").is_file() or not (cuda / "lib/libcudart.so.13").is_file():
        raise RuntimeError("Install requirements.txt before configuring CUDA")
    # Pip wheels use lib/ and versioned names; native JIT linkers expect lib64/.
    for link, target in [(cuda / "lib64", "lib"), (cuda / "lib/libcudart.so", "libcudart.so.13")]:
        if not link.exists():
            link.symlink_to(target)
        if link.resolve() != (link.parent / target).resolve():
            raise RuntimeError(f"Unexpected existing CUDA link: {link}")
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda") or str(prefix.parents[1] / "bin/conda")
    libraries = ":".join(dict.fromkeys([str(cuda / "lib"), *filter(None, os.environ.get("LD_LIBRARY_PATH", "").split(":"))]))
    local_hosts = ["localhost", "127.0.0.1", "::1", socket.gethostname(), socket.gethostbyname(socket.gethostname())]
    no_proxy = ",".join(dict.fromkeys([*filter(None, os.environ.get("NO_PROXY", "").split(",")), *local_hosts]))
    subprocess.run([conda, "env", "config", "vars", "set", "-p", str(prefix),
                    f"CUDA_HOME={cuda}", f"LD_LIBRARY_PATH={libraries}", "NO_VCS_VERSION=1",
                    f"NO_PROXY={no_proxy}", f"no_proxy={no_proxy}"], check=True)
    print(f"CUDA_HOME={cuda}\nReactivate this environment to load its CUDA settings.")


if __name__ == "__main__":
    main()
