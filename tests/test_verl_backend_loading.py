import subprocess
import sys


def test_verl_backend_registers_without_skyrl():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, loom; "
            "from loom.integrations.base import build_backend; "
            "engine = object(); "
            "backend = build_backend('verl', infer_engine=engine); "
            "assert backend.infer_engine is engine; "
            "assert 'skyrl' not in sys.modules; "
            "assert 'loom.integrations.skyrl_train' not in sys.modules",
        ],
        check=True,
    )
