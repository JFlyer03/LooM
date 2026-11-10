import base64
import csv
import hashlib
import io
import importlib.util
import zipfile
from email.parser import BytesParser
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "loom_build_unified_wheels", Path(__file__).resolve().parents[1] / "scripts/build_unified_wheels.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
patch_wheel = module.patch_wheel


def make_wheel(path, requirement="engine==1.0"):
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr("example-1.0.dist-info/METADATA",
                       "Metadata-Version: 2.1\nName: example\nVersion: 1.0\n"
                       f"Requires-Dist: {requirement}\nRequires-Dist: optional==1.0\n\n")
        wheel.writestr("example-1.0.dist-info/WHEEL", "Wheel-Version: 1.0\nTag: py3-none-any\n")
        wheel.writestr("example-1.0.dist-info/RECORD", "old,invalid,record\n")
        wheel.writestr("example/native.so", b"native ABI bytes")
        wheel.writestr("example/runtime.py", b"setting = 1\n")
        wheel.writestr("example/cute/old.py", b"old shared namespace")


def test_dependency_markers_are_not_folded_into_invalid_metadata(tmp_path):
    original, output = tmp_path / "original.whl", tmp_path / "patched.whl"
    marker = 'engine==1.0; platform_machine == "x86_64" or platform_machine == "aarch64" or platform_machine == "arm64"'
    make_wheel(original, marker)
    patch_wheel(original, output, {"version": "1.0", "requires": {}})
    with zipfile.ZipFile(output) as wheel:
        metadata = BytesParser().parsebytes(wheel.read("example-1.0+loom1.dist-info/METADATA"))
        assert metadata.get_all("Requires-Dist")[0] == marker


def test_compatibility_wheel_preserves_native_bytes_and_regenerates_integrity(tmp_path):
    original, output = tmp_path / "original.whl", tmp_path / "patched.whl"
    make_wheel(original)
    original_bytes = original.read_bytes()
    config = {"version": "1.0", "requires": {"engine": ("==1.0", "==2.0"),
              "optional": ("==1.0", None)}, "remove_prefix": "example/cute/",
              "replacements": {"example/runtime.py": [["setting = 1", "setting = 2"]]}}
    removed = patch_wheel(original, output, config)
    assert original.read_bytes() == original_bytes
    assert removed == ["example/cute/old.py"]
    with zipfile.ZipFile(output) as wheel:
        assert wheel.testzip() is None
        assert wheel.read("example/native.so") == b"native ABI bytes"
        assert wheel.read("example/runtime.py") == b"setting = 2\n"
        metadata = BytesParser().parsebytes(wheel.read("example-1.0+loom1.dist-info/METADATA"))
        assert metadata["Version"] == "1.0+loom1"
        assert metadata.get_all("Requires-Dist") == ["engine==2.0"]
        records = list(csv.reader(io.StringIO(wheel.read("example-1.0+loom1.dist-info/RECORD").decode())))
        assert {row[0] for row in records} == set(wheel.namelist())
        for name, digest, size in records:
            if not digest:
                assert name.endswith(".dist-info/RECORD")
                continue
            data = wheel.read(name)
            expected = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            assert digest == "sha256=" + expected
            assert int(size) == len(data)


def test_compatibility_patch_refuses_changed_upstream_dependencies(tmp_path):
    original = tmp_path / "original.whl"
    make_wheel(original, requirement="engine==1.1")
    with pytest.raises(ValueError, match="Upstream changed"):
        patch_wheel(original, tmp_path / "patched.whl", {
            "version": "1.0", "requires": {"engine": ("==1.0", "==2.0")},
        })
