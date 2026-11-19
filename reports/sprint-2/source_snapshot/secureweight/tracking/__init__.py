"""Machine-readable provenance and exclusive-write artifact helpers."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys


def canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def object_hash(value) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_hash(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def environment_metadata():
    packages = {name: importlib.metadata.version(name) for name in
                ("secureweight", "torch", "numpy", "scikit-learn", "scipy", "matplotlib", "PyYAML")}
    return {"python": sys.version, "platform": platform.platform(),
            "machine": platform.machine(), "dependencies": packages,
            "device": "cpu", "torch_threads": 1, "deterministic_algorithms": True}


def source_metadata():
    package = Path(__file__).resolve().parents[1]
    hashes = {p.relative_to(package).as_posix(): file_hash(p) for p in sorted(package.rglob("*.py"))}
    # Resolve Git relative to the installed source, never an unrelated caller cwd.
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=package,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=package,
                                   capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"git_commit": commit, "git_dirty": dirty,
            "source_files": hashes, "source_sha256": object_hash(hashes)}
