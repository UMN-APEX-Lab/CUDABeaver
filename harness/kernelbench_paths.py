"""KernelBench import-path helpers.

Benchmark task JSONs import the upstream package as top-level
``kernelbench``. In this repo we may have either a user checkout under
``~/KernelBench/src`` or a vendored fallback under ``harness/_vendor``.
Subprocess build/test commands need one of those parent directories on
``PYTHONPATH``.
"""

from __future__ import annotations

import os
from pathlib import Path


def _package_parent(path: Path) -> Path | None:
    if (path / "kernelbench" / "__init__.py").exists():
        return path
    if (path / "src" / "kernelbench" / "__init__.py").exists():
        return path / "src"
    return None


def kernelbench_pythonpath_entries() -> list[str]:
    candidates: list[Path] = []
    for key in ("KERNELBENCH_PYTHONPATH", "KERNELBENCH_SRC", "KERNELBENCH_HOME"):
        value = os.environ.get(key)
        if value:
            candidates.append(Path(value).expanduser())

    candidates.extend([
        Path.home() / "KernelBench" / "src",
        Path.home() / "KernelBench",
        Path(__file__).resolve().parent / "_vendor",
    ])

    entries: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        parent = _package_parent(candidate)
        if parent is None:
            continue
        text = str(parent.resolve())
        if text in seen:
            continue
        seen.add(text)
        entries.append(text)
    return entries


def extend_pythonpath_for_kernelbench(env: dict[str, str]) -> None:
    entries = kernelbench_pythonpath_entries()
    if not entries:
        return
    existing = [x for x in env.get("PYTHONPATH", "").split(os.pathsep) if x]
    merged: list[str] = []
    seen: set[str] = set()
    for item in entries + existing:
        if item in seen:
            continue
        seen.add(item)
        merged.append(item)
    env["PYTHONPATH"] = os.pathsep.join(merged)
