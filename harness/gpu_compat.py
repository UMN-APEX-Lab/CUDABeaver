"""
GPU architecture compatibility checker.

Detects the current GPU's compute capability and checks whether a task's
Makefile targets an architecture that can actually run on this GPU.

Key rule: `-code=sm_XXa` generates architecture-specific SASS that is NOT
forward-compatible. sm_90a code won't run on a compute 12.0 (Blackwell) GPU.
"""

import json
import logging
import re
import subprocess
from pathlib import Path

from .task_registry import resolve_task_paths

logger = logging.getLogger("cuda_debugger_exp")

# Map sm codes to minimum compute capability required to RUN the binary.
# sm_XXa (arch-specific) requires exact architecture family match.
_ARCH_SPECIFIC = {
    "sm_90a": (9, 0),   # Hopper only
    "sm_100a": (10, 0),  # Blackwell B100/B200 only
}


def get_gpu_compute_cap() -> tuple[int, int] | None:
    """Return (major, minor) compute capability of the current GPU, or None."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            # e.g. "12.0" or "9.0"
            line = out.stdout.strip().splitlines()[0].strip()
            major, minor = line.split(".")
            return int(major), int(minor)
    except Exception:
        pass
    return None


def _parse_makefile_arch(makefile_path: Path) -> list[str]:
    """Extract -code=sm_XXX targets from a Makefile."""
    if not makefile_path.exists():
        return []
    text = makefile_path.read_text()
    # Match patterns like: -gencode arch=compute_90a,code=sm_90a
    # or: -arch=sm_80
    codes = re.findall(r"code=sm_(\w+)", text)
    codes += re.findall(r"-arch=sm_(\w+)", text)
    return [f"sm_{c}" for c in codes]


def check_task_gpu_compat(
    task_stem: str,
    dataset_dir: Path,
    gpu_cap: tuple[int, int] | None,
) -> tuple[bool, str]:
    """
    Check if a task is compatible with the current GPU.

    Returns (compatible, reason).
    """
    if gpu_cap is None:
        return True, "no GPU info, assuming compatible"

    # Check Makefile in testbench. benchmark/task_list.json entries resolve to
    # data/<instance>/testbench; legacy flat datasets still work.
    try:
        testbench = resolve_task_paths(task_stem, dataset_dir).testbench_path
    except FileNotFoundError:
        return False, "task paths not found"
    makefile = testbench / "Makefile"
    arch_codes = _parse_makefile_arch(makefile)

    for code in arch_codes:
        if code in _ARCH_SPECIFIC:
            required = _ARCH_SPECIFIC[code]
            # Arch-specific codes require the SAME architecture family
            if gpu_cap[0] != required[0]:
                return False, f"{code} requires compute {required[0]}.x, GPU is {gpu_cap[0]}.{gpu_cap[1]}"

    return True, "ok"


def filter_compatible_tasks(
    task_stems: list[str],
    dataset_dir: Path,
) -> tuple[list[str], list[tuple[str, str]]]:
    """
    Filter tasks by GPU compatibility.

    Returns (compatible_stems, [(skipped_stem, reason), ...]).
    """
    gpu_cap = get_gpu_compute_cap()
    if gpu_cap:
        logger.info(f"GPU compute capability: {gpu_cap[0]}.{gpu_cap[1]}")
    else:
        logger.warning("Could not detect GPU compute capability, skipping compat check")
        return task_stems, []

    compatible = []
    skipped = []
    for stem in task_stems:
        ok, reason = check_task_gpu_compat(stem, dataset_dir, gpu_cap)
        if ok:
            compatible.append(stem)
        else:
            skipped.append((stem, reason))
            logger.warning(f"Skipping {stem}: {reason}")

    return compatible, skipped
