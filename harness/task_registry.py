"""Task path resolution for benchmark/task_list.json and legacy flat datasets."""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK_LIST = REPO_ROOT / "benchmark" / "task_list.json"


@dataclass(frozen=True)
class TaskPaths:
    stem: str
    task_id: str
    input_path: Path
    prompt_path: Path
    testbench_path: Path
    reference_path: Path | None
    broken_start: dict[str, Any] | None
    entry: dict[str, Any] | None = None
    flat: bool = False


def _abs_repo_path(path: str | Path, repo_root: Path = REPO_ROOT) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root / p


def is_flat_dataset(dataset_dir: Path | str) -> bool:
    root = Path(dataset_dir)
    if root.resolve() == (REPO_ROOT / "data").resolve():
        return False
    return (
        (root / "input").is_dir()
        and (root / "prompts").is_dir()
        and (root / "testbench").is_dir()
    )


@lru_cache(maxsize=16)
def load_task_entries(task_list_path: str | None = None) -> tuple[dict[str, Any], ...]:
    path = Path(task_list_path).resolve() if task_list_path else DEFAULT_TASK_LIST
    data = json.loads(path.read_text(encoding="utf-8"))
    return tuple(data.get("entries", []))


def _entry_maps(task_list_path: str | None = None) -> tuple[dict[str, dict], dict[str, dict]]:
    by_stem: dict[str, dict] = {}
    by_task_id: dict[str, dict] = {}
    for entry in load_task_entries(task_list_path):
        if entry.get("stem"):
            by_stem[str(entry["stem"])] = dict(entry)
        if entry.get("task_id"):
            by_task_id[str(entry["task_id"])] = dict(entry)
    return by_stem, by_task_id


def _task_paths_from_entry(entry: dict[str, Any]) -> TaskPaths:
    paths = entry.get("dataset_paths") or {}
    if not paths:
        # Full-id task lists omit dataset_paths: the instance lives at data/<task_id>/.
        task_id = str(entry.get("task_id") or entry.get("stem"))
        instance_dir = REPO_ROOT / "data" / task_id
        paths = {
            "input": instance_dir / "input.json",
            "prompts": instance_dir / "prompt.txt",
            "testbench": instance_dir / "testbench",
            "references": instance_dir / "reference.cu",
        }
    reference = paths.get("references")
    return TaskPaths(
        stem=str(entry.get("stem") or entry.get("task_id")),
        task_id=str(entry.get("task_id") or entry.get("stem")),
        input_path=_abs_repo_path(paths["input"]),
        prompt_path=_abs_repo_path(paths["prompts"]),
        testbench_path=_abs_repo_path(paths["testbench"]),
        reference_path=_abs_repo_path(reference) if reference else None,
        broken_start=entry.get("broken_start"),
        entry=dict(entry),
        flat=False,
    )


def _task_paths_from_instance_dir(instance_dir: Path) -> TaskPaths:
    instance = json.loads((instance_dir / "instance.json").read_text(encoding="utf-8"))
    input_meta = json.loads((instance_dir / "input.json").read_text(encoding="utf-8"))
    task_id = instance_dir.name
    reference_name = (instance.get("files") or {}).get("reference", "reference.cu")
    broken_name = (instance.get("files") or {}).get("broken_start", "broken_start.cu")
    prompt_name = (instance.get("files") or {}).get("prompt", "prompt.txt")
    input_name = (instance.get("files") or {}).get("input_meta", "input.json")
    testbench_name = (instance.get("files") or {}).get("testbench", "testbench")
    entry = {
        "task_id": task_id,
        "stem": task_id,
        "source": input_meta.get("source", {}).get("benchmark") or instance.get("task_family"),
        "dataset_paths": {
            "input": str((instance_dir / input_name).relative_to(REPO_ROOT)),
            "prompts": str((instance_dir / prompt_name).relative_to(REPO_ROOT)),
            "testbench": str((instance_dir / testbench_name).relative_to(REPO_ROOT)),
            "references": str((instance_dir / reference_name).relative_to(REPO_ROOT)),
        },
        "broken_start": {
            "code_path": str((instance_dir / broken_name).relative_to(REPO_ROOT)),
            "source_origin": instance.get("broken_start_provenance", "unknown"),
            "error_category": instance.get("error_category"),
            "lines": instance.get("broken_start_lines"),
            "code_hash": instance.get("broken_start_hash"),
        },
    }
    return _task_paths_from_entry(entry)


def _resolve_from_data_scan(task_stem: str) -> TaskPaths | None:
    data_dir = REPO_ROOT / "data"
    exact = data_dir / task_stem
    if (exact / "instance.json").exists():
        return _task_paths_from_instance_dir(exact)

    matches = sorted(data_dir.glob(f"{task_stem}__*"))
    matches = [m for m in matches if (m / "instance.json").exists()]
    if len(matches) == 1:
        return _task_paths_from_instance_dir(matches[0])
    return None


def resolve_task_paths(
    task_stem: str,
    dataset_dir: Path | str,
    task_list_path: str | Path | None = None,
) -> TaskPaths:
    dataset = Path(dataset_dir)
    input_path = dataset / "input" / f"{task_stem}.json"
    prompt_path = dataset / "prompts" / f"{task_stem}.txt"
    testbench_path = dataset / "testbench" / task_stem
    if dataset.resolve() != (REPO_ROOT / "data").resolve() and (
        input_path.exists() or is_flat_dataset(dataset)
    ):
        task_id = task_stem
        if input_path.exists():
            try:
                task_id = str(json.loads(input_path.read_text(encoding="utf-8")).get("task_id", task_stem))
            except Exception:
                task_id = task_stem
        reference_path = dataset / "references" / f"{task_stem}.cu"
        return TaskPaths(
            stem=task_stem,
            task_id=task_id,
            input_path=input_path,
            prompt_path=prompt_path,
            testbench_path=testbench_path,
            reference_path=reference_path if reference_path.exists() else None,
            broken_start=None,
            entry=None,
            flat=True,
        )

    task_list = str(Path(task_list_path).resolve()) if task_list_path else None
    by_stem, by_task_id = _entry_maps(task_list)
    entry = by_stem.get(task_stem) or by_task_id.get(task_stem)
    if entry:
        return _task_paths_from_entry(entry)

    scanned = _resolve_from_data_scan(task_stem)
    if scanned is not None:
        return scanned

    raise FileNotFoundError(
        f"task {task_stem!r} not found in flat dataset {dataset} or {DEFAULT_TASK_LIST}"
    )


def discover_task_stems(
    dataset_dir: Path | str,
    task_list_path: str | Path | None = None,
) -> list[str]:
    dataset = Path(dataset_dir)
    input_dir = dataset / "input"
    if (
        dataset.resolve() != (REPO_ROOT / "data").resolve()
        and input_dir.is_dir()
        and any(input_dir.glob("*.json"))
    ):
        return sorted(p.stem for p in input_dir.glob("*.json"))
    task_list = str(Path(task_list_path).resolve()) if task_list_path else None
    return [str(e["stem"]) for e in load_task_entries(task_list) if e.get("stem")]


def broken_start_map(task_list_path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    task_list = str(Path(task_list_path).resolve()) if task_list_path else None
    out: dict[str, dict[str, Any]] = {}
    for entry in load_task_entries(task_list):
        broken = entry.get("broken_start")
        if broken:
            out[str(entry["stem"])] = dict(broken)
            out.setdefault(str(entry.get("task_id", entry["stem"])), dict(broken))
            paths = entry.get("dataset_paths") or {}
            input_path = paths.get("input")
            if input_path:
                out.setdefault(Path(input_path).parent.name, dict(broken))
    return out


def load_input_meta(task_stem: str, dataset_dir: Path | str) -> dict[str, Any]:
    paths = resolve_task_paths(task_stem, dataset_dir)
    return json.loads(paths.input_path.read_text(encoding="utf-8"))


def load_prompt_text(task_stem: str, dataset_dir: Path | str) -> str:
    paths = resolve_task_paths(task_stem, dataset_dir)
    return paths.prompt_path.read_text(encoding="utf-8")


def task_error_category(task_stem: str, dataset_dir: Path | str) -> str | None:
    paths = resolve_task_paths(task_stem, dataset_dir)
    if paths.broken_start and paths.broken_start.get("error_category"):
        return str(paths.broken_start["error_category"])
    if paths.entry and paths.entry.get("error_category"):
        return str(paths.entry["error_category"])
    try:
        meta = json.loads(paths.input_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    tags = {str(t).replace("-", "_") for t in (meta.get("tags") or [])}
    for category in ("memory_crash", "logic_error", "compile_error", "perf_broken", "timeout"):
        if category in tags:
            return category
    return None


def _thunderkittens_roots(workdir: Path | None = None) -> list[Path]:
    roots: list[Path] = []
    if workdir is not None:
        legacy = Path.home() / "CUDA-Debugger" / "testbench"
        task_map = {
            "based": "thunderkittens_kernels_based_linear_attn.cu",
            "flux": "thunderkittens_kernels_flux_flux_gate.cu",
            "linear_attention": "thunderkittens_kernels_linear_attention_linear_attention.cu",
            "mamba2": "thunderkittens_kernels_mamba2_mamba2.cu",
        }
        for kernel_dir, legacy_name in task_map.items():
            if (workdir / "kernels" / kernel_dir).exists():
                roots.append(legacy / legacy_name)
                break
    for key in ("THUNDERKITTENS_ROOT", "TK_DIR"):
        value = os.environ.get(key)
        if value:
            roots.append(Path(value).expanduser())
    roots.extend(
        [
            REPO_ROOT / "data" / "_external" / "ThunderKittens",
            Path.home() / "KernelHell" / "reference_sources" / "ThunderKittens",
            Path.home() / "KernelHell" / "third_party" / "ThunderKittens",
        ]
    )
    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        resolved = str(root.resolve()) if root.exists() else str(root)
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(root)
    return out


def supplement_thunderkittens_headers(workdir: Path) -> None:
    include_dir = workdir / "include"
    if not (include_dir / "kittens.cuh").exists():
        return
    missing = [
        rel
        for rel in (
            Path("ops/thread/memory"),
            Path("ops/group/memory"),
        )
        if not (include_dir / rel / "memory.cuh").exists()
    ]
    if not missing:
        return
    for root in _thunderkittens_roots(workdir):
        source_include = root / "include"
        if any(not (source_include / rel / "memory.cuh").exists() for rel in missing):
            continue
        for rel in missing:
            shutil.copytree(source_include / rel, include_dir / rel, symlinks=True, dirs_exist_ok=True)
        return


def copy_testbench_contents(task_stem: str, dataset_dir: Path | str, dst: Path) -> Path:
    src = resolve_task_paths(task_stem, dataset_dir).testbench_path
    if src.exists():
        shutil.copytree(src, dst, symlinks=True, dirs_exist_ok=True)
        # Relative links (e.g. reference_sources/cutlass -> ../../../_external/cutlass)
        # only resolve inside data/; re-point the copies at their absolute targets.
        links = [Path(d) / n for d, dirs, files in os.walk(dst) for n in dirs + files
                 if (Path(d) / n).is_symlink()]
        for link in links:
            if not os.path.isabs(os.readlink(link)):
                orig = src / link.relative_to(dst)
                link.unlink()
                link.symlink_to(orig.resolve())
        supplement_thunderkittens_headers(dst)
    return src


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return os.fspath(path)
