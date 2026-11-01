"""
Pre-flight checker for CUDA-Debugger evaluation.

Validates everything before spending API calls:
  1. GPU availability + compute capability
  2. API key present and valid
  3. Dataset exists (input/, prompts/, testbench/)
  4. Task-level validation (testbench exists, Makefile present for applied-kernels tasks)
  5. GPU architecture compatibility filtering
  6. Disk space check for workdir

Returns a PreflightReport with all issues found at once.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .gpu_compat import get_gpu_compute_cap, filter_compatible_tasks
from .task_registry import is_flat_dataset, resolve_task_paths


@dataclass
class PreflightIssue:
    level: str  # "error" (blocks run) or "warning" (proceeds with note)
    component: str
    message: str


@dataclass
class PreflightReport:
    ok: bool = True
    gpu_name: str = "unknown"
    gpu_compute_cap: tuple[int, int] | None = None
    total_tasks: int = 0
    compatible_tasks: int = 0
    skipped_tasks: list[tuple[str, str]] = field(default_factory=list)
    issues: list[PreflightIssue] = field(default_factory=list)

    def add_error(self, component: str, message: str):
        self.issues.append(PreflightIssue("error", component, message))
        self.ok = False

    def add_warning(self, component: str, message: str):
        self.issues.append(PreflightIssue("warning", component, message))

    def print_report(self):
        print("=" * 60)
        print("PRE-FLIGHT CHECK")
        print("=" * 60)
        print(f"  GPU: {self.gpu_name} (compute {self.gpu_compute_cap[0]}.{self.gpu_compute_cap[1]})" if self.gpu_compute_cap else "  GPU: not detected")
        print(f"  Tasks: {self.compatible_tasks}/{self.total_tasks} compatible")
        if self.skipped_tasks:
            print(f"  Skipped: {len(self.skipped_tasks)} (GPU arch incompatible)")

        if self.issues:
            print()
            for issue in self.issues:
                marker = "ERROR" if issue.level == "error" else "WARN "
                print(f"  [{marker}] {issue.component}: {issue.message}")

        print()
        print(f"  Result: {'PASS — ready to run' if self.ok else 'FAIL — fix errors before running'}")
        print("=" * 60)


def run_preflight(
    config: dict,
    task_stems: list[str],
    dataset_dir: Path,
) -> tuple[PreflightReport, list[str]]:
    """
    Run all pre-flight checks.

    Returns:
        (report, filtered_task_stems)
    """
    report = PreflightReport()

    # 1. GPU
    _check_gpu(report)

    # 2. API key
    _check_api_key(report, config)

    # 3. Dataset
    _check_dataset(report, dataset_dir)

    # 4. GPU compatibility filter
    filtered, skipped = filter_compatible_tasks(task_stems, dataset_dir)
    report.total_tasks = len(task_stems)
    report.compatible_tasks = len(filtered)
    report.skipped_tasks = skipped
    if not filtered:
        report.add_error("tasks", "No compatible tasks after GPU filtering")

    # 5. Task-level validation (spot check)
    _check_tasks(report, filtered[:10], dataset_dir)  # check first 10

    # 6. Performance reference cache
    _check_reference_cache(report, config, filtered, dataset_dir)

    # 7. Disk space
    _check_disk(report, config)

    return report, filtered


def _check_gpu(report: PreflightReport):
    cap = get_gpu_compute_cap()
    if cap is None:
        report.add_warning("gpu", "Could not detect GPU — tasks requiring specific arch may fail")
        return
    report.gpu_compute_cap = cap
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            report.gpu_name = out.stdout.strip().splitlines()[0].strip()
    except Exception:
        pass


def _check_api_key(report: PreflightReport, config: dict):
    from .llm_client import detect_backend

    backend = detect_backend(config)
    key_env = config.get("api", {}).get("api_key_env", "OPENAI_API_KEY")
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get(key_env)

    if backend == "vllm":
        # vLLM doesn't need a real API key
        if not key:
            report.add_warning("api_key", "No API key set (OK for vLLM, will use placeholder)")
        # Check vLLM endpoint is reachable
        base_url = config.get("api", {}).get("base_url", "")
        host = config.get("api", {}).get("host", "localhost")
        port = config.get("api", {}).get("port", 8000)
        if not base_url:
            base_url = f"http://{host}:{port}/v1"
        try:
            import urllib.request
            req = urllib.request.Request(f"{base_url}/models", method="GET")
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            report.add_warning("vllm", f"Cannot reach vLLM at {base_url}: {e}")
    else:
        if not key:
            report.add_error("api_key", f"Neither OPENAI_API_KEY nor {key_env} is set")
        elif len(key) < 20:
            report.add_warning("api_key", "API key looks suspiciously short")


def _check_dataset(report: PreflightReport, dataset_dir: Path):
    if not is_flat_dataset(dataset_dir):
        task_list = Path(__file__).resolve().parents[1] / "benchmark" / "task_list.json"
        if task_list.exists():
            report.add_warning(
                "dataset",
                f"{dataset_dir} is not a flat dataset; using {task_list}",
            )
            return
    for subdir in ("input", "prompts", "testbench"):
        p = dataset_dir / subdir
        if not p.exists():
            report.add_error("dataset", f"Missing directory: {p}")
        elif not any(p.iterdir()):
            report.add_error("dataset", f"Empty directory: {p}")


def _check_tasks(report: PreflightReport, task_stems: list[str], dataset_dir: Path):
    for stem in task_stems:
        try:
            paths = resolve_task_paths(stem, dataset_dir)
        except FileNotFoundError as exc:
            report.add_error("task", str(exc))
            continue

        input_file = paths.input_path
        if not input_file.exists():
            report.add_error("task", f"Missing input: {input_file}")
            continue
        try:
            meta = json.loads(input_file.read_text())
            if "build_command" not in meta:
                report.add_warning("task", f"{stem}: missing build_command in input JSON")
        except json.JSONDecodeError:
            report.add_error("task", f"{stem}: invalid JSON in input file")

        prompt_file = paths.prompt_path
        if not prompt_file.exists():
            report.add_error("task", f"Missing prompt: {prompt_file}")

        testbench = paths.testbench_path
        if not testbench.exists():
            report.add_error("task", f"Missing testbench: {testbench}")


def _performance_gate_requires_cache(config: dict) -> tuple[bool, bool]:
    raw = config.get("performance_gate")
    if raw is None:
        raw = config.get("experiment", {}).get("performance_gate")
    if not isinstance(raw, dict) or not raw.get("enabled", True):
        return False, False
    measure_on_pass = bool(config.get("experiment", {}).get("measure_perf_on_pass", True))
    return measure_on_pass, bool(raw.get("fail_on_skipped", False))


def _check_reference_cache(
    report: PreflightReport,
    config: dict,
    task_stems: list[str],
    dataset_dir: Path,
) -> None:
    required, hard_fail = _performance_gate_requires_cache(config)
    if not required:
        return

    try:
        from .benchmark import cache_lookup, load_reference
    except Exception as exc:
        report.add_error("reference_cache", f"Could not import benchmark cache helpers: {exc}")
        return

    missing: list[str] = []
    errors: list[str] = []
    skipped: list[str] = []
    for stem in task_stems:
        try:
            ref = load_reference(stem, dataset_root=dataset_dir)
            cached = cache_lookup(ref, dataset_root=dataset_dir)
        except Exception as exc:
            errors.append(f"{stem}: {type(exc).__name__}: {str(exc)[:160]}")
            continue
        if cached is None:
            missing.append(stem)
        elif cached.skipped:
            skipped.append(f"{stem}:{cached.skipped}")

    if not missing and not errors and not skipped:
        return

    details = []
    if missing:
        details.append(
            f"missing {len(missing)} cache entries ({', '.join(missing[:12])}"
            f"{'...' if len(missing) > 12 else ''})"
        )
    if skipped:
        details.append(
            f"skipped {len(skipped)} references ({', '.join(skipped[:6])}"
            f"{'...' if len(skipped) > 6 else ''})"
        )
    if errors:
        details.append(
            f"errors {len(errors)} ({'; '.join(errors[:4])}"
            f"{'...' if len(errors) > 4 else ''})"
        )
    message = (
        "; ".join(details)
        + ". Precompute references first, e.g. "
        "`python3 scripts/measure_logic_baselines.py --dataset-root data "
        "--task-list benchmark/task_list.json --error-category memory_crash --gpu <id>`."
    )
    if hard_fail:
        report.add_error("reference_cache", message)
    else:
        report.add_warning("reference_cache", message)


def _check_disk(report: PreflightReport, config: dict):
    workdir = config.get("build", {}).get("workdir", "/tmp/cuda_debugger_workspace")
    parent = Path(workdir).parent
    try:
        usage = shutil.disk_usage(str(parent))
        free_gb = usage.free / (1024 ** 3)
        if free_gb < 1.0:
            report.add_error("disk", f"Only {free_gb:.1f}GB free in {parent} — need at least 1GB")
        elif free_gb < 5.0:
            report.add_warning("disk", f"Low disk: {free_gb:.1f}GB free in {parent}")
    except Exception:
        report.add_warning("disk", f"Could not check disk space for {parent}")
