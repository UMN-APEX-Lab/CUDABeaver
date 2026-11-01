"""
Compile and test module.

Extracted from run_experiment.py. Handles:
  1. Copy testbench to workspace (preserving symlinks)
  2. Write solution file (solution.cu, solution.py, or custom path)
  3. Set up include/helper headers
  4. Run build command (nvcc, python static check, make, etc.)
  5. Run test command (binary, python eval, make run, etc.)

Supports multiple task types via input_meta["solution_file"]:
  - "solution.cu" (default) — standard CUDA tasks
  - "solution.py"           — kernelbench Python tasks
  - "kernels/sub/file.cu"   — thunderkittens tasks (custom path)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from .generate_headers import setup_include_dir
from .subprocess_util import run_pg
from .task_registry import copy_testbench_contents



def _detected_gpu_arch():
    """kernelbench set_gpu_arch family for the local GPU.

    Override with CUDA_DEBUGGER_GPU_ARCH (e.g. "Hopper"). Detection maps the
    first visible GPU name from nvidia-smi; returns None when undetectable
    (command left unchanged). Assumes homogeneous GPUs per node.
    """
    import subprocess as _sp
    env = os.environ.get("CUDA_DEBUGGER_GPU_ARCH")
    if env:
        return env
    global _GPU_ARCH_CACHE
    try:
        return _GPU_ARCH_CACHE
    except NameError:
        pass
    arch = None
    try:
        out = _sp.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                      capture_output=True, text=True, timeout=10).stdout
        name = out.strip().splitlines()[0].lower() if out.strip() else ""
        for pat, fam in (("h100", "Hopper"), ("h200", "Hopper"), ("gh200", "Hopper"),
                         ("blackwell", "Blackwell"), ("b100", "Blackwell"), ("b200", "Blackwell"),
                         ("b300", "Blackwell"), ("rtx 50", "Blackwell"),
                         ("l40", "Ada"), ("rtx 40", "Ada"),
                         ("a100", "Ampere"), ("a30", "Ampere"), ("a40", "Ampere")):
            if pat in name:
                arch = fam
                break
    except Exception:
        arch = None
    _GPU_ARCH_CACHE = arch
    return arch


def adapt_gpu_arch_cmd(cmd: str) -> str:
    """Rewrite a hardcoded set_gpu_arch(['<family>']) in a task command to the
    arch of the machine actually running it. The dataset was authored on
    Blackwell; without this, kernelbench tasks JIT-compile for the wrong arch
    on other GPUs and fail wholesale."""
    import re as _re
    has_arch = "set_gpu_arch" in cmd
    has_make_gpu = _re.search(r"\bmake\b.*\bGPU=\w+", cmd) is not None
    if not (has_arch or has_make_gpu):
        return cmd
    arch = _detected_gpu_arch()
    if not arch:
        return cmd
    if has_arch:
        cmd = _re.sub(r"set_gpu_arch\(\[[^\]]*\]\)", "set_gpu_arch(['" + arch + "'])", cmd)
    if has_make_gpu:
        # ThunderKittens Makefiles select the kernel API branch via GPU=<name>;
        # the corpus hardcodes the authoring machine's GPU. The wgmma-based
        # kernels of this corpus require the H100 (sm_90a) branch on Hopper.
        make_gpu = {"Hopper": "H100", "Blackwell": "RTX5070"}.get(arch)
        if make_gpu:
            cmd = _re.sub(r"\bGPU=\w+", "GPU=" + make_gpu, cmd)
    return cmd



def _fix_python_cmd(cmd: str) -> str:
    """Replace bare 'python' calls with the current interpreter path.

    Handles 'python -c', 'python script.py', etc. without affecting
    'python3' or words like 'pythonpath'.
    """
    import re
    return re.sub(r'\bpython\b(?!3)', sys.executable, cmd)


def _python_subprocess_env() -> dict:
    """Environment for task build/test commands run outside the harness package."""
    env = os.environ.copy()

    # Ensure `python` resolves to the interpreter that launched the harness.
    python_bin = Path(sys.executable).parent
    env["PATH"] = str(python_bin) + os.pathsep + env.get("PATH", "")

    # KernelBench is vendored in this repository. Build/test commands run from
    # each task workdir, so they cannot import it unless the vendor dir is on
    # PYTHONPATH.
    vendor_dir = Path(__file__).resolve().parent / "_vendor"
    pythonpath = env.get("PYTHONPATH")
    entries = [str(vendor_dir)]
    if pythonpath:
        entries.append(pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def compile_code(
    solution_code: str,
    task_stem: str,
    dataset_dir: Path,
    input_meta: dict,
    workdir: Path,
    timeout: int = 60,
) -> tuple[bool, str]:
    """
    Compile solution.cu in a temporary workspace.
    Returns (build_ok, build_output).
    """
    # Library-bypass / structural pre-check (rejects #include cutlass etc.)
    from .static_checker import validate_solution
    solution_file = input_meta.get("solution_file", "solution.cu")
    chk = validate_solution(solution_code, solution_file)
    if not chk.valid:
        msg = "Static check failed:\n" + "\n".join(f"  - {e}" for e in chk.errors)
        return False, msg

    task_workdir = workdir / task_stem
    if task_workdir.exists():
        shutil.rmtree(task_workdir)
    task_workdir.mkdir(parents=True)

    # Copy testbench files from either legacy flat datasets or benchmark/task_list.json.
    copy_testbench_contents(task_stem, dataset_dir, task_workdir)

    # Write solution to the appropriate file
    solution_file = input_meta.get("solution_file", "solution.cu")
    solution_path = task_workdir / solution_file
    solution_path.parent.mkdir(parents=True, exist_ok=True)
    solution_path.write_text(solution_code, encoding="utf-8")

    # Auto-generate headers from prompts
    setup_include_dir(task_stem, dataset_dir, task_workdir)

    # applied-kernels-aware build path: skip input.json's build_command and call applied_kernels_eval directly
    from .applied_kernels_eval import is_applied_kernels_task, applied_kernels_build
    if is_applied_kernels_task(input_meta):
        return applied_kernels_build(input_meta["task_id"], task_workdir, timeout=max(timeout, 300))

    # Default path — input.json build_command + nvcc
    build_cmd = adapt_gpu_arch_cmd(_fix_python_cmd(input_meta["build_command"]))
    build_env = _python_subprocess_env()
    try:
        result = run_pg(
            build_cmd,
            shell=True,
            cwd=str(task_workdir),
            timeout=timeout,
            env=build_env,
        )
        output = result.stdout + "\n" + result.stderr
        build_ok = result.returncode == 0
    except subprocess.TimeoutExpired:
        output = "[BUILD TIMEOUT]"
        build_ok = False

    return build_ok, output.strip()


def run_test(
    task_stem: str,
    input_meta: dict,
    workdir: Path,
    timeout: int = 30,
    performance_gate: dict | None = None,
) -> tuple[int | None, str]:
    """
    Run the compiled test. Returns (returncode, test_output).
    """
    task_workdir = workdir / task_stem
    # applied-kernels-aware test path
    from .applied_kernels_eval import is_applied_kernels_task, applied_kernels_test
    if is_applied_kernels_task(input_meta):
        return applied_kernels_test(input_meta["task_id"], task_workdir, timeout=max(timeout, 300))

    test_cmd = input_meta["test_command"]
    if _performance_gate_enabled(performance_gate):
        test_cmd = _enable_kernelbench_perf(test_cmd, input_meta)
    test_cmd = adapt_gpu_arch_cmd(_fix_python_cmd(test_cmd))
    test_env = _python_subprocess_env()
    try:
        result = run_pg(
            test_cmd,
            shell=True,
            cwd=str(task_workdir),
            timeout=timeout,
            env=test_env,
        )
        output = result.stdout + "\n" + result.stderr
        return result.returncode, output.strip()
    except subprocess.TimeoutExpired:
        return -1, "[TEST TIMEOUT] Execution timed out"


def _performance_gate_enabled(performance_gate: dict | None) -> bool:
    return bool(performance_gate and performance_gate.get("enabled", True))


def _enable_kernelbench_perf(test_cmd: str, input_meta: dict) -> str:
    """Ask KernelBench eval commands to emit runtime/ref_runtime for gating.

    The official KernelBench input JSONs in this repo usually set
    measure_performance=False because correctness-only debug was the default.
    When a YAML performance gate is enabled, the harness needs these timings
    so classifier.py can apply the configured min_speedup.
    """
    if input_meta.get("solution_file") != "solution.py":
        return test_cmd
    if "eval_kernel_against_ref" not in test_cmd:
        return test_cmd
    return test_cmd.replace("measure_performance=False", "measure_performance=True")


def run_sanitizers(
    task_stem: str,
    input_meta: dict,
    workdir: Path,
    sanitizer_config: dict | None = None,
) -> dict:
    """Run configured compute-sanitizer tools for an already-built task.

    The agent framework imports this symbol even when automatic sanitizers are
    disabled. Keep the function small and dependency-light so normal build/test
    paths are unaffected.
    """
    cfg = sanitizer_config or {}
    if not cfg.get("enabled", False):
        return {
            "triggered": False,
            "results": {},
            "hit_tools": [],
            "real_hit_tools": [],
            "timed_out_tools": [],
            "unavailable_tools": [],
            "skipped": "disabled",
        }

    from .diagnostics import run_compute_sanitizer

    tools = cfg.get("tools") or cfg.get("tool") or ["memcheck"]
    if isinstance(tools, str):
        tools = [tools]

    timeout = int(cfg.get("timeout", 180))
    options = cfg.get("options") or {}
    task_workdir = workdir / task_stem
    test_cmd = _fix_python_cmd(input_meta.get("test_command", ""))
    test_env = _python_subprocess_env()

    summary = {
        "triggered": True,
        "results": {},
        "hit_tools": [],
        "real_hit_tools": [],
        "timed_out_tools": [],
        "unavailable_tools": [],
    }

    for tool in tools:
        result = run_compute_sanitizer(
            test_cmd,
            task_workdir,
            tool=str(tool),
            timeout=timeout,
            env=test_env,
            options=options,
        )
        info = {
            "ok": result.ok,
            "summary": result.summary,
            "error": result.error,
            "output": result.to_prompt_block(),
            "meta": result.meta,
            "truncated": result.truncated,
        }
        rc = (result.meta or {}).get("returncode")
        if result.error:
            if "not found" in result.error:
                summary["unavailable_tools"].append(str(tool))
            if "timed out" in result.error:
                summary["timed_out_tools"].append(str(tool))
        if result.ok and rc not in (0, None):
            summary["hit_tools"].append(str(tool))
            if rc == 99:
                summary["real_hit_tools"].append(str(tool))
            info["primary_finding"] = {
                "tool": str(tool),
                "error_type": "compute_sanitizer_error",
                "returncode": rc,
            }
        summary["results"][str(tool)] = info

    return summary
