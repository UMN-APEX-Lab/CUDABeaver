"""Performance benchmark pipeline.

Loads references from benchmark/references/{task_id}.json, runs them and
candidates, caches reference timings per (gpu, cuda_version) signature.
"""
from __future__ import annotations
import json
import os
import re
import signal
import subprocess
import sys
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Optional

import shutil
import tempfile

from .references.schema import Reference, ReferenceSolution, TimingMode
from .kernelbench_paths import extend_pythonpath_for_kernelbench
from .task_registry import resolve_task_paths, supplement_thunderkittens_headers

# Assets root: the repo itself by default (self-contained release); override
# with CUDA_DEBUGGER_ASSETS_ROOT to relocate references/cache/testbench data.
ROOT = Path(os.environ["CUDA_DEBUGGER_ASSETS_ROOT"]) if os.environ.get("CUDA_DEBUGGER_ASSETS_ROOT") else Path(__file__).resolve().parents[1]
REFS_DIR = ROOT / "benchmark/references"
CACHE_DIR = ROOT / "benchmark/references_cache"


def _dataset_root(dataset_root: Path | str | None = None) -> Path:
    return Path(dataset_root).resolve() if dataset_root is not None else ROOT


def _legacy_eval_dir(dataset_root: Path | str | None) -> Path | None:
    """Flat/variant datasets may ship their own evaluation/references tree."""
    if dataset_root is None:
        return None
    root = _dataset_root(dataset_root)
    if root == (ROOT / "data").resolve():
        return None  # the release corpus always uses benchmark/references
    d = root / "evaluation"
    return d if (d / "references").is_dir() else None


def _refs_dir(dataset_root: Path | str | None = None) -> Path:
    legacy = _legacy_eval_dir(dataset_root)
    return legacy / "references" if legacy else REFS_DIR


def _cache_dir(dataset_root: Path | str | None = None) -> Path:
    legacy = _legacy_eval_dir(dataset_root)
    return legacy / "references_cache" if legacy else CACHE_DIR


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


@lru_cache(maxsize=1)
def gpu_signature() -> str:
    """Return a stable string identifying (GPU model, sm, cuda toolkit)."""
    name = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits", "-i", "0"],
        text=True,
    ).strip()
    cap = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits", "-i", "0"],
        text=True,
    ).strip()
    sm = "sm_" + cap.replace(".", "")
    nvcc_out = subprocess.check_output(["nvcc", "--version"], text=True)
    cuda_match = re.search(r"release (\d+)\.(\d+)", nvcc_out)
    cuda = (
        f"cuda_{cuda_match.group(1)}_{cuda_match.group(2)}"
        if cuda_match
        else "cuda_unknown"
    )
    return f"{_slug(name)}_{sm}_{cuda}"


@dataclass
class CacheEntry:
    task_id: str
    reference_hash: str
    gpu_signature: str
    cuda_version: str
    timing: dict
    measured_at: str
    elapsed_total_s: float
    skipped: Optional[str] = None  # e.g. "incompatible_arch"
    timing_fingerprint: Optional[str] = None


def task_id_to_stem(task_id: str) -> str:
    return task_id.replace("/", "_")


def _kernelbench_reference_command(correct_trials: int, perf_trials: int) -> str:
    return (
        "python -c \""
        "from pathlib import Path; "
        "from kernelbench.eval import eval_kernel_against_ref, get_torch_dtype_from_string; "
        "from kernelbench.utils import set_gpu_arch; "
        "set_gpu_arch(['Blackwell']); "
        "problem = Path('problem.py').read_text(encoding='utf-8'); "
        "solution = Path('solution.py').read_text(encoding='utf-8'); "
        "result = eval_kernel_against_ref("
        "original_model_src=problem, "
        "custom_model_src=solution, "
        "verbose=False, "
        "measure_performance=True, "
        "timing_method='cuda_event', "
        f"num_correct_trials={correct_trials}, "
        f"num_perf_trials={perf_trials}, "
        "backend='cuda', "
        "precision=get_torch_dtype_from_string('fp32'), "
        "check_for_excessive_speedup=False"
        "); "
        "print(result); "
        "raise SystemExit(0 if (result and result.compiled and result.correctness and result.runtime_stats) else 1)"
        "\""
    )


def _applied_kernels_benchmark_command(correct_trials: int = 3, perf_trials: int = 10) -> str:
    return (
        "python -c \""
        "import os; "
        "from harness.applied_kernels_runner.eval import eval_kernel; "
        "cached=float(os.environ.get('CUDABEAVER_REFERENCE_MEAN_MS') or 0.0); "
        "result=eval_kernel("
        "task_dir='.', "
        "solution_cu='solution.cu', "
        "measure_performance=True, "
        f"num_correct_trials={correct_trials}, "
        f"num_perf_trials={perf_trials}, "
        "atol=5e-2, "
        "rtol=5e-2, "
        "verbose=False, "
        "cached_ref_time_ms=cached"
        "); "
        "print('BUILD_OK=%s' % bool(result.compiled)); "
        "import sys; "
        "compiled=bool(result.compiled); "
        "correct=bool(result.correctness); "
        "err=(result.compilation_error or result.runtime_error or ''); "
        "lat=(result.solution_time.mean_ms if result.solution_time else 0.0); "
        "ref=(result.reference_time.mean_ms if result.reference_time else 0.0); "
        "sp=(result.speedup or ((ref/lat) if ref > 0.0 and lat > 0.0 else 0.0)); "
        "print('Kernel time: %.9f ms' % lat); "
        "print('Ref time: %.9f ms' % ref); "
        "print('Speedup: %.9fx' % sp); "
        "print(err if err else ''); "
        "raise SystemExit(0 if compiled and correct and lat > 0.0 else (2 if not compiled else 3))"
        "\""
    )


def _cutlass_benchmark_command() -> str:
    return (
        "if [ -f solution.h ]; then "
        "make test CUTLASS_DIR=./reference_sources/cutlass TK_DIR=./reference_sources/ThunderKittens "
        "&& ./test; "
        "else "
        "make ref CUTLASS_DIR=./reference_sources/cutlass TK_DIR=./reference_sources/ThunderKittens "
        "&& ./ref; "
        "fi"
    )


def _thunderkittens_benchmark_command(paths, solution_file: str, meta: dict) -> str:
    import shlex

    kernel_dir = Path(solution_file).parent.as_posix()
    return f"cd {shlex.quote(kernel_dir)} && make GPU=H200 clean && make GPU=H200 run"


def _cuda_benchmark_build_command(build_command: str, workdir: Path) -> str | None:
    """Derive a benchmark binary build command from the task's test build command."""
    if not (workdir / "benchmark" / "benchmark_main.cu").exists():
        return None
    cmd = re.sub(r"-o\s+\S+", "-o benchmark.out", build_command, count=1)
    if cmd == build_command:
        return None
    if "test/*.cu" in cmd:
        cmd = cmd.replace("test/*.cu", "benchmark/*.cu")
    elif "benchmark/*.cu" not in cmd:
        cmd = f"{cmd} benchmark/*.cu"
    if (workdir / "benchmark" / "benchmark_kernel.cu").exists():
        cmd = re.sub(r"\s+\S*src/kernel\.cu\b", "", cmd)
    return cmd


def _reference_from_task_registry(task_id: str, dataset_root: Path | str | None = None) -> Reference:
    root = _dataset_root(dataset_root)
    paths = resolve_task_paths(task_id, root)
    meta = json.loads(paths.input_path.read_text(encoding="utf-8"))
    level = str(meta.get("level", ""))
    solution_file = meta.get("solution_file", "solution.cu")

    if solution_file == "solution.py" or level.startswith("kernelbench"):
        problem_path = paths.testbench_path / "problem.py"
        correct_trials = 2 if level == "kernelbench_level2" else 5
        perf_trials = 100
        return Reference(
            task_id=task_id,
            reference_solution=ReferenceSolution(
                type="kernelbench_problem",
                problem_path=str(problem_path),
                problem_content=problem_path.read_text(encoding="utf-8"),
            ),
            benchmark_command=_kernelbench_reference_command(correct_trials, perf_trials),
            timing_mode=TimingMode(type="kernelbench_eval_result"),
            n_trials=perf_trials,
            warmup_trials=0,
            requires_arch=None,
        )

    if level == "applied_kernels_class":
        if paths.reference_path is None or not paths.reference_path.exists():
            raise FileNotFoundError(f"missing reference source for {task_id}: {paths.reference_path}")
        return Reference(
            task_id=task_id,
            reference_solution=ReferenceSolution(
                type="file",
                files=[
                    {
                        "path": solution_file,
                        "content": paths.reference_path.read_text(encoding="utf-8"),
                    }
                ],
            ),
            benchmark_command=_applied_kernels_benchmark_command(),
            timing_mode=TimingMode(type="applied_kernels_make_speedup"),
            n_trials=10,
            warmup_trials=0,
            requires_arch=None,
        )

    if level == "cutlass_class":
        if paths.reference_path is None or not paths.reference_path.exists():
            raise FileNotFoundError(f"missing reference source for {task_id}: {paths.reference_path}")
        return Reference(
            task_id=task_id,
            reference_solution=ReferenceSolution(
                type="file",
                files=[
                    {
                        "path": ".cudabeaver_reference_manifest",
                        "content": paths.reference_path.read_text(encoding="utf-8"),
                    }
                ],
            ),
            benchmark_command=_cutlass_benchmark_command(),
            timing_mode=TimingMode(type="applied_kernels_make_speedup"),
            n_trials=1,
            warmup_trials=0,
            requires_arch=None,
        )

    if level == "thunderkittens":
        if paths.reference_path is None or not paths.reference_path.exists():
            raise FileNotFoundError(f"missing reference source for {task_id}: {paths.reference_path}")
        return Reference(
            task_id=task_id,
            reference_solution=ReferenceSolution(
                type="file",
                files=[
                    {
                        "path": solution_file,
                        "content": paths.reference_path.read_text(encoding="utf-8"),
                    }
                ],
            ),
            benchmark_command=_thunderkittens_benchmark_command(paths, solution_file, meta),
            timing_mode=TimingMode(type="thunderkittens_stdout"),
            n_trials=1,
            warmup_trials=0,
            requires_arch="sm_90a",
        )

    if paths.reference_path is None or not paths.reference_path.exists():
        raise FileNotFoundError(f"missing reference source for {task_id}: {paths.reference_path}")

    source = meta.get("source") or {}
    tags = set(meta.get("tags") or [])
    is_cudabench = (
        level.startswith("cudabench")
        or source.get("benchmark") == "CUDABench"
        or "cudabench" in tags
    )
    has_benchmark_main = (paths.testbench_path / "benchmark" / "benchmark_main.cu").exists()
    if is_cudabench:
        benchmark_command = "python benchmark/generate_inputs_only.py && ./test.out"
    elif has_benchmark_main:
        benchmark_command = "./benchmark.out"
    else:
        benchmark_command = "./test.out --perf"
    return Reference(
        task_id=task_id,
        reference_solution=ReferenceSolution(
            type="file",
            files=[
                {
                    "path": solution_file,
                    "content": paths.reference_path.read_text(encoding="utf-8"),
                }
            ],
        ),
        benchmark_command=benchmark_command,
        timing_mode=TimingMode(type="cuda_event_stdout"),
        n_trials=100,
        warmup_trials=3,
        requires_arch=None,
    )


def load_reference(task_id: str, dataset_root: Path | str | None = None) -> Reference:
    stem = task_id_to_stem(task_id)
    path = _refs_dir(dataset_root) / f"{stem}.json"
    if path.exists():
        return Reference.from_dict(json.load(open(path)))
    return _reference_from_task_registry(task_id, dataset_root)


def _cache_path(task_id: str, dataset_root: Path | str | None = None) -> Path:
    sig = gpu_signature()
    return _cache_dir(dataset_root) / sig / f"{task_id_to_stem(task_id)}.json"


def cache_lookup(ref: Reference, dataset_root: Path | str | None = None) -> Optional[CacheEntry]:
    p = _cache_path(ref.task_id, dataset_root)
    if not p.exists():
        return None
    d = json.load(open(p))
    if d.get("reference_hash") != ref.canonical_hash():
        return None
    cached_timing_fingerprint = d.get("timing_fingerprint")
    if (
        cached_timing_fingerprint is not None
        and cached_timing_fingerprint != ref.timing_fingerprint()
    ):
        return None
    if cached_timing_fingerprint is None and ref.timing_mode.type == "cuda_event_stdout":
        return None
    if d.get("gpu_signature") != gpu_signature():
        return None
    return CacheEntry(**d)


def cache_write(entry: CacheEntry, dataset_root: Path | str | None = None) -> None:
    p = _cache_path(entry.task_id, dataset_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(entry), indent=2) + "\n")


def prepare_workdir(
    task_id: str,
    solution: ReferenceSolution,
    dataset_root: Path | str | None = None,
) -> Path:
    """Create a fresh workdir: copy testbench, materialize solution files.

    Returns the workdir path. This module never deletes — caller is responsible
    for cleanup if explicitly authorized.
    """
    stem = task_id_to_stem(task_id)
    root = _dataset_root(dataset_root)
    try:
        testbench_src = resolve_task_paths(task_id, root).testbench_path
    except FileNotFoundError:
        testbench_src = root / "testbench" / stem
    workdir = Path(tempfile.mkdtemp(prefix=f"benchmark_{stem}_"))

    if testbench_src.is_dir():
        shutil.copytree(testbench_src, workdir, dirs_exist_ok=True, symlinks=True)
        supplement_thunderkittens_headers(workdir)

    if solution.type == "file":
        for f in solution.files:
            tgt = workdir / f["path"]
            tgt.parent.mkdir(parents=True, exist_ok=True)
            tgt.write_text(f["content"])
    elif solution.type == "kernelbench_problem":
        # Prefer embedded problem_content (self-contained); fall back to file
        if solution.problem_content is not None:
            problem_text = solution.problem_content
        else:
            problem_text = (root / solution.problem_path).read_text()
        (workdir / "problem.py").write_text(problem_text)
        # For reference timing, solution.py mirrors problem with a ModelNew alias
        # so kernelbench.eval_kernel_against_ref accepts it and runtime ≈ ref_runtime
        solution_alias = problem_text + "\n\nModelNew = Model\n"
        (workdir / "solution.py").write_text(solution_alias)
    return workdir


import datetime  # noqa: E402
import time  # noqa: E402
from typing import Tuple  # noqa: E402

# Interpreter used for python-based build/bench commands. Overridable so the
# packaged harness runs outside the original environment.
def _python_bin() -> str:
    return os.environ.get("CUDA_DEBUGGER_PYTHON") or os.environ.get("PYTHON_BIN") or sys.executable


def _build_env(gpu_id: int) -> dict:
    env = os.environ.copy()
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    pybin = Path(_python_bin()).parent
    env["PATH"] = f"{pybin}:{env.get('PATH','')}"
    existing_pythonpath = [x for x in env.get("PYTHONPATH", "").split(os.pathsep) if x]
    root_text = str(ROOT)
    if root_text not in existing_pythonpath:
        env["PYTHONPATH"] = os.pathsep.join([root_text] + existing_pythonpath)
    extend_pythonpath_for_kernelbench(env)
    return env


_CUDA_EVENT_HELPER = r'''

#ifndef CUDABEAVER_CUDA_EVENT_TIMING_HELPER
#define CUDABEAVER_CUDA_EVENT_TIMING_HELPER
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
static void __cudabeaver_cuda_check(cudaError_t status, const char* expr, const char* file, int line) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d for %s: %s\n", file, line, expr, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}
#define CUDABEAVER_CUDA_CHECK(call) __cudabeaver_cuda_check((call), #call, __FILE__, __LINE__)
static void __cudabeaver_print_cuda_event_timing(float total_ms, double n_iters) {
    if (n_iters <= 0.0) n_iters = 1.0;
    std::printf("CUDA_EVENT_TOTAL_MS=%.9f\n", static_cast<double>(total_ms));
    std::printf("CUDA_EVENT_N_ITERS=%.0f\n", n_iters);
    std::printf("CUDA_EVENT_MEAN_MS=%.9f\n", static_cast<double>(total_ms) / n_iters);
}
#endif
'''


def _add_cuda_event_helper(src: str) -> str:
    if "CUDABEAVER_CUDA_EVENT_TIMING_HELPER" in src:
        return src
    include_matches = list(re.finditer(r"^\s*#\s*include\b.*$", src, flags=re.MULTILINE))
    if include_matches:
        idx = include_matches[-1].end()
        return src[:idx] + _CUDA_EVENT_HELPER + src[idx:]
    return _CUDA_EVENT_HELPER + "\n" + src


def _strip_nvtx_include_if_unused(src: str) -> str:
    if "nvtxRange" in src:
        return src
    return re.sub(r'^\s*#\s*include\s*[<"]nvtx3/nvToolsExt\.h[>"]\s*\n', "", src, flags=re.MULTILINE)


def _guess_timed_iters_expr(region_body: str) -> str:
    m = re.search(
        r"\bfor\s*\([^;]*;\s*[^;]*<\s*([A-Za-z_][A-Za-z0-9_]*|[0-9]+)\s*;",
        region_body,
    )
    return m.group(1) if m else "1"


def _scope_stack_before(src: str, offset: int) -> list[int]:
    stack: list[int] = []
    for i, ch in enumerate(src[:offset]):
        if ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            stack.pop()
    return stack


def _guess_cuda_stream_expr(src: str, push_offset: int) -> str:
    for scope_start in reversed(_scope_stack_before(src, push_offset)):
        scoped_prefix = src[scope_start + 1:push_offset]
        declarations = re.findall(
            r"\bcudaStream_t\s+([A-Za-z_][A-Za-z0-9_]*)\b(?!\s*\[)",
            scoped_prefix,
        )
        if declarations:
            return declarations[-1]
    return "0"


def _sync_before_event_if_default_stream(stream: str, indent: str) -> str:
    if stream.strip() != "0":
        return ""
    return f"{indent}CUDABEAVER_CUDA_CHECK(cudaDeviceSynchronize());\n"


def _rewrite_nvtx_regions_to_cuda_events(src: str) -> tuple[str, int]:
    pattern = re.compile(
        r"(?P<indent>[ \t]*)nvtxRangePushA\s*\(\s*\"bench_region\"\s*\)\s*;\s*"
        r"(?P<body>.*?)"
        r"(?P=indent)nvtxRangePop\s*\(\s*\)\s*;",
        flags=re.DOTALL,
    )
    out_parts: list[str] = []
    last = 0
    replacements = 0
    for match in pattern.finditer(src):
        body = match.group("body")
        indent = match.group("indent")
        n_iters = _guess_timed_iters_expr(body)
        stream = _guess_cuda_stream_expr(src, match.start())
        pre_sync = _sync_before_event_if_default_stream(stream, indent)
        post_sync = _sync_before_event_if_default_stream(stream, indent)
        replacement = (
            f"{indent}cudaEvent_t __cudabeaver_start, __cudabeaver_stop;\n"
            f"{indent}CUDABEAVER_CUDA_CHECK(cudaEventCreate(&__cudabeaver_start));\n"
            f"{indent}CUDABEAVER_CUDA_CHECK(cudaEventCreate(&__cudabeaver_stop));\n"
            f"{pre_sync}"
            f"{indent}CUDABEAVER_CUDA_CHECK(cudaEventRecord(__cudabeaver_start, {stream}));\n"
            f"{body}"
            f"{post_sync}"
            f"{indent}CUDABEAVER_CUDA_CHECK(cudaEventRecord(__cudabeaver_stop, {stream}));\n"
            f"{indent}CUDABEAVER_CUDA_CHECK(cudaEventSynchronize(__cudabeaver_stop));\n"
            f"{indent}float __cudabeaver_total_ms = 0.0f;\n"
            f"{indent}CUDABEAVER_CUDA_CHECK(cudaEventElapsedTime(&__cudabeaver_total_ms, __cudabeaver_start, __cudabeaver_stop));\n"
            f"{indent}CUDABEAVER_CUDA_CHECK(cudaEventDestroy(__cudabeaver_start));\n"
            f"{indent}CUDABEAVER_CUDA_CHECK(cudaEventDestroy(__cudabeaver_stop));\n"
            f"{indent}__cudabeaver_print_cuda_event_timing(__cudabeaver_total_ms, static_cast<double>({n_iters}));"
        )
        out_parts.append(src[last:match.start()])
        out_parts.append(replacement)
        last = match.end()
        replacements += 1
    if replacements == 0:
        return src, 0
    out_parts.append(src[last:])
    rewritten = _strip_nvtx_include_if_unused(_add_cuda_event_helper("".join(out_parts)))
    return rewritten, replacements


def _find_matching_brace(src: str, open_brace: int) -> int:
    depth = 0
    i = open_brace
    while i < len(src):
        ch = src[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _rewrite_timed_loop_to_cuda_events(src: str) -> tuple[str, int]:
    if "CUDA_EVENT_MEAN_MS" in src:
        return src, 0
    loop_re = re.compile(
        r"\bfor\s*\([^;]*;\s*[^;]*<\s*"
        r"(?P<iters>TIMED_ITERS|timed_iters|TIMED|timed|ITERS)"
        r"\s*;[^)]*\)\s*\{",
        flags=re.MULTILINE,
    )
    matches = list(loop_re.finditer(src))
    if not matches:
        return src, 0

    match = matches[-1]
    open_brace = src.find("{", match.start(), match.end())
    close_brace = _find_matching_brace(src, open_brace)
    if open_brace < 0 or close_brace < 0:
        return src, 0

    line_start = src.rfind("\n", 0, match.start()) + 1
    indent = src[line_start:match.start()]
    if not re.fullmatch(r"[ \t]*", indent):
        indent = ""
    stream = _guess_cuda_stream_expr(src, match.start())
    n_iters = match.group("iters")
    loop_block = src[match.start():close_brace + 1]
    pre_sync = _sync_before_event_if_default_stream(stream, indent)
    post_sync = _sync_before_event_if_default_stream(stream, indent)
    replacement = (
        f"{indent}cudaEvent_t __cudabeaver_start, __cudabeaver_stop;\n"
        f"{indent}CUDABEAVER_CUDA_CHECK(cudaEventCreate(&__cudabeaver_start));\n"
        f"{indent}CUDABEAVER_CUDA_CHECK(cudaEventCreate(&__cudabeaver_stop));\n"
        f"{pre_sync}"
        f"{indent}CUDABEAVER_CUDA_CHECK(cudaEventRecord(__cudabeaver_start, {stream}));\n"
        f"{loop_block}\n"
        f"{post_sync}"
        f"{indent}CUDABEAVER_CUDA_CHECK(cudaEventRecord(__cudabeaver_stop, {stream}));\n"
        f"{indent}CUDABEAVER_CUDA_CHECK(cudaEventSynchronize(__cudabeaver_stop));\n"
        f"{indent}float __cudabeaver_total_ms = 0.0f;\n"
        f"{indent}CUDABEAVER_CUDA_CHECK(cudaEventElapsedTime(&__cudabeaver_total_ms, __cudabeaver_start, __cudabeaver_stop));\n"
        f"{indent}CUDABEAVER_CUDA_CHECK(cudaEventDestroy(__cudabeaver_start));\n"
        f"{indent}CUDABEAVER_CUDA_CHECK(cudaEventDestroy(__cudabeaver_stop));\n"
        f"{indent}__cudabeaver_print_cuda_event_timing(__cudabeaver_total_ms, static_cast<double>({n_iters}));"
    )
    rewritten = src[:match.start()] + replacement + src[close_brace + 1:]
    return _add_cuda_event_helper(rewritten), 1


def _wrap_solution_main_with_cuda_event(src: str) -> tuple[str, bool]:
    if "CUDA_EVENT_MEAN_MS" in src or "__cudabeaver_original_main" in src:
        return src, False
    match = re.search(
        r"\bint\s+main\s*\((?P<params>[^)]*)\)",
        src,
        flags=re.DOTALL,
    )
    if not match:
        return src, False

    params = " ".join(match.group("params").split())
    has_args = bool(params and params != "void")
    renamed = src[:match.start()] + "int __cudabeaver_original_main(" + match.group("params") + ")" + src[match.end():]
    renamed = _add_cuda_event_helper(renamed)
    wrapper_params = "int argc, char** argv" if has_args else ""
    call_args = "argc, argv" if has_args else ""
    wrapper = f'''

int main({wrapper_params}) {{
    cudaEvent_t __cudabeaver_start, __cudabeaver_stop;
    CUDABEAVER_CUDA_CHECK(cudaEventCreate(&__cudabeaver_start));
    CUDABEAVER_CUDA_CHECK(cudaEventCreate(&__cudabeaver_stop));
    CUDABEAVER_CUDA_CHECK(cudaEventRecord(__cudabeaver_start, 0));
    int __cudabeaver_rc = __cudabeaver_original_main({call_args});
    CUDABEAVER_CUDA_CHECK(cudaEventRecord(__cudabeaver_stop, 0));
    CUDABEAVER_CUDA_CHECK(cudaEventSynchronize(__cudabeaver_stop));
    float __cudabeaver_total_ms = 0.0f;
    CUDABEAVER_CUDA_CHECK(cudaEventElapsedTime(&__cudabeaver_total_ms, __cudabeaver_start, __cudabeaver_stop));
    CUDABEAVER_CUDA_CHECK(cudaEventDestroy(__cudabeaver_start));
    CUDABEAVER_CUDA_CHECK(cudaEventDestroy(__cudabeaver_stop));
    __cudabeaver_print_cuda_event_timing(__cudabeaver_total_ms, 1.0);
    return __cudabeaver_rc;
}}
'''
    return renamed + wrapper, True


def _is_benchmark_driver(path: Path, text: str) -> bool:
    name = path.name.lower()
    if name in {"test_main.cu", "benchmark_main.cu"}:
        return True
    return "--perf" in text and re.search(r"\bint\s+main\s*\(", text) is not None


def _inject_cuda_event_timing(workdir: Path) -> dict:
    """Materialize cudaEvent stdout timing in the copied benchmark workdir."""
    changed_files: list[str] = []
    nvtx_regions = 0
    timed_loops = 0
    cu_files = sorted(workdir.rglob("*.cu"))

    for src_path in cu_files:
        text = src_path.read_text()
        rewritten, count = _rewrite_nvtx_regions_to_cuda_events(text)
        if count:
            src_path.write_text(rewritten)
            changed_files.append(str(src_path.relative_to(workdir)))
            nvtx_regions += count

    if nvtx_regions == 0:
        for src_path in cu_files:
            text = src_path.read_text()
            if not _is_benchmark_driver(src_path, text):
                continue
            rewritten, loop_count = _rewrite_timed_loop_to_cuda_events(text)
            if loop_count:
                src_path.write_text(rewritten)
                changed_files.append(str(src_path.relative_to(workdir)))
                timed_loops += loop_count

    wrapped_main = False
    if nvtx_regions == 0 and timed_loops == 0:
        solution = workdir / "solution.cu"
        if solution.exists():
            text = solution.read_text()
            rewritten, wrapped_main = _wrap_solution_main_with_cuda_event(text)
            if wrapped_main:
                solution.write_text(rewritten)
                changed_files.append("solution.cu")

    return {
        "changed_files": changed_files,
        "nvtx_regions": nvtx_regions,
        "timed_loops": timed_loops,
        "wrapped_solution_main": wrapped_main,
    }


def _run_pg(cmd: str, *, cwd, env, timeout: int):
    """Run cmd in a new process group. On timeout, SIGKILL the whole group
    so grandchildren (kernelbench eval python under sh -c) don't orphan.
    Returns subprocess.CompletedProcess-like with returncode/stdout/stderr.
    """
    proc = subprocess.Popen(
        cmd, shell=True, cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
        return subprocess.CompletedProcess(cmd, proc.returncode, out, err)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            out, err = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            out, err = "", "killed (process group timeout)"
        raise subprocess.TimeoutExpired(cmd, timeout, output=out, stderr=err)


def _execute_benchmark(
    ref: Reference,
    workdir: Path,
    env: dict,
    dataset_root: Path | str | None = None,
) -> Tuple[dict, float]:
    """Run benchmark_command in workdir, parse with the timing-mode adapter.

    Two-phase: build first via `build_command` from the matching input/{stem}.json
    if such a build is required (curated_cuda_pool needs nvcc; kernelbench is run-only).
    Then wrap the benchmark_command per the timing-mode adapter.
    """
    from .timing_modes import get_timing_mode
    from .compiler import adapt_gpu_arch_cmd

    # Per-task isolated torch_extensions cache. LLM solutions frequently use
    # generic load_inline names (e.g. name="fused_ops"). Sharing the global
    # ~/.cache/torch_extensions/ across tasks causes ninja-lock contention
    # AND can return a stale .so from another task — both have happened.
    ext_dir = workdir / ".torch_ext"
    ext_dir.mkdir(exist_ok=True)
    env = dict(env)
    env["TORCH_EXTENSIONS_DIR"] = str(ext_dir)

    parser = get_timing_mode(ref.timing_mode)
    if ref.timing_mode.type == "applied_kernels_make_speedup":
        cached_ref = cache_lookup(ref, dataset_root)
        if cached_ref is not None:
            ref_mean = cached_ref.timing.get("mean_ms")
            if isinstance(ref_mean, (int, float)) and ref_mean > 0:
                env["CUDABEAVER_REFERENCE_MEAN_MS"] = str(ref_mean)

    if ref.timing_mode.type in {"cuda_event_stdout", "region", "kernels"}:
        injection = _inject_cuda_event_timing(workdir)
        if not injection["changed_files"]:
            raise RuntimeError(
                f"{ref.timing_mode.type} requested for {ref.task_id}, but no benchmark "
                "region, timed loop, or solution.cu main() could be instrumented"
            )

    # Phase 1: build (if input/{stem}.json has a build_command). Skip for
    # kernelbench tasks since their build_command is a static checker that
    # rejects valid PyTorch reference baselines; the kernelbench eval itself
    # does any required JIT compilation when called.
    if ref.timing_mode.type != "kernelbench_eval_result":
        stem = task_id_to_stem(ref.task_id)
        try:
            input_json = resolve_task_paths(ref.task_id, _dataset_root(dataset_root)).input_path
        except FileNotFoundError:
            input_json = _dataset_root(dataset_root) / "input" / f"{stem}.json"
        if input_json.exists():
            meta = json.loads(input_json.read_text())
            build_cmd = meta.get("build_command")
            if build_cmd and meta.get("level") not in {
                "applied_kernels_class",
                "cutlass_class",
                "thunderkittens",
            }:
                if ref.benchmark_command == "./benchmark.out":
                    build_cmd = _cuda_benchmark_build_command(build_cmd, workdir) or build_cmd
                build_cmd = adapt_gpu_arch_cmd(re.sub(r"\bpython\b(?!3)", _python_bin(), build_cmd))
                bp = _run_pg(build_cmd, cwd=workdir, env=env, timeout=300)
                if bp.returncode != 0:
                    raise RuntimeError(
                        f"build failed for {ref.task_id}: rc={bp.returncode}\n"
                        f"stderr_tail: {bp.stderr[-500:]}"
                    )

    # Phase 2: benchmark
    cmd = adapt_gpu_arch_cmd(ref.benchmark_command)
    cmd = re.sub(r"\bpython\b(?!3)", _python_bin(), cmd)
    cmd = parser.wrap_command(cmd, str(workdir))

    # Per-GPU perf serialization. Multiple daemon children sharing one GPU can
    # collide on CUDA driver resources. flock the cmd by CUDA_VISIBLE_DEVICES so
    # only one perf process at a time per physical GPU.
    gpu_for_lock = env.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip() or "0"
    lock_dir = os.environ.get("CUDA_DEBUGGER_LOCK_DIR", "/tmp")
    lock_path = f"{lock_dir}/cuda_debugger_perf_lock_gpu{gpu_for_lock}.lock"
    import shlex
    cmd = f"flock {shlex.quote(lock_path)} sh -c {shlex.quote(cmd)}"

    t0 = time.time()
    proc = _run_pg(cmd, cwd=workdir, env=env, timeout=1800)
    elapsed = time.time() - t0

    timing = parser.parse(stdout=proc.stdout, stderr=proc.stderr)
    timing.setdefault("n_trials", ref.n_trials)
    timing.setdefault("warmup_trials", ref.warmup_trials)
    return timing, elapsed


def _arch_compatible(ref: Reference) -> bool:
    if ref.requires_arch is None:
        return True
    sig = gpu_signature()
    needed = ref.requires_arch.replace("a", "")  # sm_90a -> sm_90
    return needed in sig


def _now_iso() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _cuda_version_from_signature() -> str:
    m = re.search(r"cuda_(\d+_\d+)", gpu_signature())
    return m.group(1).replace("_", ".") if m else ""


def run_reference(
    task_id: str,
    gpu_id: int = 0,
    dataset_root: Path | str | None = None,
) -> CacheEntry:
    ref = load_reference(task_id, dataset_root)
    cached = cache_lookup(ref, dataset_root)
    if cached is not None:
        return cached

    if not _arch_compatible(ref):
        entry = CacheEntry(
            task_id=task_id,
            reference_hash=ref.canonical_hash(),
            gpu_signature=gpu_signature(),
            cuda_version="",
            timing={},
            measured_at=_now_iso(),
            elapsed_total_s=0.0,
            skipped="incompatible_arch",
            timing_fingerprint=ref.timing_fingerprint(),
        )
        cache_write(entry, dataset_root)
        return entry

    workdir = prepare_workdir(task_id, ref.reference_solution, dataset_root)
    env = _build_env(gpu_id)
    timing, elapsed = _execute_benchmark(ref, workdir, env, dataset_root)
    entry = CacheEntry(
        task_id=task_id,
        reference_hash=ref.canonical_hash(),
        gpu_signature=gpu_signature(),
        cuda_version=_cuda_version_from_signature(),
        timing=timing,
        measured_at=_now_iso(),
        elapsed_total_s=elapsed,
        skipped=None,
        timing_fingerprint=ref.timing_fingerprint(),
    )
    cache_write(entry, dataset_root)
    return entry


@dataclass
class CandidateTiming:
    task_id: str
    timing: dict
    elapsed_total_s: float
    measured_at: str


def run_candidate(
    task_id: str,
    candidate_files: list,
    gpu_id: int = 0,
    n_runs: int = 1,
    dataset_root: Path | str | None = None,
) -> CandidateTiming:
    """Build + time the candidate. When n_runs > 1, repeat the timing
    (each run still does its own internal 100 trials) and compute CV
    across the n_runs mean values. CV ('coefficient of variation') is
    reported as `cv` in the timing dict; absent when n_runs == 1 or the
    parser already reports stddev_ms > 0.
    """
    ref = load_reference(task_id, dataset_root)
    if not _arch_compatible(ref):
        return CandidateTiming(
            task_id=task_id,
            timing={"skipped": "incompatible_arch"},
            elapsed_total_s=0.0,
            measured_at=_now_iso(),
        )
    cand_solution = ReferenceSolution(type="file", files=candidate_files)
    workdir = prepare_workdir(task_id, cand_solution, dataset_root)
    env = _build_env(gpu_id)

    n_runs = max(1, int(n_runs))
    if n_runs == 1:
        timing, elapsed = _execute_benchmark(ref, workdir, env, dataset_root)
        # If parser reported a real stddev, derive CV
        mean = timing.get("mean_ms")
        std = timing.get("stddev_ms")
        if isinstance(mean, (int, float)) and isinstance(std, (int, float)) and mean > 0 and std > 0:
            timing.setdefault("cv", std / mean)
        timing.setdefault("n_runs", 1)
        return CandidateTiming(
            task_id=task_id, timing=timing, elapsed_total_s=elapsed,
            measured_at=_now_iso(),
        )

    # n_runs > 1 — repeat the full measurement; each run already does
    # n_trials internal trials. CV is across the n_runs means.
    means: list = []
    last_timing = None
    total_elapsed = 0.0
    for _ in range(n_runs):
        timing, elapsed = _execute_benchmark(ref, workdir, env, dataset_root)
        total_elapsed += elapsed
        last_timing = timing
        m = timing.get("mean_ms")
        if isinstance(m, (int, float)) and m > 0:
            means.append(m)

    if not means or last_timing is None:
        # All runs failed timing — return whatever last gave us
        return CandidateTiming(
            task_id=task_id, timing=last_timing or {"skipped": "no_timing"},
            elapsed_total_s=total_elapsed, measured_at=_now_iso(),
        )

    import statistics
    mean = statistics.fmean(means)
    std = statistics.stdev(means) if len(means) >= 2 else 0.0
    last_timing["mean_ms"] = mean
    last_timing["stddev_ms"] = std
    last_timing["cv"] = (std / mean) if mean > 0 else float("nan")
    last_timing["n_runs"] = n_runs
    last_timing["per_run_means_ms"] = means
    return CandidateTiming(
        task_id=task_id, timing=last_timing,
        elapsed_total_s=total_elapsed, measured_at=_now_iso(),
    )


def compute_speedup(reference_mean_ms: float, candidate_mean_ms: float) -> float:
    if candidate_mean_ms <= 0:
        return float("nan")
    return reference_mean_ms / candidate_mean_ms
