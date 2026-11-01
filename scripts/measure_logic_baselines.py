#!/usr/bin/env python3
"""Measure logic-error task baselines on one GPU.

This utility is intentionally separate from the agent loop: it materializes the
performance reference descriptions that the harness expects where possible,
then runs/cache-measures each task's good/reference implementation.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.benchmark import (
    CacheEntry,
    cache_write,
    gpu_signature,
    load_reference,
    run_reference,
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _refs_dir(dataset_root: Path) -> Path:
    return dataset_root / "evaluation" / "references"


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def _kernelbench_command(correct_trials: int, perf_trials: int) -> str:
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


def materialize_harness_reference(entry: dict, dataset_root: Path, perf_trials: int) -> bool:
    task_id = entry["task_id"]
    meta = _read_json(ROOT / entry["dataset_paths"]["input"])
    level = meta.get("level")
    ref_path = ROOT / entry["dataset_paths"]["references"]
    out = _refs_dir(dataset_root) / f"{task_id}.json"

    if level == "curated_cuda_pool":
        data = {
            "task_id": task_id,
            "reference_solution": {
                "type": "file",
                "files": [
                    {
                        "path": meta.get("solution_file", "solution.cu"),
                        "content": ref_path.read_text(),
                    }
                ],
            },
            "benchmark_command": "./test.out --perf",
            "timing_mode": {
                "type": "cuda_event_stdout",
            },
            "n_trials": 100,
            "warmup_trials": 3,
        }
        _write_json(out, data)
        return True

    if level in {"kernelbench_level1", "kernelbench_level2"}:
        problem = ROOT / entry["dataset_paths"]["testbench"] / "problem.py"
        correct_trials = 2 if level == "kernelbench_level2" else 5
        data = {
            "task_id": task_id,
            "reference_solution": {
                "type": "kernelbench_problem",
                "problem_path": f"testbench/{task_id}/problem.py",
                "problem_content": problem.read_text(),
            },
            "benchmark_command": _kernelbench_command(correct_trials, perf_trials),
            "timing_mode": {"type": "kernelbench_eval_result"},
            "n_trials": perf_trials,
            "warmup_trials": 0,
        }
        _write_json(out, data)
        return True

    return False


_REF_TIME_RE = re.compile(
    r"Ref time:\s*([0-9.eE+-]+)\s*ms\s*"
    r"\(avg over\s*([0-9]+)\s*trials,\s*min:\s*([0-9.eE+-]+)\s*ms\)",
    re.IGNORECASE,
)
_CASE_RE = re.compile(r"^===\s*(.*?)\s*===", re.MULTILINE)


def _parse_cutlass_ref_times(text: str) -> list[dict]:
    case_matches = list(_CASE_RE.finditer(text))
    ref_matches = list(_REF_TIME_RE.finditer(text))
    rows: list[dict] = []
    for m in ref_matches:
        label = None
        for cm in case_matches:
            if cm.start() < m.start():
                label = cm.group(1).strip()
            else:
                break
        rows.append(
            {
                "case": label,
                "mean_ms": float(m.group(1)),
                "n_trials": int(m.group(2)),
                "min_ms": float(m.group(3)),
            }
        )
    return rows


def _skip_cutlass_cpu_reference(workdir: Path) -> None:
    """Patch copied CUTLASS testbench sources to time only GPU reference paths.

    The default CUTLASS class testbenches run host/CPU reference correctness
    before profiling. Baseline timing only needs the reference kernel timing,
    and some large CPU checks are prohibitively slow.
    """
    for src in workdir.glob("*.cu"):
        text = src.read_text()
        updated = text

        updated = re.sub(
            r"(cudaError_t\s+TestCorrectness\s*\([^)]*\)\s*\{)",
            (
                r"\1\n"
                "  // Baseline timing run: skip CPU/reference correctness.\n"
                "  return cudaSuccess;\n"
            ),
            updated,
            count=1,
            flags=re.DOTALL,
        )

        if "Disposition run()" in updated:
            updated = updated.replace(
                "Disposition disposition = Disposition::kNotVerified;",
                "Disposition disposition = Disposition::kPassed;",
                1,
            )
            updated = updated.replace(
                "compute_reference();",
                "/* compute_reference skipped for baseline timing. */",
                1,
            )
            updated = updated.replace(
                "if (options.verification_enabled) {",
                "if (false && options.verification_enabled) {",
                1,
            )

        if updated != text:
            src.write_text(updated)


def measure_cutlass_reference(entry: dict, dataset_root: Path, gpu: int, timeout_s: int) -> dict:
    task_id = entry["task_id"]
    testbench = ROOT / entry["dataset_paths"]["testbench"]

    workdir = Path(tempfile.mkdtemp(prefix=f"baseline_{task_id}_"))
    shutil.copytree(testbench, workdir, symlinks=True, dirs_exist_ok=True)
    _skip_cutlass_cpu_reference(workdir)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    build_cmd = (
        "make ref "
        "CUTLASS_DIR=./reference_sources/cutlass "
        "TK_DIR=./reference_sources/ThunderKittens"
    )
    t0 = time.time()
    build = subprocess.run(
        build_cmd,
        shell=True,
        cwd=workdir,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s,
    )
    if build.returncode != 0:
        raise RuntimeError(
            f"build failed rc={build.returncode}\n"
            f"stdout_tail={build.stdout[-1000:]}\n"
            f"stderr_tail={build.stderr[-1000:]}"
        )

    run = subprocess.run(
        "./ref",
        shell=True,
        cwd=workdir,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s,
    )
    elapsed = time.time() - t0
    text = (run.stdout or "") + "\n" + (run.stderr or "")
    if run.returncode != 0:
        raise RuntimeError(
            f"run failed rc={run.returncode}\n"
            f"stdout_tail={run.stdout[-1000:]}\n"
            f"stderr_tail={run.stderr[-1000:]}"
        )

    per_case = _parse_cutlass_ref_times(text)
    if not per_case:
        raise RuntimeError(f"no Ref time lines found in output tail:\n{text[-1000:]}")

    means = [x["mean_ms"] for x in per_case if x["mean_ms"] > 0]
    mins = [x["min_ms"] for x in per_case if x["min_ms"] > 0]
    mean_ms = statistics.geometric_mean(means)
    min_geomean_ms = statistics.geometric_mean(mins)
    rec = {
        "task_id": task_id,
        "status": "ok",
        "level": "cutlass_class",
        "timing": {
            "method": "cutlass_ref_time_geomean",
            "mean_ms": mean_ms,
            "min_geomean_ms": min_geomean_ms,
            "per_case": per_case,
            "n_cases": len(per_case),
        },
        "elapsed_total_s": elapsed,
        "workdir": str(workdir),
    }
    ref = load_reference(task_id, dataset_root=dataset_root)
    sig = gpu_signature()
    cuda_match = re.search(r"cuda_(\d+_\d+)", sig)
    cache_write(
        CacheEntry(
            task_id=task_id,
            reference_hash=ref.canonical_hash(),
            gpu_signature=sig,
            cuda_version=cuda_match.group(1).replace("_", ".") if cuda_match else "",
            timing=rec["timing"],
            measured_at=_dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            elapsed_total_s=elapsed,
            skipped=None,
            timing_fingerprint=ref.timing_fingerprint(),
        ),
        dataset_root=dataset_root,
    )
    return rec


def measure_harness_reference(entry: dict, dataset_root: Path, gpu: int, perf_trials: int) -> dict:
    task_id = entry["task_id"]
    meta = _read_json(ROOT / entry["dataset_paths"]["input"])
    materialize_harness_reference(entry, dataset_root, perf_trials)
    t0 = time.time()
    cache_entry = run_reference(task_id, gpu_id=gpu, dataset_root=dataset_root)
    elapsed = time.time() - t0
    d = asdict(cache_entry)
    d["status"] = "skipped" if cache_entry.skipped else "ok"
    d["level"] = meta.get("level")
    d["cache_lookup_elapsed_s"] = elapsed
    return d


def _summarize(records: list[dict]) -> dict:
    by_level: dict[str, dict] = {}
    for r in records:
        level = r.get("level", "unknown")
        bucket = by_level.setdefault(level, {"ok": 0, "failed": 0, "skipped": 0})
        status = r.get("status")
        if status not in bucket:
            status = "failed"
        bucket[status] += 1
    ok_times = [
        r.get("timing", {}).get("mean_ms")
        for r in records
        if r.get("status") == "ok"
        and isinstance(r.get("timing", {}).get("mean_ms"), (int, float))
        and not math.isnan(r["timing"]["mean_ms"])
    ]
    return {
        "n_total": len(records),
        "n_ok": sum(1 for r in records if r.get("status") == "ok"),
        "n_failed": sum(1 for r in records if r.get("status") == "failed"),
        "n_skipped": sum(1 for r in records if r.get("status") == "skipped"),
        "by_level": by_level,
        "ok_mean_ms_geomean": statistics.geometric_mean(ok_times) if ok_times else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-list", default=str(ROOT / "benchmark" / "task_list.json"))
    parser.add_argument("--dataset-root", default=str(ROOT / "data"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--kernelbench-trials", type=int, default=100)
    parser.add_argument("--timeout-s", type=int, default=1800)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--error-category",
        default=None,
        help="Optional broken_start error category filter, e.g. memory_crash or logic_error.",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional explicit task ids/stems to measure.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    task_list = _read_json(Path(args.task_list))
    entries = task_list["entries"]
    if args.error_category:
        entries = [
            e for e in entries
            if (e.get("broken_start") or {}).get("error_category") == args.error_category
        ]
    if args.tasks:
        wanted = set(args.tasks)
        entries = [
            e for e in entries
            if e.get("task_id") in wanted or e.get("stem") in wanted
        ]
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    category_part = f"_{args.error_category}" if args.error_category else ""
    out = Path(args.out or f"logs/reference_baselines{category_part}_gpu{args.gpu}_{stamp}.jsonl")
    summary_out = out.with_suffix(".summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    records: list[dict] = []
    print(f"writing results to {out}", flush=True)
    for i, entry in enumerate(entries, start=1):
        task_id = entry["task_id"]
        meta = _read_json(ROOT / entry["dataset_paths"]["input"])
        level = meta.get("level")
        print(f"[{i:02d}/{len(entries):02d}] {task_id} ({level})", flush=True)
        t0 = time.time()
        try:
            if level == "cutlass_class":
                rec = measure_cutlass_reference(entry, dataset_root, args.gpu, args.timeout_s)
            else:
                rec = measure_harness_reference(
                    entry, dataset_root, args.gpu, args.kernelbench_trials
                )
            rec.setdefault("task_id", task_id)
            rec.setdefault("level", level)
            rec.setdefault("status", "ok")
        except Exception as exc:
            rec = {
                "task_id": task_id,
                "level": level,
                "status": "failed",
                "error": f"{type(exc).__name__}: {str(exc)[:4000]}",
                "elapsed_total_s": time.time() - t0,
            }
        records.append(rec)
        with out.open("a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        timing = rec.get("timing", {})
        mean = timing.get("mean_ms")
        mean_text = f"{mean:.6g} ms" if isinstance(mean, (int, float)) else "-"
        print(
            f"    {rec.get('status')} mean={mean_text} elapsed={rec.get('elapsed_total_s', 0):.1f}s",
            flush=True,
        )
        if rec.get("status") == "failed":
            print(f"    error: {rec.get('error', '')[:600]}", flush=True)

    summary = _summarize(records)
    summary["results_jsonl"] = str(out)
    summary["gpu"] = args.gpu
    summary["kernelbench_trials"] = args.kernelbench_trials
    _write_json(summary_out, summary)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"summary written to {summary_out}", flush=True)
    return 0 if summary["n_failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
