"""Performance measurement helpers shared by experiment runners."""

from __future__ import annotations

import math
from pathlib import Path

from .benchmark import (
    cache_lookup as _bench_cache_lookup,
    compute_speedup as _bench_compute_speedup,
    load_reference as _bench_load_reference,
    run_candidate as _bench_run_candidate,
)


def measure_iter_perf(
    task_id: str,
    code: str,
    solution_file: str,
    test_output: str | None = None,
    gpu_id: int = 0,
    dataset_dir: Path | str | None = None,
) -> dict:
    """Measure a passing candidate against the cached reference.

    Always returns a dict. When measurement is unavailable, the dict contains
    a `skipped` reason instead of raising.
    """
    try:
        ref = _bench_load_reference(task_id, dataset_root=dataset_dir)
    except FileNotFoundError:
        if test_output:
            from .classifier import _extract_kh_speedups

            kh_sps = _extract_kh_speedups(test_output)
            if kh_sps:
                import statistics

                geomean = statistics.geometric_mean(kh_sps)
                return {
                    "speedup": geomean,
                    "kh_per_size_speedups": kh_sps,
                    "kh_min_speedup": min(kh_sps),
                    "kh_max_speedup": max(kh_sps),
                    "timing_method": "kh_test_output_geomean",
                    "n_trials": None,
                }
        return {"skipped": "reference_not_defined"}

    try:
        ref_entry = _bench_cache_lookup(ref, dataset_root=dataset_dir)
    except Exception as e:
        return {
            "skipped": "reference_cache_lookup_failed",
            "error": f"{type(e).__name__}: {str(e)[:200]}",
        }
    if ref_entry is None:
        return {"skipped": "reference_not_cached"}
    if ref_entry.skipped:
        return {"skipped": ref_entry.skipped}

    files = [{"path": solution_file, "content": code}]
    try:
        cand = _bench_run_candidate(
            task_id,
            files,
            gpu_id=gpu_id,
            dataset_root=dataset_dir,
        )
    except Exception as e:
        return {
            "skipped": "build_failed_on_perf",
            "error": f"{type(e).__name__}: {str(e)[:4000]}",
        }

    if "skipped" in cand.timing:
        return {"skipped": cand.timing["skipped"]}

    ref_mean = ref_entry.timing.get("mean_ms")
    cand_mean = cand.timing.get("mean_ms", float("nan"))
    if ref_mean is None or math.isnan(cand_mean) or cand_mean <= 0:
        return {
            "skipped": "candidate_runtime_invalid",
            "candidate_runtime_ms": cand_mean,
        }

    cv = cand.timing.get("cv")
    cv_threshold = 0.03
    cv_ok = (cv is not None) and (cv < cv_threshold)
    return {
        "candidate_runtime_ms": cand_mean,
        "candidate_stddev_ms": cand.timing.get("stddev_ms", 0.0),
        "candidate_cv": cv,
        "candidate_cv_ok": cv_ok,
        "candidate_n_runs": cand.timing.get("n_runs", 1),
        "reference_runtime_ms": ref_mean,
        "reference_stddev_ms": ref_entry.timing.get("stddev_ms", 0.0),
        "speedup": _bench_compute_speedup(ref_mean, cand_mean),
        "timing_method": cand.timing.get("method"),
        "n_trials": cand.timing.get("n_trials", ref.n_trials),
        "measured_at": cand.measured_at,
    }


def evaluate_performance_gate(perf: dict, gate_cfg: dict | None = None) -> dict:
    """Return structured pass/fail information for a performance gate."""
    cfg = dict(gate_cfg or {})
    enabled = bool(cfg.get("enabled", True))
    min_speedup = float(cfg.get("min_speedup", cfg.get("threshold_speedup", 0.3)))
    fail_on_skipped = bool(cfg.get("fail_on_skipped", False))
    result = {
        "enabled": enabled,
        "min_speedup": min_speedup,
        "passed": True,
        "reason": "disabled" if not enabled else "ok",
    }
    if not enabled:
        return result

    if "speedup" not in perf:
        result["passed"] = not fail_on_skipped
        result["reason"] = (
            f"skipped:{perf.get('skipped', 'unknown')}"
            if not result["passed"]
            else "skipped_allowed"
        )
        return result

    speedup = float(perf["speedup"])
    result["speedup"] = speedup
    if speedup < min_speedup:
        result["passed"] = False
        result["reason"] = "speedup_below_threshold"
    return result
