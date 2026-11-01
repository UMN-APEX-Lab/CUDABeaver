#!/usr/bin/env python3
"""
CUDA-Debugger Experiment Pipeline
==================================
Runs GPT-5.2 (or any OpenAI-compatible model) on the CUDA-Debugger benchmark tasks.

Flow per task:
  1. Read prompt + testbench from CUDA-Debugger dataset
  2. Call LLM to generate solution.cu
  3. Compile with nvcc (using build_command from input JSON)
  4. Run test (using test_command)
  5. Classify result into five categories
  6. If failed, construct feedback prompt and goto step 2
  7. Record all iterations in manifest.jsonl (same format as CUDA-Debugger)
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from .classifier import classify_result
from .compiler import compile_code, run_test
from .feedback import build_feedback
from .gpu_compat import get_gpu_compute_cap, check_task_gpu_compat
from .generate_headers import setup_include_dir
from .llm_client import call_llm, extract_code, create_client
from .memory_updater import update_memory
from .metrics import compute_task_metrics, compute_aggregate_metrics, metrics_to_dict
from .performance import evaluate_performance_gate as _evaluate_performance_gate
from .benchmark import (
    run_candidate as _bench_run_candidate,
    load_reference as _bench_load_reference,
    cache_lookup as _bench_cache_lookup,
    compute_speedup as _bench_compute_speedup,
)
from .stagnation import check_stagnation, make_record as make_stagnation_record
from .task_registry import (
    broken_start_map,
    discover_task_stems,
    load_input_meta,
    load_prompt_text,
    task_error_category,
)

logger = logging.getLogger("cuda_debugger_exp")

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_CUDA = """\
You are an expert CUDA programmer. You will be given a task description \
for writing CUDA kernel code. Write ONLY the solution code that should go \
into `solution.cu`. Do not include any explanation, just the code. \
Do not include a main function — the test harness provides one.

OUTPUT REQUIREMENT:
- do not overthink.
- You must generate the complete contents of `solution.cu` before stopping.
- Do not emit prose, markdown, comments outside the code, or analysis."""

SYSTEM_PROMPT_PYTHON = """\
You are an expert CUDA/PyTorch programmer. You will be given a task description \
for writing a custom CUDA kernel as a PyTorch extension. Write ONLY the solution \
code that should go into `solution.py`. Do not include any explanation, just the code.

OUTPUT REQUIREMENT:
- do not overthink.
- You must generate the complete contents of `solution.py` before stopping.
- Do not emit prose, markdown, comments outside the code, or analysis."""

SYSTEM_PROMPT_GENERIC = """\
You are an expert CUDA programmer. You will be given a task description \
for writing CUDA kernel code. Write ONLY the solution code that should go \
into `{solution_file}`. Do not include any explanation, just the code.

OUTPUT REQUIREMENT:
- do not overthink.
- You must generate the complete contents of `{solution_file}` before stopping.
- Do not emit prose, markdown, comments outside the code, or analysis."""

FIRST_TURN_TEMPLATE = """\
Write the code for the following task. Output ONLY the content of `{solution_file}`.

## Task Description
{prompt}
{performance_requirement}

OUTPUT REQUIREMENT:
- do not overthink.
- You must generate the complete contents of `{solution_file}` before stopping.
- Output only code.
"""

# v2 debug-mode prompt: model starts from a broken implementation rather than
# generating from scratch. Always provides the broken code; provides a build
# log excerpt only if available (some manual broken-starts have no log).
DEBUG_FIRST_TURN_TEMPLATE = """\
You are debugging a CUDA kernel implementation. The current code FAILS the \
correctness or performance check. Find the bug(s) and return the FIXED full \
contents of `{solution_file}` — output ONLY the fixed code, no explanation.

## Task Description
{prompt}

## Current (broken) implementation
The following code is in `{solution_file}` and currently fails:

```cuda
{broken_code}
```

## Why it fails
Error category: `{error_category}`
{error_excerpt}
{performance_requirement}

Output the fixed full contents of `{solution_file}`.

OUTPUT REQUIREMENT:
- do not overthink.
- You must generate the complete fixed contents of `{solution_file}` before stopping.
- Output only code.
"""


def _get_system_prompt(input_meta: dict) -> str:
    solution_file = input_meta.get("solution_file", "solution.cu")
    if solution_file == "solution.cu":
        return SYSTEM_PROMPT_CUDA
    elif solution_file == "solution.py":
        return SYSTEM_PROMPT_PYTHON
    else:
        return SYSTEM_PROMPT_GENERIC.format(solution_file=solution_file)

# FEEDBACK_TEMPLATE removed — now using feedback.py with error-aware templates


# Few-shot example pool — built lazily once per process (Exp 7).
_FEW_SHOT_POOL_CACHE: dict = {}


def _few_shot_section(config: dict, broken_start: dict | None, task_stem: str) -> str:
    """Build the few-shot examples section to append to the system prompt.

    No-op (returns "") unless config["experiment"]["few_shot_k"] > 0 AND
    a strategy is set. LOO contamination prevention applied via
    few_shot.select_examples.
    """
    exp = config.get("experiment", {})
    K = int(exp.get("few_shot_k", 0))
    if K <= 0:
        return ""
    strategy = exp.get("few_shot_strategy", "random")
    seed = int(exp.get("few_shot_pair_pool_seed", 42))
    pool_committee = exp.get(
        "few_shot_pool_committee_dirs",
        [
            "evaluation/outputs/gemma4/20260424_123214",
            "evaluation/outputs/glm_4_7/20260424_151941",
        ],
    )
    cache_key = tuple(pool_committee)
    if cache_key not in _FEW_SHOT_POOL_CACHE:
        from .few_shot import build_pair_pool
        repo_root = Path(__file__).resolve().parents[1]
        dirs = [repo_root / d for d in pool_committee if (repo_root / d).exists()]
        _FEW_SHOT_POOL_CACHE[cache_key] = build_pair_pool(dirs)
        logger.info(
            f"few_shot pool built from {len(dirs)} dirs: "
            f"{len(_FEW_SHOT_POOL_CACHE[cache_key])} pairs available"
        )
    pool = _FEW_SHOT_POOL_CACHE[cache_key]

    # Resolve test task error_category (for matched strategy)
    err_cat = "unknown"
    if broken_start:
        err_cat = broken_start.get("error_category", "unknown")

    from .few_shot import select_examples, format_examples_for_system

    # Map task_stem back to task_id for LOO check
    task_id = task_stem.replace("_", "/", 1)
    curated_file = exp.get("few_shot_curated_file")
    if curated_file:
        # Resolve relative to repo root for portability
        cf_path = Path(curated_file)
        if not cf_path.is_absolute():
            cf_path = Path(__file__).resolve().parents[1] / curated_file
        curated_file = str(cf_path)
    examples = select_examples(
        pool=pool,
        test_task_id=task_id,
        test_error_category=err_cat,
        strategy=strategy,
        K=K,
        seed=seed,
        curated_file=curated_file,
    )
    return format_examples_for_system(examples)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def sha256_prefix(code: str, length: int = 16) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:length]


def _performance_gate_config(config: dict) -> dict | None:
    """Return normalized YAML performance gate config, if enabled."""
    raw = config.get("performance_gate")
    if raw is None:
        raw = config.get("experiment", {}).get("performance_gate")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("performance_gate must be a YAML mapping")
    if not raw.get("enabled", True):
        return {"enabled": False}
    min_speedup = raw.get("min_speedup", raw.get("threshold"))
    if min_speedup is None:
        raise ValueError("enabled performance_gate requires min_speedup")
    return {
        "enabled": True,
        "min_speedup": float(min_speedup),
        "fail_on_skipped": bool(raw.get("fail_on_skipped", False)),
    }


def _measure_pass_performance_enabled(config: dict) -> bool:
    """Whether to run post-pass benchmark timing for manifest metadata."""
    exp = config.get("experiment", {}) or {}
    return bool(exp.get("measure_perf_on_pass", True))


def _performance_requirement_section(performance_gate: dict | None) -> str:
    if not performance_gate or not performance_gate.get("enabled", True):
        return ""
    threshold = float(performance_gate.get("min_speedup", 0.0))
    return (
        "\n## Performance Requirement\n"
        f"- A candidate only passes if it is correct and reaches >= {threshold:.3f}x "
        "reference speedup.\n"
        "- Fix the bug without replacing parallel CUDA work with a slow serial "
        "fallback; preserve or improve memory coalescing, occupancy, and launch "
        "efficiency where relevant.\n"
    )


def _classification_from_perf_gate(perf: dict, perf_gate: dict) -> dict:
    threshold = float(perf_gate.get("min_speedup", 0.0))
    if "speedup" in perf_gate:
        category = "perf_below_threshold"
        error_type = "perf_gate"
        detail = (
            f"correctness OK but speedup {float(perf_gate['speedup']):.3f}x "
            f"< {threshold:.3f}x reference"
        )
    else:
        category = "perf_unmeasured"
        error_type = "perf_gate_unmeasured"
        reason = perf_gate.get("reason", "performance_unmeasured")
        detail = f"correctness OK but performance gate failed: {reason}"
    return {
        "category": category,
        "five_category": "logic_error",
        "error_type": error_type,
        "build_ok": True,
        "status": "failed",
        "logic_error_detail": detail,
        "performance_gate": perf_gate,
    }


def _classification_passed_from_perf_gate(perf_gate: dict) -> dict:
    return {
        "category": "passed",
        "five_category": "passed",
        "error_type": None,
        "build_ok": True,
        "status": "passed",
        "logic_error_category": None,
        "logic_error_detail": None,
        "performance_gate": perf_gate,
    }


def _needs_reference_perf_measurement(classification: dict) -> bool:
    """Correctness passed, but stdout did not contain a usable speedup.

    CUDA/CUDABench tests commonly report only correctness in stdout. In that
    case the runner must use the reference-performance cache before enforcing
    fail_on_skipped; otherwise correct CUDA fixes get mislabeled as
    perf_unmeasured before the benchmark adapter has a chance to run.
    """
    return (
        classification.get("category") == "perf_unmeasured"
        and classification.get("error_type") == "perf_gate_unmeasured"
        and bool(classification.get("build_ok"))
    )


def _has_stdout_perf_gate(classification: dict) -> bool:
    gate = classification.get("performance_gate") or {}
    return "speedup" in gate and gate.get("passed") is not None


# compile_code, run_test → compiler.py
# call_llm, extract_code → llm_client.py


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------


def _measure_iter_perf(
    task_id: str,
    code: str,
    solution_file: str,
    test_output: str | None = None,
    dataset_dir: Path | str | None = None,
) -> dict:
    """Inline perf measurement for a passing iter. Returns dict to attach to record.
    Always returns a dict with either runtime numbers or a 'skipped' reason —
    never raises. GPU selection is inherited via parent shell CUDA_VISIBLE_DEVICES.

    applied-kernels fallback: when reference_not_defined (applied-kernels tasks lack evaluation/references/
    JSON because they self-time inside the test binary), we parse "Speedup:" lines
    out of the applied-kernels ./test stdout (passed in via test_output) and report the geomean
    so downstream c-gate sweeps work without a separate perf re-run.
    """
    import math
    try:
        ref = _bench_load_reference(task_id, dataset_root=dataset_dir)
    except FileNotFoundError:
        # applied-kernels-aware fallback: parse Speedup: lines from applied-kernels ./test output
        if test_output:
            from .classifier import _extract_kh_speedups
            kh_sps = _extract_kh_speedups(test_output)
            if kh_sps:
                # Geomean to be robust to outlier sizes
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
    ref_entry = _bench_cache_lookup(ref, dataset_root=dataset_dir)
    if ref_entry is None:
        return {"skipped": "reference_not_cached"}
    if ref_entry.skipped:
        return {"skipped": ref_entry.skipped}
    files = [{"path": solution_file, "content": code}]
    try:
        cand = _bench_run_candidate(task_id, files, gpu_id=0, dataset_root=dataset_dir)
    except Exception as e:
        return {"skipped": "build_failed_on_perf", "error": str(e)[:200]}
    if "skipped" in cand.timing:
        return {"skipped": cand.timing["skipped"]}
    ref_mean = ref_entry.timing.get("mean_ms")
    cand_mean = cand.timing.get("mean_ms", float("nan"))
    if ref_mean is None or math.isnan(cand_mean) or cand_mean <= 0:
        return {"skipped": "candidate_runtime_invalid", "candidate_runtime_ms": cand_mean}
    # CV gating (Phase D2): record CV when available, plus a boolean flag
    # for the canonical 3% threshold. cv == None means single-shot timing
    # with no stddev — analysis must treat as untrusted.
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


def run_task(
    task_stem: str,
    dataset_dir: Path,
    output_dir: Path,
    config: dict,
    client,
    broken_start: dict | None = None,
) -> list[dict]:
    """Run all iterations for a single task. Returns list of manifest records.

    broken_start (v2 debug mode): if provided, the first LLM turn is a
    debug-prompt with the broken code + error excerpt prepended. Format:
        {'code_path': '...', 'error_category': '...', 'log_excerpt': '...'}
    The code_path is read relative to the repo root.

    Test-time method (Exp 2):
        config["experiment"]["method"] in {"iterative" (default), "repeated"}.
        - iterative: each iter sees prior attempts + cumulative feedback,
          stops early on pass/stagnation. (Existing behavior.)
        - repeated: each attempt is independent — same initial prompt, no
          feedback or history. Per-attempt seed for vLLM reproducibility.
          Runs all n_attempts (no early stop on pass — needed for pass@k).
    """
    method = config["experiment"].get("method", "iterative")
    mode = "debug" if broken_start else "generation"
    logger.info(f"=== Starting task: {task_stem}  (mode={mode}, method={method}) ===")

    if method == "agent":
        from .agent_loop import run_task_agent

        return run_task_agent(
            task_stem,
            dataset_dir,
            output_dir,
            config,
            client,
            broken_start=broken_start,
        )

    # Load task metadata from either legacy flat datasets or benchmark/task_list.json.
    input_meta = load_input_meta(task_stem, dataset_dir)
    prompt_text = load_prompt_text(task_stem, dataset_dir)

    max_iters = config["experiment"]["max_iterations"]
    workdir = Path(config["build"]["workdir"])
    build_timeout = config["build"]["timeout"]
    test_timeout = config["build"]["test_timeout"]
    performance_gate = _performance_gate_config(config)

    solution_file = input_meta.get("solution_file", "solution.cu")

    records = []
    if broken_start:
        # v2 debug mode: load broken code + format DEBUG_FIRST_TURN_TEMPLATE
        repo_root = Path(__file__).resolve().parents[1]
        bs_path = repo_root / broken_start["code_path"]
        broken_code = bs_path.read_text(encoding="utf-8")
        excerpt = broken_start.get("log_excerpt") or ""
        excerpt_section = (
            f"Error log excerpt:\n```\n{excerpt[:1500]}\n```"
            if excerpt else
            "(No error log captured for this starting code — infer the bug from the code + task spec.)"
        )
        first_user = DEBUG_FIRST_TURN_TEMPLATE.format(
            solution_file=solution_file,
            prompt=prompt_text,
            broken_code=broken_code,
            error_category=broken_start.get("error_category", "unknown"),
            error_excerpt=excerpt_section,
            performance_requirement=_performance_requirement_section(performance_gate),
        )
        system_msg = _get_system_prompt(input_meta) + _few_shot_section(
            config, broken_start, task_stem
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": first_user},
        ]
    else:
        system_msg = _get_system_prompt(input_meta) + _few_shot_section(
            config, broken_start, task_stem
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": FIRST_TURN_TEMPLATE.format(
                prompt=prompt_text,
                solution_file=solution_file,
                performance_requirement=_performance_requirement_section(performance_gate),
            )},
        ]

    stagnation_history = []
    stop_reason = "max_iterations"
    MAX_HISTORY_ROUNDS = config["experiment"].get("max_history_rounds", 4)

    # Snapshot the initial 2-message prompt: needed by repeated-method (every
    # attempt) and by iterative L0 feedback (reset on every iter, no history).
    initial_messages = list(messages)

    # Repeated-method may use n_attempts instead of max_iterations
    if method == "repeated":
        max_iters = config["experiment"].get("n_attempts", max_iters)

    for iteration in range(1, max_iters + 1):
        iter_label = f"{task_stem}__iter_{iteration:04d}"
        loop_label = "Attempt" if method == "repeated" else "Iteration"
        logger.info(f"  {loop_label} {iteration}/{max_iters} ({iter_label})")

        # Repeated: reset to initial prompt every attempt.
        if method == "repeated":
            messages = list(initial_messages)

        # Iterative: prune conversation: keep system + first user + last
        # MAX_HISTORY_ROUNDS round-trips
        if method == "iterative":
            fixed = 2
            if len(messages) > fixed + MAX_HISTORY_ROUNDS * 2:
                messages = messages[:fixed] + messages[-(MAX_HISTORY_ROUNDS * 2):]

        # Repeated-method: per-attempt seed (vLLM reproducible; cloud may ignore)
        call_config = config
        if method == "repeated":
            call_config = {
                **config,
                "api": {**config["api"], "seed": iteration},
            }

        # --- Call LLM ---
        t0 = time.time()
        try:
            raw_response = call_llm(client, call_config, messages)
        except Exception as e:
            import traceback
            err_msg = traceback.format_exc()
            logger.error(f"  API error: {e}\n{err_msg}")
            log_dir = output_dir / "build_log_raw"
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / f"{iter_label}.txt").write_text(
                f"BUILD_OK=False\n\n[API ERROR]\n{err_msg}", encoding="utf-8"
            )
            api_err_record = _make_record(
                iter_label, input_meta, task_stem, iteration,
                five_category="compile_error", error_type="api_error",
                build_ok=False, status="failed",
            )
            _annotate_method_fields(api_err_record, config, method)
            records.append(api_err_record)
            stop_reason = "api_error"
            break
        api_time = time.time() - t0
        logger.info(f"  API response in {api_time:.1f}s")

        code = extract_code(raw_response)
        code_hash = sha256_prefix(code)

        # --- Save code ---
        code_dir = output_dir / "error_code"
        code_dir.mkdir(parents=True, exist_ok=True)
        (code_dir / f"{code_hash}.cu").write_text(code, encoding="utf-8")

        # --- Save raw API response ---
        if config["logging"]["save_api_responses"]:
            api_dir = output_dir / "api_responses"
            api_dir.mkdir(parents=True, exist_ok=True)
            (api_dir / f"{iter_label}.txt").write_text(raw_response, encoding="utf-8")

        # --- Compile ---
        build_ok, build_output = compile_code(
            code, task_stem, dataset_dir, input_meta, workdir, build_timeout
        )

        # --- Test (only if build succeeded) ---
        test_returncode = None
        test_output = None
        if build_ok:
            test_returncode, test_output = run_test(
                task_stem, input_meta, workdir, test_timeout,
                performance_gate=performance_gate,
            )

        # --- Classify ---
        classification = classify_result(
            build_ok, build_output, test_output, test_returncode,
            performance_gate=performance_gate,
        )

        # --- Save build log ---
        log_dir = output_dir / "build_log_raw"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_content = f"BUILD_OK={build_ok}\n\n[BUILD OUTPUT]\n{build_output}"
        if test_output is not None:
            log_content += f"\n\n[TEST OUTPUT]\n{test_output}"
        (log_dir / f"{iter_label}.txt").write_text(log_content, encoding="utf-8")

        # --- Build manifest record ---
        record = _make_record(
            iter_label, input_meta, task_stem, iteration,
            code_hash=code_hash,
            **classification,
        )
        _annotate_method_fields(record, config, method)
        records.append(record)

        logger.info(
            f"  Result: {classification['five_category']} "
            f"(build_ok={build_ok})"
        )

        # --- Check if correctness passed, then enforce reference perf gate if needed ---
        correctness_passed = (
            classification["five_category"] == "passed"
            or _needs_reference_perf_measurement(classification)
        )
        if correctness_passed:
            if _needs_reference_perf_measurement(classification):
                logger.info(
                    f"  Correctness passed at {loop_label.lower()} {iteration}; "
                    "stdout speedup missing, measuring reference perf"
                )
            else:
                logger.info(f"  Correctness passed at {loop_label.lower()} {iteration}")

            stdout_gate = _has_stdout_perf_gate(classification)
            if _measure_pass_performance_enabled(config) and not (
                performance_gate is not None and stdout_gate
            ):
                # Inline perf: measured every time a candidate passes.
                perf = _measure_iter_perf(
                    input_meta.get("task_id", task_stem),
                    code,
                    solution_file,
                    test_output=test_output,
                    dataset_dir=dataset_dir,
                )
                record["performance"] = perf
                if "speedup" in perf:
                    # applied-kernels path returns geomean speedup only (no per-trial cand/ref ms).
                    cand = perf.get("candidate_runtime_ms")
                    ref = perf.get("reference_runtime_ms")
                    if cand is not None and ref is not None:
                        logger.info(
                            f"  PERF: speedup={perf['speedup']:.2f}x "
                            f"cand={cand:.3f}ms ref={ref:.3f}ms"
                        )
                    else:
                        method_tag = perf.get("timing_method", "?")
                        logger.info(f"  PERF: speedup={perf['speedup']:.2f}x ({method_tag})")
                else:
                    logger.info(f"  PERF skipped: {perf.get('skipped')}")
                if performance_gate is not None:
                    perf_gate = _evaluate_performance_gate(perf, performance_gate)
                    record["performance_gate"] = perf_gate
                    if not perf_gate.get("passed", True):
                        classification = _classification_from_perf_gate(perf, perf_gate)
                        record.update(classification)
                        record["final_solution_status"] = "FAILED_TEST"
                        if "speedup" in perf_gate:
                            logger.info(
                                f"  PERF GATE failed: speedup={perf_gate['speedup']:.2f}x "
                                f"< {perf_gate['min_speedup']:.2f}x"
                            )
                        else:
                            logger.info(
                                f"  PERF GATE failed: {perf_gate.get('reason')}"
                            )
                    else:
                        classification = _classification_passed_from_perf_gate(perf_gate)
                        record.update(classification)
                        record["final_solution_status"] = "PASSED"
                        logger.info(
                            f"  PERF GATE passed: >= {perf_gate['min_speedup']:.2f}x"
                        )
            elif performance_gate is not None and stdout_gate:
                gate = classification.get("performance_gate") or {}
                logger.info(
                    f"  PERF GATE passed from test output: "
                    f"speedup={gate.get('speedup'):.2f}x >= {gate.get('min_speedup'):.2f}x"
                )
            else:
                logger.info("  PERF skipped: disabled by experiment.measure_perf_on_pass=false")
            # Iterative stops only after correctness and any enabled perf gate pass.
            if classification["five_category"] == "passed":
                logger.info(f"  PASSED at {loop_label.lower()} {iteration}!")
            if method == "iterative" and classification["five_category"] == "passed":
                stop_reason = "passed"
                break

        # Repeated mode: no stagnation check, no feedback accumulation —
        # each attempt is independent.
        if method == "repeated":
            continue

        # --- Multi-signal stagnation check (iterative only) ---
        stagnation_history.append(make_stagnation_record(iteration, code, classification))
        should_stop, stag_reason = check_stagnation(stagnation_history)
        if should_stop:
            logger.warning(f"  Stagnation detected ({stag_reason}) — stopping early")
            stop_reason = stag_reason
            break

        # --- Error-aware feedback for next iteration ---
        feedback_level = config["experiment"].get("feedback_level", 3)
        feedback_mode = config["experiment"].get("feedback_mode", "aware")
        ncu_data = None
        if feedback_level == 4 and build_ok and not (test_returncode == 0):
            try:
                from .ncu_runner import collect_metrics
                ncu_data = collect_metrics(workdir, task_stem)
            except Exception as e:
                logger.warning(f"  ncu collection failed: {e}")
        feedback_msg = build_feedback(
            classification, build_output, test_output, code,
            level=feedback_level, ncu_data=ncu_data, mode=feedback_mode,
        )
        # Descriptive metric for length-confound check (Risk: see spec §7).
        if records:
            records[-1]["feedback_msg_chars"] = len(feedback_msg) if feedback_msg else 0
        if feedback_msg is None:
            # L0 — no carryover; reset to initial conversation state.
            messages = list(initial_messages) if initial_messages is not None else messages[:2]
        else:
            messages.append({"role": "assistant", "content": raw_response})
            messages.append({"role": "user", "content": feedback_msg})

    # Persist task-level stop reason on the last record for robust resume decisions.
    if records:
        records[-1]["stop_reason"] = stop_reason

    return records, stop_reason


def _annotate_method_fields(record: dict, config: dict, method: str) -> None:
    """Stamp method-level metadata onto a manifest record (Exp 2/3/6/7).

    Mutates record in-place. Adds whichever experiment-axis fields are set
    in config; missing keys are simply omitted (back-compat with old runs).
    """
    exp = config.get("experiment", {})
    record["method"] = method
    if "max_history_rounds" in exp:
        record["max_history_rounds"] = exp["max_history_rounds"]
    if "feedback_level" in exp:
        record["feedback_level"] = exp["feedback_level"]
    if "feedback_mode" in exp:
        record["feedback_mode"] = exp["feedback_mode"]
    if "n_attempts" in exp:
        record["n_attempts"] = exp["n_attempts"]
    if "few_shot_strategy" in exp:
        record["few_shot_strategy"] = exp["few_shot_strategy"]
    if "few_shot_k" in exp:
        record["few_shot_k"] = exp["few_shot_k"]


def _make_record(
    sample_id: str,
    input_meta: dict,
    task_stem: str,
    iteration: int,
    five_category: str = "compile_error",
    category: str | None = None,
    error_type: str | None = None,
    build_ok: bool = False,
    status: str = "failed",
    logic_error_category: str | None = None,
    logic_error_detail: str | None = None,
    performance_gate: dict | None = None,
    code_hash: str | None = None,
) -> dict:
    source_task_id = input_meta.get("task_id")
    record = {
        "sample_id": sample_id,
        "task_id": task_stem,
        "iteration": iteration,
        "status": status,
        "five_category": five_category,
        "category": category or five_category,
        "error_type": error_type,
        "build_ok": build_ok,
        "difficulty": input_meta.get("difficulty", "unknown"),
        "level": input_meta.get("level", "unknown"),
        "paths": {
            "prompt": f"prompts/{task_stem}.txt",
            "input": f"input/{task_stem}.json",
            "testbench": f"testbench/{task_stem}",
            "error_code": f"error_code/{code_hash}.cu" if code_hash else None,
            "build_log_raw": f"build_log_raw/{sample_id}.txt",
        },
    }
    if source_task_id and source_task_id != task_stem:
        record["source_task_id"] = source_task_id
    if logic_error_category:
        record["logic_error_category"] = logic_error_category
    if logic_error_detail:
        record["logic_error_detail"] = logic_error_detail
    if performance_gate is not None:
        record["performance_gate"] = performance_gate
    if status == "passed":
        record["final_solution_status"] = "PASSED"
    else:
        record["final_solution_status"] = (
            "FAILED_BUILD" if not build_ok else "FAILED_TEST"
        )
    return record


# ---------------------------------------------------------------------------
# Export: generate the same directory structure as CUDA-Debugger
# ---------------------------------------------------------------------------


def export_results(
    all_records: list[dict],
    dataset_dir: Path,
    output_dir: Path,
    config: dict | None = None,
):
    """
    After all tasks complete, generate:
      - index/manifest.jsonl
      - index/by_task.json
      - summary.json
      - category_*/index/manifest.jsonl + symlinked error_code & build_log_raw
    """
    logger.info("Exporting results...")

    # --- manifest.jsonl ---
    idx_dir = output_dir / "index"
    idx_dir.mkdir(parents=True, exist_ok=True)
    with open(idx_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --- by_task.json ---
    by_task: dict[str, list[str]] = {}
    for r in all_records:
        by_task.setdefault(r["task_id"], []).append(r["sample_id"])
    (idx_dir / "by_task.json").write_text(
        json.dumps(by_task, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # --- Copy prompts and input from dataset (symlinks) ---
    for subdir in ("prompts", "input", "testbench"):
        dst = output_dir / subdir
        src = dataset_dir / subdir
        if not dst.exists() and src.exists():
            os.symlink(src.resolve(), dst)

    # --- Category subdirectories ---
    categories = ["compile_error", "memory_crash", "timeout", "logic_error", "passed"]
    cat_stats = {c: 0 for c in categories}

    for cat in categories:
        cat_dir = output_dir / f"category_{cat}"
        cat_idx = cat_dir / "index"
        cat_idx.mkdir(parents=True, exist_ok=True)

        cat_records = [r for r in all_records if r["five_category"] == cat]
        cat_stats[cat] = len(cat_records)

        with open(cat_idx / "manifest.jsonl", "w", encoding="utf-8") as f:
            for r in cat_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # by_task for this category
        cat_by_task: dict[str, list[str]] = {}
        for r in cat_records:
            cat_by_task.setdefault(r["task_id"], []).append(r["sample_id"])
        (cat_idx / "by_task.json").write_text(
            json.dumps(cat_by_task, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Symlink error_code and build_log_raw into category dir
        for sub in ("error_code", "build_log_raw"):
            link = cat_dir / sub
            target = output_dir / sub
            if not link.exists() and target.exists():
                os.symlink(target.resolve(), link)

    # --- summary.json ---
    total = len(all_records)
    passed = sum(1 for r in all_records if r["status"] == "passed")
    failed = sum(1 for r in all_records if r["status"] == "failed")
    unique_codes = len(set(
        r["paths"]["error_code"] for r in all_records if r["paths"].get("error_code")
    ))
    # Count unique tasks
    task_ids = set(r["task_id"] for r in all_records)
    # Per-task pass rate
    tasks_passed = set()
    for r in all_records:
        if r["status"] == "passed":
            tasks_passed.add(r["task_id"])

    # --- Compute per-task and aggregate metrics via metrics.py ---
    max_iters = (config or {}).get("experiment", {}).get("max_iterations", 5)
    task_metrics_list = []
    per_task_results = {}
    for task_id in task_ids:
        task_records = sorted(
            [r for r in all_records if r["task_id"] == task_id],
            key=lambda r: r["iteration"],
        )
        tm = compute_task_metrics(task_id, task_records)
        task_metrics_list.append(tm)
        per_task_results[task_id] = {
            "status": tm.status,
            "iterations_used": tm.iterations_used,
            "trajectory": tm.trajectory,
            "trajectory_8cat": tm.trajectory_8cat,
            "pass_at_1": tm.pass_at_1,
            "stagnation": tm.stagnation,
            "oscillation": tm.oscillation,
            "progression": tm.progression,
            "stop_reason": tm.stop_reason,
            "final_category": tm.trajectory[-1] if tm.trajectory else None,
            "final_category_8cat": tm.trajectory_8cat[-1] if tm.trajectory_8cat else None,
        }

    agg = compute_aggregate_metrics(task_metrics_list, max_iterations=max_iters)
    n_tasks = agg.total_tasks

    # Preserve performance fields from existing summary.json if present.
    # refill_model_perf_gaps.py writes performance into per_task_results[tid].
    # Without this preservation, finalize wipes out refill data on next resume.
    existing_summary_path = output_dir / "summary.json"
    if existing_summary_path.exists():
        try:
            old_summary = json.loads(existing_summary_path.read_text(encoding="utf-8"))
            old_ptr = old_summary.get("per_task_results", {}) or {}
            preserved_perf = 0
            for tid, info in per_task_results.items():
                old_info = old_ptr.get(tid) or {}
                old_perf = old_info.get("performance")
                # Only preserve if old has perf and new build didn't compute one.
                # (Currently new build never writes performance — it only comes from
                # _measure_iter_perf below for newly-passed iters, or from refill.)
                if old_perf is not None and "performance" not in info:
                    info["performance"] = old_perf
                    preserved_perf += 1
            if preserved_perf:
                logger.info(f"preserved performance field for {preserved_perf} tasks from old summary.json")
        except Exception as e:
            logger.warning(f"failed to preserve old performance fields: {e}")

    summary = {
        "experiment_name": (config or {}).get("experiment", {}).get("name", "cuda_debugger"),
        "model": (config or {}).get("api", {}).get("model", "unknown"),
        "timestamp": datetime.now().isoformat(),
        "total_tasks": n_tasks,
        "tasks_solved": len(tasks_passed),
        "solve_rate": f"{len(tasks_passed) / n_tasks * 100:.1f}%" if n_tasks else "0%",
        "total_iteration_records": total,
        "failed_iterations": failed,
        "passed_iterations": passed,
        "unique_error_code_files": unique_codes,
        "five_category_stats": cat_stats,
        **metrics_to_dict(agg),
        "per_task_results": per_task_results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info(f"Export complete. Summary: {json.dumps({k: v for k, v in summary.items() if k != 'per_task_results'}, indent=2)}")

    # Auto-update memory system
    try:
        update_memory(output_dir, config, all_records)
    except Exception as e:
        logger.warning(f"Memory update failed (non-fatal): {e}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _task_id_to_stem(task_id: str) -> str:
    """Convert 'CUDA/15' -> 'CUDA_15', 'cublas_level3/7' -> 'cublas_level3_7'."""
    return task_id.replace("/", "_")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="CUDA-Debugger Experiment Pipeline")
    parser.add_argument(
        "-c", "--config",
        default="configs/default.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        help="Override: specific task stems to run (e.g. CUDA_15 CUDA_85)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print task list without running",
    )
    parser.add_argument(
        "--two-phase",
        action="store_true",
        help="Two-phase protocol: pass@1 scan all tasks, then debug@k only on failures",
    )
    parser.add_argument(
        "--resume",
        metavar="RESULT_DIR",
        help="Resume from an existing result directory (skip already-completed tasks)",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, sequential). Requires multiple GPUs.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU IDs to use (e.g., '0,2,5'). Auto-detected if not set.",
    )
    parser.add_argument(
        "--mode",
        choices=["generation", "debug"],
        default="generation",
        help="generation: LLM writes from scratch (legacy). debug: LLM starts from a broken implementation (v2).",
    )
    parser.add_argument(
        "--v2-task-list",
        type=str,
        default=None,
        help="Path to v2_task_list.json (required for --mode debug). Provides broken_start per task.",
    )
    args = parser.parse_args()

    # Load v2 broken-start map if in debug mode
    v2_broken_starts: dict[str, dict] = {}
    if args.mode == "debug":
        v2_path = Path(args.v2_task_list) if args.v2_task_list else (
            Path(__file__).resolve().parent.parent / "benchmark" / "task_list.json"
        )
        if not v2_path.exists():
            logger.error(f"--mode debug requires v2_task_list.json (looked at {v2_path})")
            sys.exit(1)
        v2_broken_starts = broken_start_map(v2_path)
        logger.info(f"v2 debug mode: loaded {len(v2_broken_starts)} broken_start entries")

    config = load_config(args.config)

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, config["logging"]["level"]),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Resolve paths
    project_root = Path(__file__).resolve().parent.parent
    dataset_dir = (project_root / config["experiment"]["dataset_dir"]).resolve()

    if args.resume:
        output_dir = Path(args.resume).resolve()
        if not output_dir.exists():
            logger.error(f"Resume dir not found: {output_dir}")
            sys.exit(1)
        logger.info(f"Resuming from: {output_dir}")
    else:
        output_dir = (
            project_root
            / config["experiment"]["output_dir"]
            / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.config, output_dir / "config.yaml")

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Dataset:  {dataset_dir}")
    logger.info(f"Output:   {output_dir}")

    # Discover tasks
    if args.tasks:
        task_stems = args.tasks
    elif config["experiment"]["tasks"]:
        task_stems = config["experiment"]["tasks"]
    else:
        task_stems = discover_task_stems(dataset_dir, args.v2_task_list)

    error_category_filter = (
        config.get("experiment", {}).get("error_category")
        or config.get("experiment", {}).get("task_error_category")
    )
    if error_category_filter:
        if isinstance(error_category_filter, str):
            wanted_categories = {error_category_filter}
        else:
            wanted_categories = set(error_category_filter)
        task_stems = [
            stem for stem in task_stems
            if task_error_category(stem, dataset_dir) in wanted_categories
        ]

    # Pre-flight checks (GPU compat, API key, dataset, disk)
    from .preflight import run_preflight
    preflight_report, task_stems = run_preflight(config, task_stems, dataset_dir)
    preflight_report.print_report()
    if not preflight_report.ok:
        logger.error("Pre-flight check failed. Fix errors above before running.")
        sys.exit(1)

    logger.info(f"Tasks ({len(task_stems)}): {task_stems}")

    if args.dry_run:
        print(f"Would run {len(task_stems)} tasks:")
        for t in task_stems:
            print(f"  - {t}")
        if preflight_report.skipped_tasks:
            print(f"\nSkipped {len(preflight_report.skipped_tasks)} GPU-incompatible tasks:")
            for stem, reason in preflight_report.skipped_tasks:
                print(f"  - {stem}: {reason}")
        return

    # Init OpenAI client
    client = create_client(config)

    # Load already-completed records if resuming
    all_records = []
    done_task_stems: set[str] = set()
    existing_manifest = output_dir / "index" / "manifest.jsonl"
    if existing_manifest.exists():
        with open(existing_manifest, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    all_records.append(r)
        # A task is "done" if its final iteration reached passed OR it consumed all iters.
        # Derive task IDs from manifest records so resume also works when by_task.json
        # does not exist yet (e.g. interrupted run before final export).
        early_stop_reasons = {
            "duplicate_code",
            "code_cycle",
            "category_oscillation",
            "no_progress",
        }
        for task_id in {r["task_id"] for r in all_records}:
            task_records = [r for r in all_records if r["task_id"] == task_id]
            task_records.sort(key=lambda r: r["iteration"])
            last_stop_reason = task_records[-1].get("stop_reason") if task_records else None
            if any(r["status"] == "passed" for r in task_records):
                done_task_stems.add(_task_id_to_stem(task_id))
            elif len(task_records) >= config["experiment"]["max_iterations"]:
                done_task_stems.add(_task_id_to_stem(task_id))
            elif last_stop_reason in early_stop_reasons:
                done_task_stems.add(_task_id_to_stem(task_id))
        if done_task_stems:
            logger.info(f"Skipping {len(done_task_stems)} already-finished tasks: {sorted(done_task_stems)}")

    # Run tasks
    remaining = [t for t in task_stems if t not in done_task_stems]
    max_iters = config["experiment"]["max_iterations"]

    # --- Parallel execution ---
    if args.n_workers > 1:
        from .parallel_runner import run_parallel, run_parallel_two_phase

        gpu_ids = None
        if args.gpus:
            gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]

        if args.two_phase and max_iters > 1:
            new_records = run_parallel_two_phase(
                remaining, dataset_dir, output_dir, config,
                n_workers=args.n_workers, gpu_ids=gpu_ids,
            )
        else:
            new_records = run_parallel(
                remaining, dataset_dir, output_dir, config,
                n_workers=args.n_workers, gpu_ids=gpu_ids,
            )

        all_records.extend(new_records)
        _write_incremental_manifest(all_records, output_dir)
        export_results(all_records, dataset_dir, output_dir, config)
        logger.info("All done!")
        return

    # --- Sequential execution ---
    if args.two_phase and max_iters > 1:
        # --- Two-phase protocol ---
        # Phase 1: pass@1 scan (all tasks, 1 iteration)
        logger.info(f"\n{'='*60}")
        logger.info(f"PHASE 1: pass@1 scan ({len(remaining)} tasks, 1 iteration each)")
        logger.info(f"{'='*60}")

        phase1_config = dict(config)
        phase1_config["experiment"] = {**config["experiment"], "max_iterations": 1}

        phase1_failed = []
        for i, task_stem in enumerate(remaining):
            logger.info(f"  [{i+1}/{len(remaining)}] {task_stem}")
            try:
                records, stop_reason = run_task(task_stem, dataset_dir, output_dir, phase1_config, client, broken_start=v2_broken_starts.get(task_stem))
                all_records.extend(records)
                if records and records[-1]["status"] != "passed":
                    phase1_failed.append(task_stem)
            except Exception as e:
                logger.error(f"Task {task_stem} failed: {e}", exc_info=True)
                phase1_failed.append(task_stem)
                continue

        _write_incremental_manifest(all_records, output_dir)
        pass1 = len(remaining) - len(phase1_failed)
        logger.info(f"\nPhase 1 complete: {pass1}/{len(remaining)} passed at iter 1")

        # Phase 2: debug@k (failed tasks only, remaining iterations)
        if phase1_failed and max_iters > 1:
            logger.info(f"\n{'='*60}")
            logger.info(f"PHASE 2: debug@{max_iters} ({len(phase1_failed)} failed tasks)")
            logger.info(f"{'='*60}")

            for i, task_stem in enumerate(phase1_failed):
                logger.info(f"  [{i+1}/{len(phase1_failed)}] {task_stem}")
                try:
                    records, stop_reason = run_task(task_stem, dataset_dir, output_dir, config, client, broken_start=v2_broken_starts.get(task_stem))
                    all_records.extend(records)
                except Exception as e:
                    logger.error(f"Task {task_stem} failed: {e}", exc_info=True)
                    continue
                _write_incremental_manifest(all_records, output_dir)

    else:
        # --- Standard single-phase ---
        logger.info(f"Tasks to run: {len(remaining)}/{len(task_stems)}")
        for i, task_stem in enumerate(remaining):
            logger.info(f"\n{'='*60}")
            logger.info(f"Task {i+1}/{len(task_stems)}: {task_stem}")
            logger.info(f"{'='*60}")

            try:
                records, stop_reason = run_task(task_stem, dataset_dir, output_dir, config, client, broken_start=v2_broken_starts.get(task_stem))
                all_records.extend(records)
            except Exception as e:
                logger.error(f"Task {task_stem} failed with exception: {e}", exc_info=True)
                continue
            _write_incremental_manifest(all_records, output_dir)

    # Final export
    export_results(all_records, dataset_dir, output_dir, config)
    logger.info("All done!")


def _write_incremental_manifest(all_records: list[dict], output_dir: Path):
    """Write manifest/index incrementally for crash-safe resume."""
    idx_dir = output_dir / "index"
    idx_dir.mkdir(parents=True, exist_ok=True)
    with open(idx_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_task: dict[str, list[str]] = {}
    for r in all_records:
        by_task.setdefault(r["task_id"], []).append(r["sample_id"])
    (idx_dir / "by_task.json").write_text(
        json.dumps(by_task, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
