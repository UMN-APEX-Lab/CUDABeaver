"""Iteration records and artifact persistence for Cudeubgger."""

from __future__ import annotations

import hashlib
from pathlib import Path


def _persist_iter_artifacts(
    output_dir: Path, iter_label: str, new_code: str,
    build_ok: bool, build_output: str, test_output: str | None,
    solution_file: str = "solution.cu",
) -> str:
    """Write candidate code and raw build/test logs for an iteration."""
    code_hash = hashlib.sha256(new_code.encode("utf-8")).hexdigest()[:16]
    suffix = Path(solution_file).suffix or ".cu"
    code_dir = output_dir / "error_code"
    code_dir.mkdir(parents=True, exist_ok=True)
    (code_dir / f"{code_hash}{suffix}").write_text(new_code, encoding="utf-8")
    log_dir = output_dir / "build_log_raw"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_content = f"BUILD_OK={build_ok}\n\n[BUILD OUTPUT]\n{build_output}"
    if test_output is not None:
        log_content += f"\n\n[TEST OUTPUT]\n{test_output}"
    (log_dir / f"{iter_label}.txt").write_text(log_content, encoding="utf-8")
    return code_hash


def _make_iter_record(
    task_stem: str,
    input_meta: dict,
    iteration: int,
    five_category: str = "compile_error",
    category: str | None = None,
    error_type: str | None = None,
    build_ok: bool = False,
    status: str = "failed",
    logic_error_category: str | None = None,
    logic_error_detail: str | None = None,
    repair_mode: str | None = None,
    code_hash: str | None = None,
    artifact_suffix: str = ".cu",
    rationale: str = "",
    stop_reason: str | None = None,
    located: dict | None = None,
) -> dict:
    sample_id = f"{task_stem}__iter_{iteration:04d}"
    task_id = input_meta.get("task_id", task_stem.replace("_", "/", 1))
    record = {
        "sample_id": sample_id,
        "task_id": task_id,
        "iteration": iteration,
        "status": status,
        "five_category": five_category,
        "category": category or five_category,
        "error_type": error_type,
        "build_ok": build_ok,
        "repair_mode": repair_mode,
        "difficulty": input_meta.get("difficulty", "unknown"),
        "level": input_meta.get("level", "unknown"),
        "paths": {
            "prompt": f"prompts/{task_stem}.txt",
            "input": f"input/{task_stem}.json",
            "testbench": f"testbench/{task_stem}",
            "error_code": f"error_code/{code_hash}{artifact_suffix}" if code_hash else None,
            "build_log_raw": f"build_log_raw/{sample_id}.txt",
            "tool_trace": f"tool_trace/{task_stem}.jsonl",
        },
        "agent_rationale": rationale[:200] if rationale else None,
    }
    if logic_error_category:
        record["logic_error_category"] = logic_error_category
    if logic_error_detail:
        record["logic_error_detail"] = logic_error_detail
    if located:
        record["located"] = located
    if stop_reason:
        record["stop_reason"] = stop_reason
    if status == "passed":
        record["final_solution_status"] = "PASSED"
    else:
        record["final_solution_status"] = "FAILED_BUILD" if not build_ok else "FAILED_TEST"
    return record
