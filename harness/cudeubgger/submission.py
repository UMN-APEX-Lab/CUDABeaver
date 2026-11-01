"""Patch submission evaluation for Cudeubgger."""

from __future__ import annotations

import logging
from pathlib import Path

from ..agent_tools import ToolEnv
from . import deps
from .common import (
    _LoopState,
    _append_action_result_message,
    _hard_stop_on_stagnation,
    _max_patch_submissions,
    _performance_feedback_text,
    _performance_gpu_id,
    _solution_artifact_suffix,
    _truncate_text,
)
from .prompts import BUILD_TEST_FEEDBACK_TEMPLATE
from .records import _make_iter_record, _persist_iter_artifacts

logger = logging.getLogger("cuda_debugger_exp")


def _handle_submission(
    tc: dict, result,
    state: _LoopState, env: ToolEnv, messages: list[dict],
    task_stem: str, dataset_dir: Path, input_meta: dict,
    config: dict, output_dir: Path,
    client, prompt_text: str,
    json_action: bool = False,
) -> str | None:
    """Build, test, classify, and feed back one write_solution submission."""
    if bool(env.current_code.strip()) and not state.located_this_iteration:
        logger.warning(
            "  [agent] backstop: write_solution called in LOCATE phase "
            "(tool-list filter bypassed) - rejected"
        )
        rejection = (
            "[write_solution rejected] You are still in the LOCATE "
            "phase. Call `locate_bug` first - declare the exact file, "
            "1-indexed line, what is wrong, and the evidence. "
            "write_solution unlocks after."
        )
        if json_action:
            _append_action_result_message(
                messages, tc["function"]["name"], rejection, ok=False
            )
        else:
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": rejection,
            })
        return None

    workdir = Path(config["build"]["workdir"])
    build_timeout = int(config["build"]["timeout"])
    test_timeout = int(config["build"]["test_timeout"])
    max_patch_submissions = _max_patch_submissions(config)
    submitted_location = (
        dict(state.pending_location) if isinstance(state.pending_location, dict)
        else state.pending_location
    )

    new_code = env.current_code
    state.iteration += 1
    iter_label = f"{task_stem}__iter_{state.iteration:04d}"
    logger.info(
        f"  [agent patch submit {state.iteration}/{max_patch_submissions}] "
        f"{iter_label} ({len(new_code)} chars)"
    )

    build_ok, build_output = deps.compile_code(
        new_code, task_stem, dataset_dir, input_meta, workdir, build_timeout,
    )
    test_returncode, test_output = None, None
    if build_ok:
        test_returncode, test_output = deps.run_test(
            task_stem, input_meta, workdir, test_timeout,
        )
    classification = deps.classify_result(
        build_ok, build_output, test_output, test_returncode,
    )
    perf = None
    perf_gate = None
    if classification.get("five_category") == "passed":
        perf, perf_gate, classification, test_output = _evaluate_performance_gate(
            task_stem,
            input_meta,
            new_code,
            test_output,
            config,
            dataset_dir,
            classification,
        )

    sanitizer_summary = _maybe_run_sanitizers(
        build_ok,
        test_returncode,
        config,
        task_stem,
        input_meta,
        workdir,
    )
    patch_scope = deps.analyze_patch_scope(
        config,
        state.broken_code,
        new_code,
        submitted_location,
    )

    code_hash = _persist_iter_artifacts(
        output_dir, iter_label, new_code, build_ok, build_output, test_output,
        solution_file=input_meta.get("solution_file", "solution.cu"),
    )
    _record_submission(
        state,
        task_stem,
        input_meta,
        state.iteration,
        code_hash,
        result.meta.get("rationale", ""),
        submitted_location,
        classification,
        build_output,
        test_output,
        perf,
        perf_gate,
        sanitizer_summary,
        patch_scope,
    )

    stagnation_reason = _update_stagnation(state, new_code, classification, config)
    feedback_body = _build_submission_feedback(
        client,
        config,
        task_stem,
        iter_label,
        state,
        prompt_text,
        new_code,
        submitted_location,
        classification,
        build_output,
        test_output,
        sanitizer_summary,
        patch_scope,
        stagnation_reason,
    )
    tool_reply = BUILD_TEST_FEEDBACK_TEMPLATE.format(
        build_ok=build_ok,
        category=classification.get("category", "?"),
        feedback_body=feedback_body or "(no feedback)",
    )
    if json_action:
        _append_action_result_message(
            messages, tc["function"]["name"], tool_reply, ok=True
        )
    else:
        messages.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": tool_reply,
        })

    if classification.get("five_category") == "passed":
        logger.info(f"  [agent] PASSED at iter {state.iteration}")
        return "passed"
    if stagnation_reason and _hard_stop_on_stagnation(config):
        return stagnation_reason
    return None


def _evaluate_performance_gate(
    task_stem: str,
    input_meta: dict,
    new_code: str,
    test_output: str | None,
    config: dict,
    dataset_dir: Path,
    classification: dict,
) -> tuple[dict | None, dict | None, dict, str | None]:
    solution_file = input_meta.get("solution_file", "solution.cu")
    perf = deps.measure_iter_perf(
        input_meta.get("task_id", task_stem),
        new_code,
        solution_file,
        test_output=test_output,
        gpu_id=_performance_gpu_id(config),
        dataset_dir=dataset_dir,
    )
    perf_gate = deps.evaluate_performance_gate(
        perf,
        (config.get("agent", {}) or {}).get("performance_gate"),
    )
    if perf_gate.get("enabled"):
        if "speedup" in perf:
            logger.info(
                f"  [agent] PERF: speedup={perf['speedup']:.2f}x "
                f"gate>={perf_gate['min_speedup']:.2f}x"
            )
        else:
            logger.info(
                f"  [agent] PERF skipped: {perf.get('skipped')} "
                f"(gate reason={perf_gate.get('reason')})"
            )
    if not perf_gate.get("passed", True):
        perf_feedback = _performance_feedback_text(perf, perf_gate)
        classification = {
            "category": "perf_below_threshold",
            "five_category": "logic_error",
            "error_type": "perf_gate",
            "build_ok": True,
            "status": "failed",
            "logic_error_category": "perf_below_threshold",
            "logic_error_detail": perf_feedback,
        }
        test_output = (test_output or "") + "\n\n[PERFORMANCE GATE]\n" + perf_feedback
    return perf, perf_gate, classification, test_output


def _maybe_run_sanitizers(
    build_ok: bool,
    test_returncode: int | None,
    config: dict,
    task_stem: str,
    input_meta: dict,
    workdir: Path,
) -> dict | None:
    if (
        not build_ok
        or test_returncode in (None, 0)
        or not deps.section_enabled(config, "sanitizers", default=False)
    ):
        return None
    sanitizer_cfg = (config.get("debug_tools") or {}).get("sanitizers")
    try:
        return deps.run_sanitizers(task_stem, input_meta, workdir, sanitizer_cfg)
    except Exception as e:
        logger.warning(f"  [agent] auto-sanitizer failed: {e}")
        return None


def _record_submission(
    state: _LoopState,
    task_stem: str,
    input_meta: dict,
    iteration: int,
    code_hash: str,
    rationale: str,
    submitted_location,
    classification: dict,
    build_output: str,
    test_output: str | None,
    perf: dict | None,
    perf_gate: dict | None,
    sanitizer_summary: dict | None,
    patch_scope: dict | None,
) -> None:
    record = _make_iter_record(
        task_stem, input_meta, iteration,
        code_hash=code_hash,
        rationale=rationale,
        located=submitted_location,
        artifact_suffix=_solution_artifact_suffix(input_meta),
        **classification,
    )
    record["method"] = "agent"
    if perf is not None:
        record["performance"] = perf
    if perf_gate is not None:
        record["performance_gate"] = perf_gate
    if state.patch_actions_this_iteration:
        record["patch_actions"] = list(state.patch_actions_this_iteration)
    compact_san = deps.compact_sanitizer_summary(sanitizer_summary)
    if compact_san:
        record["sanitizer_summary"] = compact_san
    if patch_scope:
        record["patch_scope"] = patch_scope
    state.records.append(record)
    state.attempt_history.append({
        "iteration": iteration,
        "rationale": rationale,
        "located": submitted_location,
        # The verifier's single fix direction for this attempt (gap 6).
        "fix_direction": state.current_fix_direction,
        "patch_actions": list(state.patch_actions_this_iteration),
        "classification": classification,
        "build_output_excerpt": _truncate_text(build_output, 2000),
        "test_output_excerpt": _truncate_text(test_output, 2000),
        "performance": perf,
        "performance_gate": perf_gate,
    })

    state.located_this_iteration = False
    state.pending_location = None
    state.applied_edit_this_iteration = False
    state.patch_actions_this_iteration = []
    state.non_submission_rounds = 0


def _update_stagnation(
    state: _LoopState,
    new_code: str,
    classification: dict,
    config: dict,
) -> str | None:
    if classification.get("five_category") == "passed":
        return None
    state.stagnation_history.append(
        deps.make_stagnation_record(state.iteration, new_code, classification)
    )
    should_stop, reason = deps.check_stagnation(state.stagnation_history)
    if not should_stop:
        return None
    if _hard_stop_on_stagnation(config):
        logger.warning(f"  [agent] stagnation hard stop: {reason}")
    else:
        logger.warning(
            f"  [agent] stagnation signal: {reason} "
            "(telemetry only; continuing until patch budget)"
        )
    return reason


def _build_submission_feedback(
    client,
    config: dict,
    task_stem: str,
    iter_label: str,
    state: _LoopState,
    prompt_text: str,
    new_code: str,
    submitted_location,
    classification: dict,
    build_output: str,
    test_output: str | None,
    sanitizer_summary: dict | None,
    patch_scope: dict | None,
    stagnation_reason: str | None,
) -> str:
    feedback_body = deps.build_feedback(
        classification, build_output, test_output, new_code,
        level=3, mode="aware",
        sanitizer_summary=sanitizer_summary,
        repair_mode=classification.get("repair_mode"),
        minimal_patch=True,
    )
    verifier_result = None
    if (
        classification.get("status") != "passed"
        and deps.debug_tools_enabled(config)
        and deps.section_enabled(config, "patch_verifier", default=False)
    ):
        try:
            verifier_result = deps.verify_failed_patch(
                client=client,
                config=config,
                task_stem=task_stem,
                sample_id=iter_label,
                iteration=state.iteration,
                prompt_text=prompt_text,
                broken_code=state.broken_code,
                candidate_code=new_code,
                localization=submitted_location,
                classification=classification,
                build_output=build_output,
                test_output=test_output,
            )
        except Exception as e:
            logger.warning(f"  [agent] patch verifier failed: {e}")

    feedback_body = deps.append_debug_feedback(
        feedback_body, scope=patch_scope, verifier=verifier_result,
    )
    if stagnation_reason and not _hard_stop_on_stagnation(config):
        feedback_body = (
            f"{feedback_body or ''}\n\n"
            "[STAGNATION SIGNAL]\n"
            f"reason={stagnation_reason}. This is not a stop condition. "
            "Use the latest build/test/performance result plus code and tool "
            "evidence to form a revised localization before the next patch. "
            "Do not resubmit the same change."
        )
    return feedback_body
