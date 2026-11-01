"""Optional debug-assist helpers for the v2 CUDA repair loop.

All helpers are inert unless ``debug_tools.enabled`` is true. The main
experiment harness keeps the existing compile/test/classify/feedback path as
the default and calls these helpers only behind that feature flag.
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from .task_registry import resolve_task_paths
from typing import Any

from .trace_io import compact_sanitizer_summary, sanitizer_output_artifacts


def debug_tools_enabled(config: dict) -> bool:
    return bool((config.get("debug_tools") or {}).get("enabled", False))


def section_enabled(config: dict, section: str, default: bool = True) -> bool:
    if not debug_tools_enabled(config):
        return False
    sec = (config.get("debug_tools") or {}).get(section, {})
    if sec is None:
        sec = {}
    return bool(sec.get("enabled", default))


def debug_tools_manifest(config: dict) -> dict | None:
    if not debug_tools_enabled(config):
        return None
    cfg = config.get("debug_tools") or {}
    return {
        "enabled": True,
        "precheck": section_enabled(config, "precheck"),
        "localization": section_enabled(config, "localization"),
        "fact_selection": section_enabled(config, "fact_selection"),
        "patch_scope_guard": section_enabled(config, "patch_scope_guard", default=False),
        "patch_verifier": section_enabled(config, "patch_verifier", default=False),
        "patch_verifier_mode": (cfg.get("patch_verifier") or {}).get("mode", "on_failed_attempt"),
    }


def run_precheck(
    config: dict,
    task_stem: str,
    dataset_dir: Path,
    input_meta: dict,
    workdir: Path,
    broken_code: str,
    sanitizer_config: dict | None,
    trace_writer: Any | None = None,
) -> dict | None:
    """Compile/test the original broken code to obtain grounded failure facts."""
    if not section_enabled(config, "precheck"):
        return None

    from .classifier import classify_result
    from .compiler import compile_code, run_sanitizers, run_test

    pre_cfg = (config.get("debug_tools") or {}).get("precheck") or {}
    run_sanitizer = bool(pre_cfg.get("run_sanitizer", False))
    timeout = int(config.get("build", {}).get("timeout", 60))
    test_timeout = int(config.get("build", {}).get("test_timeout", 60))
    result: dict[str, Any] = {"enabled": True, "status": "started"}
    build_output = ""
    test_output = None
    test_returncode = None
    sanitizer_summary = None
    try:
        build_ok, build_output = compile_code(
            broken_code, task_stem, dataset_dir, input_meta, workdir, timeout
        )
        if build_ok:
            test_returncode, test_output = run_test(
                task_stem, input_meta, workdir, test_timeout
            )
            if (
                test_returncode != 0
                and run_sanitizer
                and (sanitizer_config or {}).get("enabled", False)
            ):
                sanitizer_summary = run_sanitizers(
                    task_stem, input_meta, workdir, sanitizer_config
                )
        classification = classify_result(
            build_ok,
            build_output,
            test_output,
            test_returncode,
            sanitizer_summary=sanitizer_summary,
        )
        result.update(
            {
                "status": "ok",
                "build_ok": build_ok,
                "test_returncode": test_returncode,
                "classification": classification,
                "build_output_excerpt": _truncate(build_output or "", 1600),
                "test_output_excerpt": _truncate(test_output or "", 1600),
                "sanitizer": _compact_sanitizer(sanitizer_summary),
            }
        )
    except Exception as exc:  # pragma: no cover - defensive path
        result.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})

    trace_ref = _write_precheck_trace(
        trace_writer=trace_writer,
        task_stem=task_stem,
        input_meta=input_meta,
        broken_code=broken_code,
        result=result,
        build_output=build_output,
        test_output=test_output,
        sanitizer_summary=sanitizer_summary,
    )
    if trace_ref:
        result["trace"] = trace_ref
    return result


def localize_root_cause(
    client: Any,
    config: dict,
    task_stem: str,
    dataset_dir: Path,
    input_meta: dict,
    prompt_text: str,
    broken_code: str,
    precheck: dict | None,
    trace_writer: Any | None = None,
) -> dict | None:
    """Ask the model for a structured localization report."""
    if not section_enabled(config, "localization"):
        return None

    from .llm_client import call_llm

    loc_cfg = (config.get("debug_tools") or {}).get("localization") or {}
    testbench = ""
    if bool(loc_cfg.get("include_testbench", False)):
        testbench = collect_testbench_context(dataset_dir, task_stem)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a CUDA debugging localization assistant. "
                "Return ONLY compact JSON. Do not return fixed code."
            ),
        },
        {
            "role": "user",
            "content": _localization_prompt(
                input_meta=input_meta,
                prompt_text=prompt_text,
                broken_code=broken_code,
                precheck=precheck,
                testbench=testbench,
            ),
        },
    ]
    try:
        raw = call_llm(client, config, messages)
    except Exception as exc:  # pragma: no cover - network/API defensive path
        result = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "parsed": None,
        }
        trace_ref = _write_llm_trace(
            trace_writer=trace_writer,
            task_stem=task_stem,
            stage_parts="localization",
            stage="localization",
            task_id=input_meta.get("task_id", task_stem),
            messages=messages,
            response_text=None,
            parsed=result,
            summary={"status": result["status"], "error": result["error"]},
        )
        if trace_ref:
            result["trace"] = trace_ref
        return result

    parsed = parse_localization_response(raw)
    parsed["status"] = "ok" if parsed.get("parsed") else "parse_error"
    trace_ref = _write_llm_trace(
        trace_writer=trace_writer,
        task_stem=task_stem,
        stage_parts="localization",
        stage="localization",
        task_id=input_meta.get("task_id", task_stem),
        messages=messages,
        response_text=raw,
        parsed=parsed,
        summary={
            "status": parsed.get("status"),
            "parse_error": parsed.get("parse_error"),
            "localization_summary": localization_summary(parsed),
        },
    )
    if trace_ref:
        parsed["trace"] = trace_ref
    return parsed


def parse_localization_response(text: str) -> dict:
    """Parse localization JSON while safely preserving raw output on failure."""
    raw = text or ""
    body = _strip_code_fence(raw.strip())
    if not body.startswith("{"):
        match = re.search(r"\{.*\}", body, flags=re.DOTALL)
        if match:
            body = match.group(0)
    try:
        payload = json.loads(body)
    except Exception as exc:
        return {
            "parsed": None,
            "raw_response": _truncate(raw, 2000),
            "parse_error": f"{type(exc).__name__}: {exc}",
        }
    if not isinstance(payload, dict):
        return {
            "parsed": None,
            "raw_response": _truncate(raw, 2000),
            "parse_error": f"Expected JSON object, got {type(payload).__name__}",
        }
    return {
        "parsed": normalize_localization_payload(payload),
        "raw_response": _truncate(raw, 2000),
        "parse_error": None,
    }


def normalize_localization_payload(payload: dict) -> dict:
    suspicious = []
    raw_lines = payload.get("suspicious_lines") or payload.get("bug_locations") or []
    if isinstance(raw_lines, dict):
        raw_lines = [raw_lines]
    for item in raw_lines:
        if isinstance(item, int):
            suspicious.append({"line": item, "reason": ""})
        elif isinstance(item, str):
            line = _first_int(item)
            suspicious.append({"line": line, "reason": item[:240]})
        elif isinstance(item, dict):
            line = item.get("line") or item.get("line_number") or item.get("start_line")
            suspicious.append(
                {
                    "line": _coerce_int(line),
                    "reason": str(item.get("reason") or item.get("why") or "")[:240],
                }
            )
    suspicious = [s for s in suspicious if s["line"] is not None]
    return {
        "suspicious_lines": suspicious[:12],
        "root_cause_hypotheses": _list_of_strings(
            payload.get("root_cause_hypotheses")
            or payload.get("hypotheses")
            or payload.get("root_cause")
        )[:6],
        "repair_constraints": _list_of_strings(
            payload.get("repair_constraints")
            or payload.get("constraints")
            or payload.get("fix_constraints")
        )[:6],
        "facts_to_keep": _list_of_strings(
            payload.get("facts_to_keep")
            or payload.get("facts")
            or payload.get("useful_facts")
        )[:12],
        "confidence": payload.get("confidence"),
    }


def select_facts(localization: dict | None, max_facts: int = 6) -> list[str]:
    parsed = (localization or {}).get("parsed") or {}
    facts: list[str] = []
    for loc in parsed.get("suspicious_lines") or []:
        line = loc.get("line")
        reason = loc.get("reason") or "candidate root-cause line"
        facts.append(f"line {line}: {reason}")
    facts.extend(parsed.get("root_cause_hypotheses") or [])
    facts.extend(parsed.get("repair_constraints") or [])
    facts.extend(parsed.get("facts_to_keep") or [])

    deduped = []
    seen = set()
    for fact in facts:
        key = fact.strip().lower()
        if key and key not in seen:
            deduped.append(fact.strip())
            seen.add(key)
    return deduped[:max_facts]


def format_localization_report(config: dict, localization: dict | None) -> str:
    if not debug_tools_enabled(config) or not localization or not localization.get("parsed"):
        return ""
    fact_cfg = (config.get("debug_tools") or {}).get("fact_selection") or {}
    use_selection = section_enabled(config, "fact_selection")
    max_facts = int(fact_cfg.get("max_facts", 6)) if use_selection else 50
    facts = select_facts(localization, max_facts=max_facts)
    parsed = localization.get("parsed") or {}
    lines = [
        "",
        "## Debug Localization Report",
        "Use this report as advisory guidance. The final answer must still be the full corrected code.",
    ]
    suspicious = parsed.get("suspicious_lines") or []
    if suspicious:
        locs = ", ".join(str(x.get("line")) for x in suspicious[:8] if x.get("line") is not None)
        lines.append(f"Likely buggy line(s) in the broken implementation: {locs}.")
    if facts:
        lines.append("Selected facts:")
        lines.extend(f"- {fact}" for fact in facts)
    return "\n".join(lines)


def localization_summary(localization: dict | None) -> dict | None:
    if not localization or not localization.get("parsed"):
        return None
    parsed = localization["parsed"]
    return {
        "suspicious_lines": [x.get("line") for x in parsed.get("suspicious_lines", [])],
        "root_cause_hypotheses": parsed.get("root_cause_hypotheses", [])[:3],
        "repair_constraints": parsed.get("repair_constraints", [])[:3],
        "confidence": parsed.get("confidence"),
        "parse_error": localization.get("parse_error"),
    }


def analyze_patch_scope(
    config: dict,
    baseline_code: str | None,
    candidate_code: str,
    localization: dict | None,
) -> dict | None:
    if not section_enabled(config, "patch_scope_guard", default=False):
        return None
    suspicious = _suspicious_lines_from_localization(localization)
    if not baseline_code:
        return {
            "enabled": True,
            "warning": "no baseline code available for scope comparison",
            "suspicious_lines": suspicious,
        }

    changed = _changed_old_lines(baseline_code, candidate_code)
    similarity = SequenceMatcher(None, baseline_code, candidate_code).ratio()
    base_line_count = max(1, len(baseline_code.splitlines()))
    changed_ratio = len(changed) / base_line_count
    expanded_changed = set()
    for line in changed:
        expanded_changed.update(range(max(1, line - 3), line + 4))
    overlap = sorted(set(suspicious) & expanded_changed)
    large_rewrite = similarity < 0.55 or changed_ratio > 0.45
    missed = bool(suspicious) and not overlap and not large_rewrite
    warning = None
    if large_rewrite:
        warning = "candidate is a broad rewrite; verify it preserves the task contract"
    elif missed:
        warning = "candidate changes do not overlap the localized suspicious lines"
    return {
        "enabled": True,
        "suspicious_lines": suspicious,
        "changed_old_line_count": len(changed),
        "changed_old_line_ratio": round(changed_ratio, 4),
        "similarity_to_baseline": round(similarity, 4),
        "overlap_lines": overlap,
        "large_rewrite": large_rewrite,
        "missed_suspicious_lines": missed,
        "warning": warning,
    }


def _suspicious_lines_from_localization(localization: dict | None) -> list[int]:
    """Accept both debug-assist and agent-loop localisation shapes.

    The optional LLM localizer stores lines under
    ``{"parsed": {"suspicious_lines": [...]}}``. The agent loop's mandatory
    ``locate_bug`` tool stores a direct ``{"file", "line", ...}`` claim.
    Patch-scope checks should work for both.
    """
    loc = localization or {}
    raw_lines = []
    parsed = loc.get("parsed")
    if isinstance(parsed, dict):
        raw_lines.extend(parsed.get("suspicious_lines") or [])
    raw_lines.extend(loc.get("suspicious_lines") or [])
    if loc.get("line") is not None:
        raw_lines.append({"line": loc.get("line")})
    reports = loc.get("line_level_bug_report") or loc.get("bug_report") or []
    if isinstance(reports, dict):
        reports = [reports]
    raw_lines.extend(reports)

    lines: list[int] = []
    seen: set[int] = set()
    for item in raw_lines:
        if isinstance(item, dict):
            value = item.get("line") or item.get("line_number") or item.get("start_line")
        else:
            value = item
        line = _coerce_int(value)
        if line is None or line in seen:
            continue
        seen.add(line)
        lines.append(line)
    return lines


def verify_failed_patch(
    client: Any,
    config: dict,
    task_stem: str | None,
    sample_id: str | None,
    iteration: int | None,
    prompt_text: str,
    broken_code: str,
    candidate_code: str,
    localization: dict | None,
    classification: dict,
    build_output: str,
    test_output: str | None,
    trace_writer: Any | None = None,
) -> dict | None:
    if not section_enabled(config, "patch_verifier", default=False):
        return None
    mode = ((config.get("debug_tools") or {}).get("patch_verifier") or {}).get(
        "mode", "on_failed_attempt"
    )
    if mode != "on_failed_attempt":
        return None

    from .llm_client import call_llm

    report = format_localization_report(config, localization)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a CUDA patch review assistant. Return ONLY compact JSON. "
                "Do not return full corrected code."
            ),
        },
        {
            "role": "user",
            "content": "\n".join(
                [
                    "Review why this attempted CUDA fix still fails and give concise guidance for the next attempt.",
                    "",
                    "Expected JSON keys: patch_issue, next_fix_guidance, confidence.",
                    "",
                    "## Task",
                    _truncate(prompt_text, 1800),
                    report,
                    "",
                    "## Broken Code",
                    "```cuda",
                    _truncate(broken_code, 2200),
                    "```",
                    "",
                    "## Failed Candidate",
                    "```cuda",
                    _truncate(candidate_code, 2200),
                    "```",
                    "",
                    "## Failure Classification",
                    json.dumps(classification, ensure_ascii=False),
                    "",
                    "## Build/Test Output",
                    "```",
                    _truncate((build_output or "") + "\n" + (test_output or ""), 2200),
                    "```",
                ]
            ),
        },
    ]
    try:
        raw = call_llm(client, config, messages)
    except Exception as exc:  # pragma: no cover - network/API defensive path
        result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        trace_ref = _write_llm_trace(
            trace_writer=trace_writer,
            task_stem=task_stem or "unknown_task",
            stage_parts=_iter_stage_parts(iteration, "patch_verifier"),
            stage="patch_verifier",
            task_id=task_stem,
            sample_id=sample_id,
            iteration=iteration,
            messages=messages,
            response_text=None,
            parsed=result,
            summary=result,
        )
        if trace_ref:
            result["trace"] = trace_ref
        return result

    parsed = parse_verifier_response(raw)
    parsed["status"] = "ok" if parsed.get("parsed") else "parse_error"
    trace_ref = _write_llm_trace(
        trace_writer=trace_writer,
        task_stem=task_stem or "unknown_task",
        stage_parts=_iter_stage_parts(iteration, "patch_verifier"),
        stage="patch_verifier",
        task_id=task_stem,
        sample_id=sample_id,
        iteration=iteration,
        messages=messages,
        response_text=raw,
        parsed=parsed,
        summary={
            "status": parsed.get("status"),
            "parse_error": parsed.get("parse_error"),
            "parsed": parsed.get("parsed"),
        },
    )
    if trace_ref:
        parsed["trace"] = trace_ref
    return parsed


def parse_verifier_response(text: str) -> dict:
    raw = text or ""
    body = _strip_code_fence(raw.strip())
    if not body.startswith("{"):
        match = re.search(r"\{.*\}", body, flags=re.DOTALL)
        if match:
            body = match.group(0)
    try:
        payload = json.loads(body)
    except Exception as exc:
        return {
            "parsed": None,
            "raw_response": _truncate(raw, 1600),
            "parse_error": f"{type(exc).__name__}: {exc}",
        }
    if not isinstance(payload, dict):
        return {
            "parsed": None,
            "raw_response": _truncate(raw, 1600),
            "parse_error": f"Expected JSON object, got {type(payload).__name__}",
        }
    return {
        "parsed": {
            "patch_issue": str(payload.get("patch_issue") or payload.get("issue") or "")[:500],
            "next_fix_guidance": str(
                payload.get("next_fix_guidance") or payload.get("guidance") or ""
            )[:500],
            "confidence": payload.get("confidence"),
        },
        "raw_response": _truncate(raw, 1600),
        "parse_error": None,
    }


def append_debug_feedback(
    feedback_msg: str | None,
    scope: dict | None = None,
    verifier: dict | None = None,
) -> str | None:
    if feedback_msg is None:
        return None
    additions = []
    if scope and scope.get("warning"):
        additions.extend(["", "## Patch Scope Check", f"- {scope['warning']}"])
    parsed_verifier = (verifier or {}).get("parsed") or {}
    if parsed_verifier:
        additions.append("")
        additions.append("## Patch Verifier Guidance")
        if parsed_verifier.get("patch_issue"):
            additions.append(f"- Patch issue: {parsed_verifier['patch_issue']}")
        if parsed_verifier.get("next_fix_guidance"):
            additions.append(f"- Next fix guidance: {parsed_verifier['next_fix_guidance']}")
    if not additions:
        return feedback_msg
    return feedback_msg + "\n" + "\n".join(additions)


def collect_testbench_context(dataset_dir: Path, task_stem: str, max_chars: int = 2400) -> str:
    try:
        root = resolve_task_paths(task_stem, dataset_dir).testbench_path
    except FileNotFoundError:
        root = dataset_dir / "testbench" / task_stem
    try:
        if not root.exists():
            return ""
    except OSError:
        return ""
    chunks = []
    try:
        paths = sorted(root.rglob("*"))
    except OSError:
        return ""
    for path in paths:
        if len("\n".join(chunks)) >= max_chars:
            break
        if not path.is_file() or path.suffix.lower() not in {".cu", ".cuh", ".h", ".hpp", ".cpp", ".py"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel = path.relative_to(root)
        chunks.append(f"### {rel}\n{_truncate(text, 900)}")
    return _truncate("\n\n".join(chunks), max_chars)


def precheck_category(precheck: dict | None) -> str | None:
    classification = (precheck or {}).get("classification") or {}
    return classification.get("category") or classification.get("five_category")


def format_precheck_prompt_section(
    precheck: dict | None,
    *,
    fallback_excerpt: str = "",
    fallback_category: str = "unknown",
) -> str:
    """Render the real precheck result for the first debug prompt."""
    if not precheck:
        return (
            f"Error log excerpt:\n```\n{fallback_excerpt[:1500]}\n```"
            if fallback_excerpt
            else "(No error log captured for this starting code - infer the bug from the code + task spec.)"
        )

    classification = precheck.get("classification") or {}
    explicit_pre_category = classification.get("category") or classification.get("five_category")
    pre_category = explicit_pre_category or "unknown"
    lines = [
        "Precheck result from compiling and testing the current broken code:",
        f"- status: {precheck.get('status')}",
        f"- build_ok: {precheck.get('build_ok')}",
        f"- test_returncode: {precheck.get('test_returncode')}",
        f"- category: {pre_category}",
        f"- error_type: {classification.get('error_type')}",
        f"- repair_mode: {classification.get('repair_mode')}",
    ]
    if explicit_pre_category and fallback_category and fallback_category != pre_category:
        lines.append(f"- task-list category: {fallback_category} (overridden by precheck)")

    sanitizer = precheck.get("sanitizer") or {}
    if sanitizer:
        real_hits = sanitizer.get("real_hit_tools") or []
        findings = sanitizer.get("primary_findings") or []
        timed_out = sanitizer.get("timed_out_tools") or []
        unavailable = sanitizer.get("unavailable_tools") or []
        lines.append(f"- sanitizer_triggered: {bool(sanitizer.get('triggered'))}")
        lines.append(f"- sanitizer_real_hits: {', '.join(real_hits) if real_hits else 'none'}")
        for finding in findings[:4]:
            tool = finding.get("tool") or "sanitizer"
            error_type = finding.get("error_type") or "sanitizer_error"
            location = ""
            if finding.get("source_file") and finding.get("line"):
                location = f" at {finding.get('source_file')}:{finding.get('line')}"
            kernel = f" in {finding.get('kernel')}" if finding.get("kernel") else ""
            lines.append(f"- sanitizer_finding: {tool}: {error_type}{location}{kernel}")
        if timed_out:
            lines.append(f"- sanitizer_timed_out: {', '.join(timed_out)}")
        if unavailable:
            lines.append(f"- sanitizer_unavailable: {', '.join(unavailable)}")

    build_excerpt = precheck.get("build_output_excerpt") or ""
    test_excerpt = precheck.get("test_output_excerpt") or ""
    if build_excerpt:
        lines.extend(["", "Build output excerpt:", "```", _truncate(build_excerpt, 1200), "```"])
    if test_excerpt:
        lines.extend(["", "Test output excerpt:", "```", _truncate(test_excerpt, 1200), "```"])
    return "\n".join(lines)


def _write_precheck_trace(
    *,
    trace_writer: Any | None,
    task_stem: str,
    input_meta: dict,
    broken_code: str,
    result: dict,
    build_output: str,
    test_output: str | None,
    sanitizer_summary: dict | None,
) -> dict | None:
    if trace_writer is None:
        return None
    build_test_output = "\n".join(
        [
            f"BUILD_OK={result.get('build_ok')}",
            f"TEST_RETURNCODE={result.get('test_returncode')}",
            "",
            "[BUILD OUTPUT]",
            build_output or "",
            "",
            "[TEST OUTPUT]",
            test_output or "",
        ]
    )
    output_json_artifacts = {}
    compact_san = compact_sanitizer_summary(
        sanitizer_summary,
        output_excerpt_chars=getattr(trace_writer, "inline_excerpt_chars", 1200),
    )
    if compact_san is not None:
        output_json_artifacts["sanitizer_summary.json"] = compact_san
    prompt_section = format_precheck_prompt_section(result)
    suffix = Path(input_meta.get("solution_file", "solution.cu")).suffix or ".txt"
    try:
        return trace_writer.write_stage(
            task_stem,
            "precheck",
            stage="precheck",
            task_id=input_meta.get("task_id", task_stem),
            input_text_artifacts={f"broken_code{suffix}": broken_code},
            output_text_artifacts={
                "build_test_output.txt": build_test_output,
                "prompt_section.md": prompt_section,
                **sanitizer_output_artifacts(sanitizer_summary),
            },
            output_json_artifacts=output_json_artifacts,
            summary={
                "status": result.get("status"),
                "build_ok": result.get("build_ok"),
                "test_returncode": result.get("test_returncode"),
                "classification": result.get("classification"),
                "sanitizer": result.get("sanitizer"),
                "error": result.get("error"),
                "agent_prompt_effect": (
                    "This precheck result is injected into localization and the first "
                    "generation prompt; its category overrides the task-list category "
                    "when classification is available."
                ),
                "consumed_by": ["localization", "iter_0001/generation"],
            },
        )
    except Exception:
        return None


def _write_llm_trace(
    *,
    trace_writer: Any | None,
    task_stem: str,
    stage_parts: str | list[str],
    stage: str,
    task_id: str | None,
    messages: list[dict],
    response_text: str | None,
    parsed: Any | None,
    summary: dict,
    sample_id: str | None = None,
    iteration: int | None = None,
) -> dict | None:
    if trace_writer is None:
        return None
    try:
        return trace_writer.write_stage(
            task_stem,
            stage_parts,
            stage=stage,
            task_id=task_id,
            sample_id=sample_id,
            iteration=iteration,
            messages=messages,
            response_text=response_text,
            parsed=parsed,
            summary=summary,
        )
    except Exception:
        return None


def _iter_stage_parts(iteration: int | None, stage: str) -> list[str] | str:
    if iteration is None:
        return stage
    return [f"iter_{iteration:04d}", stage]


def _localization_prompt(
    input_meta: dict,
    prompt_text: str,
    broken_code: str,
    precheck: dict | None,
    testbench: str,
) -> str:
    numbered = _number_code_lines(broken_code)
    precheck_payload = {
        "classification": (precheck or {}).get("classification"),
        "build_output_excerpt": (precheck or {}).get("build_output_excerpt"),
        "test_output_excerpt": (precheck or {}).get("test_output_excerpt"),
        "sanitizer": (precheck or {}).get("sanitizer"),
        "status": (precheck or {}).get("status"),
    }
    return "\n".join(
        [
            "Locate the likely root-cause code in this broken CUDA implementation.",
            "Return JSON with keys: suspicious_lines, root_cause_hypotheses, repair_constraints, facts_to_keep, confidence.",
            "Use line numbers from the numbered broken code. Do not propose full code.",
            "",
            "## Task Metadata",
            json.dumps(
                {
                    "task_id": input_meta.get("task_id"),
                    "solution_file": input_meta.get("solution_file", "solution.cu"),
                    "tags": input_meta.get("tags", []),
                },
                ensure_ascii=False,
            ),
            "",
            "## Task Description",
            _truncate(prompt_text, 2500),
            "",
            "## Precheck Failure",
            json.dumps(precheck_payload, ensure_ascii=False),
            "",
            "## Broken Implementation With Line Numbers",
            "```cuda",
            _truncate(numbered, 8000),
            "```",
            "",
            "## Testbench Context",
            "```",
            testbench or "(not included)",
            "```",
        ]
    )


def _compact_sanitizer(summary: dict | None) -> dict | None:
    if not summary:
        return None
    primary_findings = []
    for tool, info in (summary.get("results") or {}).items():
        finding = info.get("primary_finding")
        if finding:
            primary_findings.append(finding)
    return {
        "triggered": bool(summary.get("triggered")),
        "real_hit_tools": list(summary.get("real_hit_tools") or []),
        "hit_tools": list(summary.get("hit_tools") or []),
        "timed_out_tools": list(summary.get("timed_out_tools") or []),
        "unavailable_tools": list(summary.get("unavailable_tools") or []),
        "primary_findings": primary_findings,
    }


def _changed_old_lines(old: str, new: str) -> set[int]:
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    matcher = SequenceMatcher(None, old_lines, new_lines)
    changed: set[int] = set()
    for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
        if tag != "equal":
            changed.update(range(i1 + 1, i2 + 1))
    return changed


def _number_code_lines(code: str) -> str:
    return "\n".join(f"{i:4d}: {line}" for i, line in enumerate(code.splitlines(), start=1))


def _strip_code_fence(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    return match.group(1).strip() if match else text


def _list_of_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(x) for x in value if x is not None]
    return [str(value)]


def _first_int(text: str) -> int | None:
    match = re.search(r"\d+", text or "")
    return int(match.group(0)) if match else None


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return _first_int(value)
    return None


def _truncate(text: str, max_len: int) -> str:
    if not text or len(text) <= max_len:
        return text or ""
    half = max_len // 2
    return text[:half] + "\n... [truncated] ...\n" + text[-half:]
