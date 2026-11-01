"""Blackboard multi-role workflow for Cudeubgger."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from ..agent_tools import ToolEnv, call_tool, schemas_for_config
from . import deps
from .common import (
    ErrorBlackboard,
    _LoopState,
    _RoleResult,
    _accumulate_token_usage,
    _action_to_tool_call,
    _append_action_result_message,
    _attempt_history_prompt,
    _chars_per_token,
    _extract_json_action,
    _forced_action_context,
    _full_file_submissions,
    _is_recoverable_coder_stop,
    _join_decision_constraints,
    _json_action_enabled,
    _json_action_request,
    _json_role_action_request,
    _max_coder_non_submission_rounds,
    _max_patch_submissions,
    _normalize_structured_report,
    _patch_submission_limit_stop_reason,
    _record_coder_non_submission,
    _repeated_failure_alert,
    _sanitize_assistant_content,
    _scoped_views_enabled,
    _strip_thinking,
    _structured_reports_enabled,
    _token_usage_summary,
    _tried_strategies_prompt,
    _tool_decision_overrides,
    _tool_names,
    _tools_for_phase,
    _truncate_text,
)
from .prompts import (
    ANALYZER_SYSTEM_PROMPT,
    CODER_SYSTEM_PROMPT,
    EXECUTOR_SYSTEM_PROMPT,
    ROUTE_DECISION_SYSTEM_PROMPT,
    STRUCTURED_REPORT_PROMPT,
    VERIFIER_SYSTEM_PROMPT,
)
from .runtime import (
    _append_assistant_message,
    _auto_submit_after_apply_edit,
    _build_tool_env,
    _call_json_action_or_record_api_error,
    _call_llm_or_record_api_error,
    _dispatch_tool_calls,
    _record_api_error_failure,
    _seed_broken_start,
    _write_tool_trace,
)
from ..task_registry import load_input_meta, load_prompt_text

logger = logging.getLogger("cuda_debugger_exp")


@dataclass
class _NoBlackboardReports:
    """Compact handoff object for the multi-role no-blackboard ablation."""

    task_stem: str
    initial_observation: dict | None = None
    error_type: str | None = None
    analyzer_report: str | dict | None = None
    executor_report: str | dict | None = None
    verifier_report: str | None = None
    verifier_payload: dict | None = None
    next_fix: str | None = None

    def to_dict(self) -> dict:
        return {
            "task_stem": self.task_stem,
            "initial_observation": self.initial_observation,
            "error_type": self.error_type,
            "analyzer_report": self.analyzer_report,
            "executor_report": self.executor_report,
            "verifier_report": self.verifier_report,
            "verifier_payload": self.verifier_payload,
            "next_fix": self.next_fix,
        }

    def to_prompt_block(self, max_chars: int = 14000) -> str:
        return _truncate_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            max_chars,
        )


def _optional_int_config(value, default: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {
        "",
        "none",
        "null",
        "unlimited",
        "false",
    }:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return None if parsed <= 0 else parsed


def _blackboard_prompt_max_chars(config: dict) -> int | None:
    bb_cfg = (config.get("agent", {}) or {}).get("blackboard") or {}
    if not isinstance(bb_cfg, dict) or "prompt_max_chars" not in bb_cfg:
        return 14000
    return _optional_int_config(bb_cfg.get("prompt_max_chars"), 14000)


def _blackboard_tool_result_excerpt_chars(config: dict) -> int | None:
    bb_cfg = (config.get("agent", {}) or {}).get("blackboard") or {}
    if not isinstance(bb_cfg, dict) or "tool_result_excerpt_chars" not in bb_cfg:
        return 2000
    return _optional_int_config(bb_cfg.get("tool_result_excerpt_chars"), 2000)


def _maybe_truncate_for_blackboard(text: str | None, max_chars: int | None) -> str:
    if max_chars is None:
        return text or ""
    return _truncate_text(text, max_chars)


def _update_blackboard_from_tool(
    board: ErrorBlackboard,
    role_name: str,
    tool_name: str,
    args: dict,
    result,
    config: dict | None = None,
) -> None:
    if tool_name == "locate_bug" and result.ok and result.meta.get("located"):
        board.add_suspected_line({
            "source": role_name,
            "file": result.meta.get("file"),
            "line": result.meta.get("line"),
            "bug_description": result.meta.get("bug_description"),
            "evidence": result.meta.get("evidence"),
        })
        return

    if tool_name == "run_static_analysis" and result.ok:
        for flag in result.meta.get("flags") or []:
            item = dict(flag)
            item["source"] = role_name
            board.static_flags.append(item)
        return

    if tool_name in {"build_only", "test_only", "run_compute_sanitizer"}:
        excerpt_chars = (
            _blackboard_tool_result_excerpt_chars(config or {})
            if config is not None
            else 2000
        )
        board.runtime_trace.setdefault("tool_results", []).append({
            "source": role_name,
            "tool": tool_name,
            "args": args,
            "ok": result.ok,
            "error": result.error,
            "meta": result.meta,
            "content_excerpt": _maybe_truncate_for_blackboard(
                result.content, excerpt_chars
            ),
        })


def _board_block(board: ErrorBlackboard, role: str, config: dict) -> str:
    """Render the board for a role: a scoped projection when
    ``agent.blackboard.scoped_views`` is on, otherwise the full dump."""
    max_chars = _blackboard_prompt_max_chars(config)
    if _scoped_views_enabled(config):
        return board.view_for(role, max_chars=max_chars)
    return board.to_prompt_block(max_chars=max_chars)


def _force_final_report(
    client,
    config: dict,
    messages: list[dict],
    state: _LoopState,
    task_stem: str,
    input_meta: dict,
    role_name: str,
    structured: bool,
    max_tokens,
    temperature,
    json_response_format: bool,
    asst_content_cap: int,
):
    """Make one constrained call asking the role to summarize now (gap 2).

    Called when a role used all its investigation turns without emitting a
    `final_report`. Returns a structured dict (when structured reports are on),
    a free-text string, or None on API error. Bounded to a single LLM call.
    """
    if structured:
        schema = (
            '{"tool":"final_report","args":{"summary":"<=1 sentence",'
            '"key_points":["<=4 bullets"],"ruled_out":["<=4 rejected hypotheses"]}}'
        )
    else:
        schema = '{"tool":"final_report","args":{"report":"compact evidence summary"}}'
    request = (
        f"FINAL REPORT REQUIRED ({role_name})\n"
        "You have used your investigation turns. Summarize your findings now so "
        "the verifier can act on them. Return exactly one compact JSON object, "
        "nothing else:\n"
        f"{schema}\n"
    )
    request_messages = messages + [{"role": "user", "content": request}]
    resp = _call_json_action_or_record_api_error(
        client, config, request_messages, state, task_stem, input_meta,
        max_tokens=max_tokens, temperature=temperature,
        response_format=json_response_format, role=role_name,
    )
    if resp is None:
        return None
    raw = resp.get("content") or ""
    try:
        action = _extract_json_action(raw)
    except ValueError:
        # Model ignored the schema — keep its prose as a free-text report.
        return _sanitize_assistant_content(raw, max_chars=asst_content_cap)
    tool_name = action.get("tool") or action.get("name") or action.get("action")
    args = action.get("args") if isinstance(action.get("args"), dict) else {}
    if tool_name != "final_report":
        return _sanitize_assistant_content(raw, max_chars=asst_content_cap)
    if structured:
        payload = args.get("report")
        if not isinstance(payload, dict):
            payload = args
        return _normalize_structured_report(payload)
    return str(args.get("report") or "")


def _run_tool_role_agent(
    role_name: str,
    system_prompt: str,
    user_prompt: str,
    allowed_tools: list[str],
    client,
    config: dict,
    env: ToolEnv,
    state: _LoopState,
    board: ErrorBlackboard | None,
    task_stem: str,
    input_meta: dict,
    max_turns: int,
    max_tool_calls: int,
    asst_content_cap: int,
) -> _RoleResult:
    structured = _structured_reports_enabled(config)
    if structured:
        system_prompt = system_prompt + "\n" + STRUCTURED_REPORT_PROMPT
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    reports: list[str] = []
    structured_report: dict | None = None

    def _resolve_report():
        """The role's report: the structured object if one was produced,
        else the joined free-text reports."""
        if structured_report is not None:
            return structured_report
        return "\n\n".join(reports)

    tools = schemas_for_config(allowed_tools)
    tool_decision_max_tokens, tool_decision_temperature = (
        _tool_decision_overrides(config.get("agent", {}) or {})
    )
    use_json_action = _json_action_enabled(config)
    json_response_format = bool(
        (config.get("agent", {}) or {}).get("json_action_response_format", True)
    )

    for turn in range(max_turns):
        if use_json_action:
            action_request = _json_role_action_request(
                tools,
                role_name,
                allow_final_report=True,
                extra_context=_forced_action_context(state),
                structured_report=structured,
            )
            request_messages = messages + [{"role": "user", "content": action_request}]
            resp = _call_json_action_or_record_api_error(
                client,
                config,
                request_messages,
                state,
                task_stem,
                input_meta,
                max_tokens=tool_decision_max_tokens,
                temperature=tool_decision_temperature,
                response_format=json_response_format,
                role=role_name,
            )
            if resp is None:
                return _RoleResult(stop_reason="api_error")
            raw_content = resp.get("content") or ""
            messages.append({
                "role": "assistant",
                "content": _sanitize_assistant_content(raw_content, max_chars=asst_content_cap),
            })
            try:
                action = _extract_json_action(raw_content)
            except ValueError as e:
                _append_action_result_message(
                    messages,
                    "parse_json_action",
                    f"[invalid JSON action] {e}\nRaw response:\n{raw_content[:1000]}",
                    ok=False,
                )
                continue

            tool_name = action.get("tool") or action.get("name") or action.get("action")
            if tool_name == "final_report":
                args = action.get("args") if isinstance(action.get("args"), dict) else {}
                if structured:
                    # Accept either flat args ({summary, key_points, ruled_out})
                    # or a nested {report: {...}} — whichever the model emits.
                    payload = args.get("report")
                    if not isinstance(payload, dict):
                        payload = args
                    structured_report = _normalize_structured_report(payload)
                else:
                    report = str(args.get("report") or "")
                    if report:
                        reports.append(report)
                break

            try:
                tc = _action_to_tool_call(
                    action,
                    _tool_names(tools),
                    call_id=f"{role_name}_json_action_{turn + 1}",
                )
            except ValueError as e:
                _append_action_result_message(
                    messages,
                    "parse_json_action",
                    f"[invalid JSON action] {e}",
                    ok=False,
                )
                continue

            if state.tool_calls_used >= max_tool_calls:
                _append_action_result_message(
                    messages,
                    tc["function"]["name"],
                    f"[tool budget exhausted: {state.tool_calls_used}/{max_tool_calls}]",
                    ok=False,
                )
                return _RoleResult(
                    report=_resolve_report(),
                    stop_reason="tool_budget_exhausted",
                )

            state.tool_calls_used += 1
            name = tc["function"]["name"]
            args_str = tc["function"]["arguments"]
            try:
                args = json.loads(args_str) if isinstance(args_str, str) and args_str else {}
            except json.JSONDecodeError:
                args = {}
            if state.force_read_code_reason and name != "read_code":
                _append_action_result_message(
                    messages,
                    name,
                    (
                        "[action rejected] A previous apply_edit failed. "
                        "Read the full current code before attempting another "
                        "patch."
                    ),
                    ok=False,
                )
                continue
            if (
                state.force_read_code_reason
                and name == "read_code"
                and state.force_read_code_full_file
                and not bool(args.get("full_file"))
            ):
                _append_action_result_message(
                    messages,
                    name,
                    (
                        "[action rejected] A previous apply_edit failed because "
                        "old_string did not match. Call read_code with "
                        "`{ \"full_file\": true }` before any further patch."
                    ),
                    ok=False,
                )
                continue

            result = call_tool(name, args_str, env, iteration=state.iteration + 1)
            if name == "read_code" and result.ok and state.force_read_code_reason:
                if (
                    not state.force_read_code_full_file
                    or bool(result.meta.get("full_file"))
                ):
                    state.force_read_code_reason = None
                    state.force_read_code_full_file = False
            if board is not None:
                _update_blackboard_from_tool(
                    board, role_name, name, args, result, config
                )
            _append_action_result_message(
                messages, name, result.to_message_content(), result.ok
            )
            continue

        resp = _call_llm_or_record_api_error(
            client, config, messages, tools, state, task_stem, input_meta,
            max_tokens=tool_decision_max_tokens,
            temperature=tool_decision_temperature,
            role=role_name,
        )
        if resp is None:
            return _RoleResult(stop_reason="api_error")

        content = _sanitize_assistant_content(
            resp.get("content"), max_chars=asst_content_cap,
        )
        if content:
            reports.append(content)
        _append_assistant_message(messages, resp, asst_content_cap)

        tool_calls = resp.get("tool_calls") or []
        if not tool_calls:
            break

        for tc in tool_calls:
            if state.tool_calls_used >= max_tool_calls:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": (
                        f"[tool budget exhausted: "
                        f"{state.tool_calls_used}/{max_tool_calls}]"
                    ),
                })
                return _RoleResult(
                    report=_resolve_report(),
                    stop_reason="tool_budget_exhausted",
                )

            state.tool_calls_used += 1
            name = tc["function"]["name"]
            args_str = tc["function"]["arguments"]
            try:
                args = json.loads(args_str) if isinstance(args_str, str) and args_str else {}
            except json.JSONDecodeError:
                args = {}
            if state.force_read_code_reason and name != "read_code":
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": (
                        "[action rejected] A previous apply_edit failed. "
                        "Read the full current code before attempting another patch."
                    ),
                })
                continue
            if (
                state.force_read_code_reason
                and name == "read_code"
                and state.force_read_code_full_file
                and not bool(args.get("full_file"))
            ):
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": (
                        "[action rejected] A previous apply_edit failed because "
                        "old_string did not match. Call read_code with "
                        "`{ \"full_file\": true }` before any further patch."
                    ),
                })
                continue

            result = call_tool(name, args_str, env, iteration=state.iteration + 1)
            if name == "read_code" and result.ok and state.force_read_code_reason:
                if (
                    not state.force_read_code_full_file
                    or bool(result.meta.get("full_file"))
                ):
                    state.force_read_code_reason = None
                    state.force_read_code_full_file = False
            if board is not None:
                _update_blackboard_from_tool(
                    board, role_name, name, args, result, config
                )
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result.to_message_content(),
            })

    # The role used all its investigation turns without summarizing — force a
    # final report so the verifier always has this role's conclusion (gap 2).
    if structured_report is None and not reports:
        forced = _force_final_report(
            client, config, messages, state, task_stem, input_meta,
            role_name, structured,
            tool_decision_max_tokens, tool_decision_temperature,
            json_response_format, asst_content_cap,
        )
        if isinstance(forced, dict):
            structured_report = forced
        elif forced:
            reports.append(forced)

    return _RoleResult(report=_resolve_report())


def _collect_runtime_observation(
    env: ToolEnv,
    task_stem: str,
    dataset_dir: Path,
    input_meta: dict,
    config: dict,
) -> dict:
    workdir = Path(config["build"]["workdir"])
    build_timeout = int(config["build"]["timeout"])
    test_timeout = int(config["build"]["test_timeout"])
    build_ok, build_output = deps.compile_code(
        env.current_code, task_stem, dataset_dir, input_meta, workdir, build_timeout,
    )
    test_returncode, test_output = None, None
    if build_ok:
        test_returncode, test_output = deps.run_test(
            task_stem, input_meta, workdir, test_timeout,
        )
    classification = deps.classify_result(
        build_ok, build_output, test_output, test_returncode,
    )
    return {
        "build_ok": build_ok,
        "build_output_excerpt": _truncate_text(build_output, 3000),
        "test_returncode": test_returncode,
        "test_output_excerpt": _truncate_text(test_output, 3000),
        "classification": classification,
    }


def _extract_json_payload(text: str) -> dict | None:
    raw = _strip_thinking(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    if not raw.startswith("{"):
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            raw = match.group(0)
    try:
        obj = json.loads(raw)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _parse_verifier_payload(report: str) -> tuple[dict | None, str | None, str | None]:
    """Parse verifier JSON and normalize the single next-fix direction."""
    parsed = _extract_json_payload(report)
    if not parsed:
        return None, None, None
    error_type = str(parsed["error_type"]) if parsed.get("error_type") else None
    next_fix = parsed.get("next_fix")
    if not next_fix:
        suggestions = parsed.get("fix_suggestions") or []
        if isinstance(suggestions, str):
            suggestions = [suggestions]
        next_fix = suggestions[0] if suggestions else None
    return parsed, error_type, str(next_fix).strip() if next_fix else None


def _run_blackboard_verifier(
    client,
    config: dict,
    state: _LoopState,
    board: ErrorBlackboard,
    task_stem: str,
    input_meta: dict,
    prompt_text: str,
    current_code: str,
    attempt_history: str = "[]",
    repeated_failure_alert: str | None = None,
    tried_strategies: str | None = None,
    route_diagnosis: dict | None = None,
) -> _RoleResult:
    repeat_section = (
        f"## Repeated Failure Alert\n{repeated_failure_alert}\n\n"
        if repeated_failure_alert
        else ""
    )
    tried_section = (
        f"## {tried_strategies}\n\n" if tried_strategies else ""
    )
    # When the previous attempt failed because the coder mis-implemented an
    # otherwise-sound strategy, keep the direction and fix the implementation.
    route_section = ""
    if route_diagnosis:
        route_section = (
            "## Routing Diagnosis (from the last failed attempt)\n"
            f"{json.dumps(route_diagnosis, indent=2, ensure_ascii=False)}\n\n"
        )
    messages = [
        {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "## Task Prompt\n"
                f"{_truncate_text(prompt_text, 4000)}\n\n"
                "## Current Candidate\n"
                "```cuda\n"
                f"{_truncate_text(current_code, 8000)}\n"
                "```\n\n"
                "## Previous Failed Attempts\n"
                f"{attempt_history}\n\n"
                f"{tried_section}"
                f"{route_section}"
                f"{repeat_section}"
                "## Shared Error Blackboard\n"
                f"{_board_block(board, 'verifier', config)}\n"
            ),
        },
    ]
    tool_decision_max_tokens, tool_decision_temperature = _tool_decision_overrides(
        config.get("agent", {}) or {}
    )
    try:
        raw = deps.call_llm(
            client,
            config,
            messages,
            max_tokens=tool_decision_max_tokens,
            temperature=tool_decision_temperature,
        )
    except Exception as e:
        logger.error(f"  [blackboard verifier] API error: {e}")
        stop_reason = _record_api_error_failure(
            state, task_stem, input_meta, e, role="verifier"
        )
        return _RoleResult(stop_reason=stop_reason)

    # call_llm returns plain text (no usage object), so only the estimated
    # prompt-token cost is recorded for the verifier.
    _accumulate_token_usage(
        state, "verifier", messages, None, _chars_per_token(config)
    )

    board.verifier_report = _sanitize_assistant_content(raw, max_chars=12000) or ""
    parsed, error_type, next_fix = _parse_verifier_payload(board.verifier_report)
    if parsed:
        if error_type:
            board.error_type = error_type
        reports = parsed.get("line_level_bug_report") or parsed.get("bug_report") or []
        if isinstance(reports, dict):
            reports = [reports]
        for item in reports:
            if not isinstance(item, dict):
                continue
            board.add_suspected_line({
                "source": "verifier",
                "file": item.get("file", "solution.cu"),
                "line": item.get("line"),
                "confidence": item.get("confidence"),
                "root_cause": item.get("root_cause") or item.get("bug_description"),
                "evidence": item.get("evidence"),
            })
        # Single fix direction (gap 4). Prefer `next_fix`; fall back to the
        # first legacy `fix_suggestions` entry so older outputs still work.
        board.next_fix = next_fix
        # Keep fix_suggestions populated (as the single direction) for any
        # consumer that still reads it.
        board.fix_suggestions = [board.next_fix] if board.next_fix else []
        # Stash the direction so the submission recorder can log it into the
        # tried-strategy registry (gap 6).
        state.current_fix_direction = board.next_fix

    return _RoleResult(report=board.verifier_report)


def _run_no_blackboard_verifier(
    client,
    config: dict,
    state: _LoopState,
    reports: _NoBlackboardReports,
    task_stem: str,
    input_meta: dict,
    prompt_text: str,
    current_code: str,
    attempt_history: str = "[]",
    repeated_failure_alert: str | None = None,
    tried_strategies: str | None = None,
    route_diagnosis: dict | None = None,
) -> _RoleResult:
    repeat_section = (
        f"## Repeated Failure Alert\n{repeated_failure_alert}\n\n"
        if repeated_failure_alert
        else ""
    )
    tried_section = f"## {tried_strategies}\n\n" if tried_strategies else ""
    route_section = ""
    if route_diagnosis:
        route_section = (
            "## Routing Diagnosis (from the last failed attempt)\n"
            f"{json.dumps(route_diagnosis, indent=2, ensure_ascii=False)}\n\n"
        )
    messages = [
        {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "No shared error blackboard is available in this ablation. "
                "Use only the role reports below, previous attempts, and the "
                "current code to produce the required JSON verifier output.\n\n"
                "## Task Prompt\n"
                f"{_truncate_text(prompt_text, 4000)}\n\n"
                "## Current Candidate\n"
                "```cuda\n"
                f"{_truncate_text(current_code, 8000)}\n"
                "```\n\n"
                "## Previous Failed Attempts\n"
                f"{attempt_history}\n\n"
                f"{tried_section}"
                f"{route_section}"
                f"{repeat_section}"
                "## Role Reports\n"
                f"{reports.to_prompt_block()}\n"
            ),
        },
    ]
    tool_decision_max_tokens, tool_decision_temperature = _tool_decision_overrides(
        config.get("agent", {}) or {}
    )
    try:
        raw = deps.call_llm(
            client,
            config,
            messages,
            max_tokens=tool_decision_max_tokens,
            temperature=tool_decision_temperature,
        )
    except Exception as e:
        logger.error(f"  [no-blackboard verifier] API error: {e}")
        stop_reason = _record_api_error_failure(
            state, task_stem, input_meta, e, role="verifier"
        )
        return _RoleResult(stop_reason=stop_reason)

    _accumulate_token_usage(
        state, "verifier", messages, None, _chars_per_token(config)
    )
    reports.verifier_report = _sanitize_assistant_content(raw, max_chars=12000) or ""
    parsed, error_type, next_fix = _parse_verifier_payload(reports.verifier_report)
    reports.verifier_payload = parsed
    if error_type:
        reports.error_type = error_type
    reports.next_fix = next_fix
    state.current_fix_direction = next_fix
    return _RoleResult(report=reports.verifier_report)


def _run_route_decision(
    client,
    config: dict,
    state: _LoopState,
    candidate_code: str,
    located: dict | None,
    next_fix: str | None,
    latest_attempt: dict | None,
    tried_strategies: str | None,
    task_stem: str | None = None,
    input_meta: dict | None = None,
) -> dict:
    """After a failed submission, decide whether to retry the fix or re-analyze.

    The prompt includes the submitted code, the localization, and the attempted
    fix strategy so the verifier can tell a wrong *strategy* from a coder that
    simply mis-implemented a sound one (gap 5b). Returns
    ``{next_action, diagnosis, reason}``; defaults to ``reanalyze`` when the
    response cannot be parsed (the safe, evidence-gathering choice).
    """
    cls = (latest_attempt or {}).get("classification") or {}
    result_summary = {
        "category": cls.get("category"),
        "five_category": cls.get("five_category"),
        "build_ok": cls.get("build_ok"),
        "build_output_excerpt": (latest_attempt or {}).get("build_output_excerpt"),
        "test_output_excerpt": (latest_attempt or {}).get("test_output_excerpt"),
    }
    messages = [
        {"role": "system", "content": ROUTE_DECISION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "A patch was just submitted and FAILED. Decide the next step.\n\n"
                "## Attempted fix strategy (verifier's direction)\n"
                f"{next_fix or '(none recorded)'}\n\n"
                "## Localization used\n"
                f"{json.dumps(located, indent=2, ensure_ascii=False)}\n\n"
                "## Submitted candidate code\n"
                "```cuda\n"
                f"{_truncate_text(candidate_code, 8000)}\n"
                "```\n\n"
                "## Latest execution result\n"
                f"{json.dumps(result_summary, indent=2, ensure_ascii=False)}\n\n"
                f"## {tried_strategies or 'ALREADY TRIED: (none)'}\n"
            ),
        },
    ]
    tool_decision_max_tokens, tool_decision_temperature = _tool_decision_overrides(
        config.get("agent", {}) or {}
    )
    try:
        raw = deps.call_llm(
            client,
            config,
            messages,
            max_tokens=tool_decision_max_tokens,
            temperature=tool_decision_temperature,
        )
    except Exception as e:
        if deps.is_context_length_error(e):
            logger.error(f"  [blackboard router] context limit reached: {e}")
            if task_stem is not None and input_meta is not None:
                _record_api_error_failure(
                    state, task_stem, input_meta, e, role="router"
                )
            else:
                state.stop_reason = "api_context_limit"
            return {"next_action": "fail_task", "diagnosis": None, "reason": "api_context_limit"}
        logger.error(f"  [blackboard router] API error: {e}; defaulting to reanalyze")
        return {"next_action": "reanalyze", "diagnosis": None, "reason": "api_error"}

    _accumulate_token_usage(
        state, "router", messages, None, _chars_per_token(config)
    )
    parsed = _extract_json_payload(_sanitize_assistant_content(raw, max_chars=4000) or "")
    next_action = (parsed or {}).get("next_action")
    if next_action not in ("retry_fix", "reanalyze"):
        next_action = "reanalyze"  # safe default: gather fresh evidence
    return {
        "next_action": next_action,
        "diagnosis": (parsed or {}).get("diagnosis"),
        "reason": (parsed or {}).get("reason"),
    }


def _attach_blackboard_to_latest_record(
    state: _LoopState,
    board: ErrorBlackboard,
) -> None:
    if not state.records:
        return
    state.records[-1]["agent_workflow"] = "blackboard"
    state.records[-1]["error_blackboard"] = board.to_dict()


def _attach_no_blackboard_to_latest_record(
    state: _LoopState,
    reports: _NoBlackboardReports,
) -> None:
    if not state.records:
        return
    state.records[-1]["agent_workflow"] = "multi_role_no_blackboard"
    state.records[-1]["role_reports"] = reports.to_dict()


def _run_blackboard_coder_once(
    client,
    config: dict,
    env: ToolEnv,
    state: _LoopState,
    board: ErrorBlackboard,
    task_stem: str,
    dataset_dir: Path,
    input_meta: dict,
    output_dir: Path,
    prompt_text: str,
    max_tool_calls: int,
    asst_content_cap: int,
) -> str | None:
    agent_cfg = config.get("agent", {}) or {}
    bb_cfg = agent_cfg.get("blackboard") or {}
    max_turns = int(bb_cfg.get("max_coder_turns", 8))
    coder_tools = bb_cfg.get("coder_tools") or [
        "read_code",
        "read_test_file",
        "read_prompt",
        "locate_bug",
        "apply_edit",
        "write_solution",
        "build_only",
    ]
    full_file_mode = _full_file_submissions(config)
    if full_file_mode:
        coder_tools = [t for t in coder_tools if t != "apply_edit"]
    tool_decision_max_tokens, tool_decision_temperature = (
        _tool_decision_overrides(agent_cfg)
    )
    use_json_action = _json_action_enabled(config)
    json_response_format = bool(agent_cfg.get("json_action_response_format", True))
    force_locate_first = bool(
        bb_cfg.get(
            "coder_force_locate_first",
            agent_cfg.get("coder_force_locate_first", True),
        )
    )
    state.located_this_iteration = False
    state.pending_location = None
    state.applied_edit_this_iteration = False
    local_no_ops = 0
    repeated_failure_alert = _repeated_failure_alert(state)
    repeat_section = (
        f"## Repeated Failure Alert\n{repeated_failure_alert}\n\n"
        if repeated_failure_alert
        else ""
    )
    coder_instruction = (
        "Submit a complete revised solution file based on this verified blackboard.\n\n"
        if full_file_mode
        else "Apply a minimal fix based on this verified blackboard.\n\n"
    )
    messages: list[dict] = [
        {"role": "system", "content": CODER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                coder_instruction
                + "## Task Prompt\n"
                + f"{_truncate_text(prompt_text, 4000)}\n\n"
                + "## Current Candidate (full file)\n"
                + "```cuda\n"
                + f"{env.current_code}\n"
                + "```\n\n"
                + "## Previous Failed Attempts\n"
                + f"{_attempt_history_prompt(state)}\n\n"
                + f"{repeat_section}"
                + "## Shared Error Blackboard\n"
                + f"{_board_block(board, 'coder', config)}\n"
            ),
        },
    ]

    for _ in range(max_turns):
        in_locate_phase = (
            bool(env.current_code.strip()) and not state.located_this_iteration
        )
        must_edit_phase = (
            bool(env.current_code.strip())
            and state.located_this_iteration
            and not state.applied_edit_this_iteration
            and not state.force_read_code_reason
        )
        if state.force_read_code_reason:
            tools = schemas_for_config([
                t for t in ("read_code", "write_solution") if t in coder_tools
            ], require_full_solution_code=full_file_mode)
            action_context = _forced_action_context(state)
        elif in_locate_phase and force_locate_first:
            tools = schemas_for_config(
                ["locate_bug"] if "locate_bug" in coder_tools else [],
                require_full_solution_code=full_file_mode,
            )
            action_context = _join_decision_constraints(
                _forced_action_context(state),
                "The analyzer, executor, and verifier have already produced the "
                "blackboard evidence. Your next action must be `locate_bug` using "
                "the verified line-level bug report; do not investigate further "
                "in the coder role.",
            )
        elif must_edit_phase:
            if full_file_mode:
                tools = schemas_for_config(
                    ["write_solution"] if "write_solution" in coder_tools else [],
                    require_full_solution_code=True,
                )
                action_context = _join_decision_constraints(
                    _forced_action_context(state),
                    "You already located the bug. Your next action must be "
                    "`write_solution` with a complete revised `code` file. "
                    "Include the entire solution file, not a diff, snippet, "
                    "or only the changed lines. Do not read or relocalize.",
                )
            else:
                tools = schemas_for_config(
                    [
                        t for t in ("apply_edit",)
                        if t in coder_tools
                    ]
                )
                action_context = _join_decision_constraints(
                    _forced_action_context(state),
                    "You already located the bug. Your next action must be "
                    "`apply_edit` with a minimal replacement or line-range edit. "
                    "If exact matching is brittle or an `apply_edit` mismatch "
                    "blocks progress, call `write_solution` with a complete "
                    "revised `code` file so the candidate reaches build/test. Do "
                    "not read or relocalize.",
                )
        else:
            tools = _tools_for_phase(
                coder_tools,
                in_locate_phase=in_locate_phase,
                require_full_solution_code=full_file_mode,
            )
            action_context = _forced_action_context(state)
        prev_iter = state.iteration
        request_max_tokens = tool_decision_max_tokens
        request_temperature = tool_decision_temperature
        if use_json_action:
            action_request = _json_action_request(
                tools,
                in_locate_phase,
                extra_context=action_context,
            )
            request_messages = messages + [{"role": "user", "content": action_request}]
            resp = _call_json_action_or_record_api_error(
                client,
                config,
                request_messages,
                state,
                task_stem,
                input_meta,
                max_tokens=request_max_tokens,
                temperature=request_temperature,
                response_format=json_response_format,
                role="coder",
            )
            if resp is None:
                return "api_error"
            raw_content = resp.get("content") or ""
            messages.append({
                "role": "assistant",
                "content": _sanitize_assistant_content(
                    raw_content, max_chars=asst_content_cap
                ),
            })
            try:
                action = _extract_json_action(raw_content)
                tc = _action_to_tool_call(
                    action,
                    _tool_names(tools),
                    call_id=f"coder_json_action_{len(env.trace) + 1}",
                )
                resp = {"tool_calls": [tc]}
            except ValueError as e:
                local_no_ops += 1
                _append_action_result_message(
                    messages,
                    "parse_json_action",
                    f"[invalid JSON action] {e}\nRaw response:\n{raw_content[:1000]}",
                    ok=False,
                )
                if local_no_ops >= 2:
                    return "coder_action_parse_error"
                continue
        else:
            resp = _call_llm_or_record_api_error(
                client, config, messages, tools, state, task_stem, input_meta,
                max_tokens=request_max_tokens,
                temperature=request_temperature,
                role="coder",
            )
            if resp is None:
                return "api_error"
            _append_assistant_message(messages, resp, asst_content_cap)

        if not resp["tool_calls"]:
            local_no_ops += 1
            if local_no_ops >= 2:
                return "coder_no_op"
            if must_edit_phase:
                content = (
                    "You already called locate_bug. Continue with "
                    "`write_solution(code=...)` using the complete revised "
                    "solution file; do not read or relocalize."
                    if full_file_mode
                    else (
                        "You already called locate_bug. Continue with the "
                        "apply_edit tool now; do not read or relocalize."
                    )
                )
            else:
                content = (
                    "Continue with tool calls: locate_bug, then "
                    "write_solution(code=complete_file). Do not answer in prose only."
                    if full_file_mode
                    else (
                        "Continue with tool calls: locate_bug, apply_edit, then "
                        "write_solution. Do not answer in prose only."
                    )
                )
            messages.append({
                "role": "user",
                "content": content,
            })
            continue
        local_no_ops = 0

        stop = _dispatch_tool_calls(
            resp["tool_calls"], env, state, messages,
            max_tool_calls,
            task_stem, dataset_dir, input_meta, config, output_dir,
            client, prompt_text,
            json_action=use_json_action,
        )
        if state.iteration > prev_iter:
            _attach_blackboard_to_latest_record(state, board)
            if stop is not None:
                return stop
            return "submitted"
        if stop is not None:
            return stop

        guard_stop = _auto_submit_after_apply_edit(
            env, state, messages,
            max_tool_calls,
            task_stem, dataset_dir, input_meta, config, output_dir,
            client, prompt_text,
            json_action=use_json_action,
        )
        if state.iteration > prev_iter:
            _attach_blackboard_to_latest_record(state, board)
            if guard_stop is not None:
                return guard_stop
            return "submitted"
        if guard_stop is not None:
            return guard_stop

    return "coder_no_submission"


def _run_no_blackboard_coder_once(
    client,
    config: dict,
    env: ToolEnv,
    state: _LoopState,
    reports: _NoBlackboardReports,
    task_stem: str,
    dataset_dir: Path,
    input_meta: dict,
    output_dir: Path,
    prompt_text: str,
    max_tool_calls: int,
    asst_content_cap: int,
) -> str | None:
    agent_cfg = config.get("agent", {}) or {}
    bb_cfg = agent_cfg.get("blackboard") or {}
    max_turns = int(bb_cfg.get("max_coder_turns", 8))
    coder_tools = bb_cfg.get("coder_tools") or [
        "read_code",
        "read_test_file",
        "read_prompt",
        "locate_bug",
        "apply_edit",
        "write_solution",
        "build_only",
    ]
    full_file_mode = _full_file_submissions(config)
    if full_file_mode:
        coder_tools = [t for t in coder_tools if t != "apply_edit"]
    tool_decision_max_tokens, tool_decision_temperature = (
        _tool_decision_overrides(agent_cfg)
    )
    use_json_action = _json_action_enabled(config)
    json_response_format = bool(agent_cfg.get("json_action_response_format", True))
    force_locate_first = bool(
        bb_cfg.get(
            "coder_force_locate_first",
            agent_cfg.get("coder_force_locate_first", True),
        )
    )
    state.located_this_iteration = False
    state.pending_location = None
    state.applied_edit_this_iteration = False
    local_no_ops = 0
    repeated_failure_alert = _repeated_failure_alert(state)
    repeat_section = (
        f"## Repeated Failure Alert\n{repeated_failure_alert}\n\n"
        if repeated_failure_alert
        else ""
    )
    coder_instruction = (
        "Submit a complete revised solution file based on the verified role reports.\n\n"
        if full_file_mode
        else "Apply a minimal fix based on the verified role reports.\n\n"
    )
    messages: list[dict] = [
        {"role": "system", "content": CODER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                coder_instruction
                + "No shared error blackboard is available in this ablation.\n\n"
                + "## Task Prompt\n"
                + f"{_truncate_text(prompt_text, 4000)}\n\n"
                + "## Current Candidate (full file)\n"
                + "```cuda\n"
                + f"{env.current_code}\n"
                + "```\n\n"
                + "## Previous Failed Attempts\n"
                + f"{_attempt_history_prompt(state)}\n\n"
                + f"{repeat_section}"
                + "## Role Reports and Verifier Direction\n"
                + f"{reports.to_prompt_block()}\n"
            ),
        },
    ]

    for _ in range(max_turns):
        in_locate_phase = (
            bool(env.current_code.strip()) and not state.located_this_iteration
        )
        must_edit_phase = (
            bool(env.current_code.strip())
            and state.located_this_iteration
            and not state.applied_edit_this_iteration
            and not state.force_read_code_reason
        )
        if state.force_read_code_reason:
            tools = schemas_for_config([
                t for t in ("read_code", "write_solution") if t in coder_tools
            ], require_full_solution_code=full_file_mode)
            action_context = _forced_action_context(state)
        elif in_locate_phase and force_locate_first:
            tools = schemas_for_config(
                ["locate_bug"] if "locate_bug" in coder_tools else [],
                require_full_solution_code=full_file_mode,
            )
            action_context = _join_decision_constraints(
                _forced_action_context(state),
                "The analyzer, executor, and verifier have already produced "
                "the role reports. Your next action must be `locate_bug` using "
                "the verified line-level bug report; do not investigate further "
                "in the coder role.",
            )
        elif must_edit_phase:
            if full_file_mode:
                tools = schemas_for_config(
                    ["write_solution"] if "write_solution" in coder_tools else [],
                    require_full_solution_code=True,
                )
                action_context = _join_decision_constraints(
                    _forced_action_context(state),
                    "You already located the bug. Your next action must be "
                    "`write_solution` with a complete revised `code` file. "
                    "Include the entire solution file, not a diff, snippet, "
                    "or only the changed lines. Do not read or relocalize.",
                )
            else:
                tools = schemas_for_config(
                    [
                        t for t in ("apply_edit",)
                        if t in coder_tools
                    ]
                )
                action_context = _join_decision_constraints(
                    _forced_action_context(state),
                    "You already located the bug. Your next action must be "
                    "`apply_edit` with a minimal replacement or line-range edit. "
                    "If exact matching is brittle or an `apply_edit` mismatch "
                    "blocks progress, call `write_solution` with a complete "
                    "revised `code` file so the candidate reaches build/test. Do "
                    "not read or relocalize.",
                )
        else:
            tools = _tools_for_phase(
                coder_tools,
                in_locate_phase=in_locate_phase,
                require_full_solution_code=full_file_mode,
            )
            action_context = _forced_action_context(state)
        prev_iter = state.iteration
        request_max_tokens = tool_decision_max_tokens
        request_temperature = tool_decision_temperature
        if use_json_action:
            action_request = _json_action_request(
                tools,
                in_locate_phase,
                extra_context=action_context,
            )
            request_messages = messages + [{"role": "user", "content": action_request}]
            resp = _call_json_action_or_record_api_error(
                client,
                config,
                request_messages,
                state,
                task_stem,
                input_meta,
                max_tokens=request_max_tokens,
                temperature=request_temperature,
                response_format=json_response_format,
                role="coder",
            )
            if resp is None:
                return "api_error"
            raw_content = resp.get("content") or ""
            messages.append({
                "role": "assistant",
                "content": _sanitize_assistant_content(
                    raw_content, max_chars=asst_content_cap
                ),
            })
            try:
                action = _extract_json_action(raw_content)
                tc = _action_to_tool_call(
                    action,
                    _tool_names(tools),
                    call_id=f"coder_json_action_{len(env.trace) + 1}",
                )
                resp = {"tool_calls": [tc]}
            except ValueError as e:
                local_no_ops += 1
                _append_action_result_message(
                    messages,
                    "parse_json_action",
                    f"[invalid JSON action] {e}\nRaw response:\n{raw_content[:1000]}",
                    ok=False,
                )
                if local_no_ops >= 2:
                    return "coder_action_parse_error"
                continue
        else:
            resp = _call_llm_or_record_api_error(
                client, config, messages, tools, state, task_stem, input_meta,
                max_tokens=request_max_tokens,
                temperature=request_temperature,
                role="coder",
            )
            if resp is None:
                return "api_error"
            _append_assistant_message(messages, resp, asst_content_cap)

        if not resp["tool_calls"]:
            local_no_ops += 1
            if local_no_ops >= 2:
                return "coder_no_op"
            if must_edit_phase:
                content = (
                    "You already called locate_bug. Continue with "
                    "`write_solution(code=...)` using the complete revised "
                    "solution file; do not read or relocalize."
                    if full_file_mode
                    else (
                        "You already called locate_bug. Continue with the "
                        "apply_edit tool now; do not read or relocalize."
                    )
                )
            else:
                content = (
                    "Continue with tool calls: locate_bug, then "
                    "write_solution(code=complete_file). Do not answer in prose only."
                    if full_file_mode
                    else (
                        "Continue with tool calls: locate_bug, apply_edit, then "
                        "write_solution. Do not answer in prose only."
                    )
                )
            messages.append({
                "role": "user",
                "content": content,
            })
            continue
        local_no_ops = 0

        stop = _dispatch_tool_calls(
            resp["tool_calls"], env, state, messages,
            max_tool_calls,
            task_stem, dataset_dir, input_meta, config, output_dir,
            client, prompt_text,
            json_action=use_json_action,
        )
        if state.iteration > prev_iter:
            _attach_no_blackboard_to_latest_record(state, reports)
            if stop is not None:
                return stop
            return "submitted"
        if stop is not None:
            return stop

        guard_stop = _auto_submit_after_apply_edit(
            env, state, messages,
            max_tool_calls,
            task_stem, dataset_dir, input_meta, config, output_dir,
            client, prompt_text,
            json_action=use_json_action,
        )
        if state.iteration > prev_iter:
            _attach_no_blackboard_to_latest_record(state, reports)
            if guard_stop is not None:
                return guard_stop
            return "submitted"
        if guard_stop is not None:
            return guard_stop

    return "coder_no_submission"


def run_task_blackboard_agent(
    task_stem: str,
    dataset_dir: Path,
    output_dir: Path,
    config: dict,
    client,
    broken_start: dict | None = None,
) -> tuple[list[dict], str]:
    """Run analyzer -> executor -> verifier -> coder over a shared board."""
    logger.info(f"=== [blackboard agent] Starting task: {task_stem} ===")
    if not broken_start:
        raise ValueError(
            "run_task_blackboard_agent requires `broken_start` — the agent "
            "loop is debug-only."
        )

    input_meta = load_input_meta(task_stem, dataset_dir)
    prompt_text = load_prompt_text(task_stem, dataset_dir)

    agent_cfg = config.get("agent", {}) or {}
    bb_cfg = agent_cfg.get("blackboard") or {}
    env = _build_tool_env(task_stem, dataset_dir, input_meta, config, agent_cfg)
    state = _LoopState()
    _seed_broken_start(
        broken_start, prompt_text, env, state,
        config, task_stem, dataset_dir, input_meta,
    )

    max_patch_submissions = _max_patch_submissions(config)
    max_tool_calls = int(agent_cfg.get("max_tool_calls_per_task", 40))
    asst_content_cap = int(agent_cfg.get("assistant_content_cap", 2000))
    max_role_turns = int(bb_cfg.get("max_role_turns", 3))
    max_non_submission_rounds = _max_coder_non_submission_rounds(config)
    analyzer_tools = bb_cfg.get("analyzer_tools") or [
        "read_code", "read_test_file", "read_prompt", "run_static_analysis", "locate_bug",
    ]
    executor_tools = bb_cfg.get("executor_tools") or [
        "read_code", "read_test_file", "build_only", "test_only",
        "run_compute_sanitizer", "locate_bug",
    ]

    state.stop_reason = _patch_submission_limit_stop_reason(config)

    # Guided repair loop. The first round always analyzes; after a failed
    # submission the router decides whether the next round re-analyzes
    # (analyzer + executor) or just retries the fix (verifier + coder) using
    # the existing evidence.
    need_analysis = True
    board: ErrorBlackboard | None = None
    route_diagnosis: dict | None = None

    while state.iteration < max_patch_submissions:
        repeated_failure_alert = _repeated_failure_alert(state)
        tried_strategies = _tried_strategies_prompt(state)
        repeat_section = (
            f"## Repeated Failure Alert\n{repeated_failure_alert}\n\n"
            if repeated_failure_alert
            else ""
        )

        if need_analysis or board is None:
            board = ErrorBlackboard(task_stem=task_stem)
            runtime = _collect_runtime_observation(
                env, task_stem, dataset_dir, input_meta, config,
            )
            board.runtime_trace["initial_observation"] = runtime
            cls = runtime.get("classification") or {}
            board.error_type = cls.get("category") or cls.get("five_category")

            analyzer_prompt = (
                "Populate the static side of the shared blackboard.\n\n"
                "## Task Prompt\n"
                f"{_truncate_text(prompt_text, 4000)}\n\n"
                "## Current Candidate\n"
                "```cuda\n"
                f"{_truncate_text(env.current_code, 12000)}\n"
                "```\n\n"
                "## Previous Failed Attempts\n"
                f"{_attempt_history_prompt(state)}\n\n"
                f"{repeat_section}"
                "Read code/test/prompt as needed, call run_static_analysis, "
                "then locate concrete suspected lines and finish with a report."
            )
            analyzer = _run_tool_role_agent(
                "analyzer", ANALYZER_SYSTEM_PROMPT, analyzer_prompt,
                analyzer_tools, client, config, env, state, board,
                task_stem, input_meta, max_role_turns, max_tool_calls,
                asst_content_cap,
            )
            board.analyzer_report = analyzer.report
            if analyzer.stop_reason:
                state.stop_reason = analyzer.stop_reason
                break

            executor_prompt = (
                "Populate the runtime side of the shared blackboard.\n\n"
                "## Initial Runtime Observation\n"
                f"{json.dumps(runtime, indent=2, ensure_ascii=False)}\n\n"
                "## Previous Failed Attempts\n"
                f"{_attempt_history_prompt(state)}\n\n"
                f"{repeat_section}"
                "## Current Blackboard\n"
                f"{_board_block(board, 'executor', config)}\n\n"
                "Choose the Compute Sanitizer mode based on the error category "
                "and hypothesis. Combine sanitizer output with static evidence; "
                "do not treat a clean sanitizer run as proof of correctness. "
                "Finish with a report."
            )
            executor = _run_tool_role_agent(
                "executor", EXECUTOR_SYSTEM_PROMPT, executor_prompt,
                executor_tools, client, config, env, state, board,
                task_stem, input_meta, max_role_turns, max_tool_calls,
                asst_content_cap,
            )
            board.executor_report = executor.report
            if executor.stop_reason:
                state.stop_reason = executor.stop_reason
                break

        verifier = _run_blackboard_verifier(
            client,
            config,
            state,
            board,
            task_stem,
            input_meta,
            prompt_text,
            env.current_code,
            attempt_history=_attempt_history_prompt(state),
            repeated_failure_alert=repeated_failure_alert,
            tried_strategies=tried_strategies,
            route_diagnosis=route_diagnosis,
        )
        if verifier.stop_reason:
            state.stop_reason = verifier.stop_reason
            break

        coder_stop = _run_blackboard_coder_once(
            client, config, env, state, board,
            task_stem, dataset_dir, input_meta, output_dir, prompt_text,
            max_tool_calls, asst_content_cap,
        )
        if coder_stop == "submitted":
            # A patch was submitted but did not pass. Let the router decide
            # whether to retry the fix or re-analyze (gap 5b).
            state.non_submission_rounds = 0
            latest_attempt = (
                state.attempt_history[-1] if state.attempt_history else None
            )
            route = _run_route_decision(
                client, config, state,
                candidate_code=env.current_code,
                located=(latest_attempt or {}).get("located"),
                next_fix=board.next_fix,
                latest_attempt=latest_attempt,
                tried_strategies=_tried_strategies_prompt(state),
                task_stem=task_stem,
                input_meta=input_meta,
            )
            if route["next_action"] == "fail_task":
                break
            need_analysis = route["next_action"] == "reanalyze"
            route_diagnosis = None if need_analysis else route
            logger.info(
                "  [blackboard router] next_action=%s diagnosis=%s reason=%s",
                route["next_action"], route.get("diagnosis"), route.get("reason"),
            )
            continue
        if _is_recoverable_coder_stop(coder_stop):
            _record_coder_non_submission(state, coder_stop or "coder_no_submission")
            _write_tool_trace(
                output_dir,
                task_stem,
                env,
                state,
                stop_reason=coder_stop or "coder_no_submission",
            )
            logger.warning(
                "  [blackboard coder] no patch submitted: %s (%d/%d); "
                "continuing with a fresh diagnosis",
                coder_stop,
                state.non_submission_rounds,
                max_non_submission_rounds,
            )
            if state.non_submission_rounds >= max_non_submission_rounds:
                state.stop_reason = coder_stop or "coder_no_submission"
                break
            need_analysis = True       # fresh diagnosis after a non-submission
            route_diagnosis = None
            continue
        if coder_stop is not None:
            state.stop_reason = coder_stop
            break

    if state.records:
        state.records[-1]["stop_reason"] = state.stop_reason
        state.records[-1]["token_usage"] = _token_usage_summary(state)

    _write_tool_trace(output_dir, task_stem, env, state)
    return state.records, state.stop_reason


def run_task_multi_role_no_blackboard_agent(
    task_stem: str,
    dataset_dir: Path,
    output_dir: Path,
    config: dict,
    client,
    broken_start: dict | None = None,
) -> tuple[list[dict], str]:
    """Run analyzer -> executor -> verifier -> coder without a shared board."""
    logger.info(f"=== [multi-role no-blackboard agent] Starting task: {task_stem} ===")
    if not broken_start:
        raise ValueError(
            "run_task_multi_role_no_blackboard_agent requires `broken_start` — "
            "the agent loop is debug-only."
        )

    input_meta = load_input_meta(task_stem, dataset_dir)
    prompt_text = load_prompt_text(task_stem, dataset_dir)

    agent_cfg = config.get("agent", {}) or {}
    bb_cfg = agent_cfg.get("blackboard") or {}
    env = _build_tool_env(task_stem, dataset_dir, input_meta, config, agent_cfg)
    state = _LoopState()
    _seed_broken_start(
        broken_start, prompt_text, env, state,
        config, task_stem, dataset_dir, input_meta,
    )

    max_patch_submissions = _max_patch_submissions(config)
    max_tool_calls = int(agent_cfg.get("max_tool_calls_per_task", 40))
    asst_content_cap = int(agent_cfg.get("assistant_content_cap", 2000))
    max_role_turns = int(bb_cfg.get("max_role_turns", 3))
    max_non_submission_rounds = _max_coder_non_submission_rounds(config)
    analyzer_tools = bb_cfg.get("analyzer_tools") or [
        "read_code", "read_test_file", "read_prompt", "run_static_analysis", "locate_bug",
    ]
    executor_tools = bb_cfg.get("executor_tools") or [
        "read_code", "read_test_file", "build_only", "test_only",
        "run_compute_sanitizer", "locate_bug",
    ]

    state.stop_reason = _patch_submission_limit_stop_reason(config)
    need_analysis = True
    reports: _NoBlackboardReports | None = None
    route_diagnosis: dict | None = None

    while state.iteration < max_patch_submissions:
        repeated_failure_alert = _repeated_failure_alert(state)
        tried_strategies = _tried_strategies_prompt(state)
        repeat_section = (
            f"## Repeated Failure Alert\n{repeated_failure_alert}\n\n"
            if repeated_failure_alert
            else ""
        )

        if need_analysis or reports is None:
            runtime = _collect_runtime_observation(
                env, task_stem, dataset_dir, input_meta, config,
            )
            cls = runtime.get("classification") or {}
            reports = _NoBlackboardReports(
                task_stem=task_stem,
                initial_observation=runtime,
                error_type=cls.get("category") or cls.get("five_category"),
            )

            analyzer_prompt = (
                "Produce the static-analysis report for this no-shared-blackboard "
                "ablation. Do not rely on any shared mutable blackboard; your "
                "final report is the only evidence passed forward.\n\n"
                "## Task Prompt\n"
                f"{_truncate_text(prompt_text, 4000)}\n\n"
                "## Current Candidate\n"
                "```cuda\n"
                f"{_truncate_text(env.current_code, 12000)}\n"
                "```\n\n"
                "## Previous Failed Attempts\n"
                f"{_attempt_history_prompt(state)}\n\n"
                f"{repeat_section}"
                "Read code/test/prompt as needed, call run_static_analysis, "
                "then locate concrete suspected lines and finish with a report."
            )
            analyzer = _run_tool_role_agent(
                "analyzer", ANALYZER_SYSTEM_PROMPT, analyzer_prompt,
                analyzer_tools, client, config, env, state, None,
                task_stem, input_meta, max_role_turns, max_tool_calls,
                asst_content_cap,
            )
            reports.analyzer_report = analyzer.report
            if analyzer.stop_reason:
                state.stop_reason = analyzer.stop_reason
                break

            executor_prompt = (
                "Produce the runtime-diagnostics report for this "
                "no-shared-blackboard ablation. Do not write to or depend on a "
                "shared mutable blackboard; your final report is the only "
                "runtime evidence passed forward.\n\n"
                "## Initial Runtime Observation\n"
                f"{json.dumps(runtime, indent=2, ensure_ascii=False)}\n\n"
                "## Previous Failed Attempts\n"
                f"{_attempt_history_prompt(state)}\n\n"
                f"{repeat_section}"
                "## Analyzer Report\n"
                f"{json.dumps(reports.analyzer_report, indent=2, ensure_ascii=False)}\n\n"
                "Choose the Compute Sanitizer mode based on the error category "
                "and hypothesis. Combine sanitizer output with static evidence; "
                "do not treat a clean sanitizer run as proof of correctness. "
                "Finish with a report."
            )
            executor = _run_tool_role_agent(
                "executor", EXECUTOR_SYSTEM_PROMPT, executor_prompt,
                executor_tools, client, config, env, state, None,
                task_stem, input_meta, max_role_turns, max_tool_calls,
                asst_content_cap,
            )
            reports.executor_report = executor.report
            if executor.stop_reason:
                state.stop_reason = executor.stop_reason
                break

        verifier = _run_no_blackboard_verifier(
            client,
            config,
            state,
            reports,
            task_stem,
            input_meta,
            prompt_text,
            env.current_code,
            attempt_history=_attempt_history_prompt(state),
            repeated_failure_alert=repeated_failure_alert,
            tried_strategies=tried_strategies,
            route_diagnosis=route_diagnosis,
        )
        if verifier.stop_reason:
            state.stop_reason = verifier.stop_reason
            break

        coder_stop = _run_no_blackboard_coder_once(
            client, config, env, state, reports,
            task_stem, dataset_dir, input_meta, output_dir, prompt_text,
            max_tool_calls, asst_content_cap,
        )
        if coder_stop == "submitted":
            state.non_submission_rounds = 0
            latest_attempt = (
                state.attempt_history[-1] if state.attempt_history else None
            )
            route = _run_route_decision(
                client, config, state,
                candidate_code=env.current_code,
                located=(latest_attempt or {}).get("located"),
                next_fix=reports.next_fix,
                latest_attempt=latest_attempt,
                tried_strategies=_tried_strategies_prompt(state),
                task_stem=task_stem,
                input_meta=input_meta,
            )
            if route["next_action"] == "fail_task":
                break
            need_analysis = route["next_action"] == "reanalyze"
            route_diagnosis = None if need_analysis else route
            logger.info(
                "  [no-blackboard router] next_action=%s diagnosis=%s reason=%s",
                route["next_action"], route.get("diagnosis"), route.get("reason"),
            )
            continue
        if _is_recoverable_coder_stop(coder_stop):
            _record_coder_non_submission(state, coder_stop or "coder_no_submission")
            _write_tool_trace(
                output_dir,
                task_stem,
                env,
                state,
                stop_reason=coder_stop or "coder_no_submission",
            )
            logger.warning(
                "  [no-blackboard coder] no patch submitted: %s (%d/%d); "
                "continuing with a fresh diagnosis",
                coder_stop,
                state.non_submission_rounds,
                max_non_submission_rounds,
            )
            if state.non_submission_rounds >= max_non_submission_rounds:
                state.stop_reason = coder_stop or "coder_no_submission"
                break
            need_analysis = True
            route_diagnosis = None
            continue
        if coder_stop is not None:
            state.stop_reason = coder_stop
            break

    if state.records:
        state.records[-1]["stop_reason"] = state.stop_reason
        state.records[-1]["token_usage"] = _token_usage_summary(state)

    _write_tool_trace(output_dir, task_stem, env, state)
    return state.records, state.stop_reason


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
