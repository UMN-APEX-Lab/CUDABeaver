"""Runtime execution primitives for Cudeubgger.

This module owns candidate seeding, patch submission, build/test evaluation,
tool dispatch, artifact persistence, and trace writing.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from ..agent_tools import ToolEnv, call_tool, make_env
from ..context_manager import check_budget, get_budget_nudge, manage_context
from . import deps
from .common import (
    _LoopState,
    _accumulate_token_usage,
    _append_action_result_message,
    _chars_per_token,
    _compact_patch_action,
    _full_file_submissions,
    _sanitize_assistant_content,
    _tool_call_signature,
)
from .prompts import FIRST_TURN_DEBUG
from .records import _make_iter_record, _persist_iter_artifacts
from .submission import _handle_submission

logger = logging.getLogger("cuda_debugger_exp")


def _record_api_error_failure(
    state: _LoopState,
    task_stem: str,
    input_meta: dict,
    exc: BaseException,
    *,
    role: str = "agent",
) -> str:
    """Stamp an API failure as a task-level failed iteration record."""
    stop_reason = (
        "api_context_limit"
        if deps.is_context_length_error(exc)
        else "api_error"
    )
    state.stop_reason = stop_reason
    record = _make_iter_record(
        task_stem, input_meta, state.iteration + 1,
        five_category="compile_error", error_type=stop_reason,
        build_ok=False, status="failed",
        code_hash=None, stop_reason=stop_reason,
    )
    record["method"] = "agent"
    record["agent_role"] = role
    record["api_error_excerpt"] = str(exc)[:1200]
    state.records.append(record)
    return stop_reason


def _build_tool_env(
    task_stem: str,
    dataset_dir: Path,
    input_meta: dict,
    config: dict,
    agent_cfg: dict,
) -> ToolEnv:
    """Construct the per-task ToolEnv and wire the build/test hooks that
    close over env.current_code (so apply_edit mutations flow through)."""
    workdir = Path(config["build"]["workdir"])
    build_timeout = int(config["build"]["timeout"])
    test_timeout = int(config["build"]["test_timeout"])
    env = make_env(
        task_stem=task_stem,
        task_workdir=workdir / task_stem,
        dataset_dir=dataset_dir,
        input_meta=input_meta,
        build_timeout=build_timeout,
        test_timeout=test_timeout,
        quotas_override=agent_cfg.get("quotas") or {},
    )
    env.rebuild_fn = lambda: deps.compile_code(
        env.current_code, task_stem, dataset_dir, input_meta,
        workdir, build_timeout,
    )
    env.rerun_test_fn = lambda: deps.run_test(
        task_stem, input_meta, workdir, test_timeout,
    )
    return env


def _seed_broken_start(
    broken_start: dict,
    prompt_text: str,
    env: ToolEnv,
    state: _LoopState,
    config: dict,
    task_stem: str,
    dataset_dir: Path,
    input_meta: dict,
) -> str:
    """Load broken_start into env.current_code and state.broken_code;
    return the first-turn user message. The loop is debug-only — every
    task must start from a seeded broken candidate.

    Optionally runs a precheck (config.debug_tools.precheck.enabled): a
    fresh compile+test+sanitizer pass on the broken_start, giving the
    agent grounded failure facts instead of the curated log_excerpt.
    """
    project_root = Path(__file__).resolve().parents[2]
    code_path = Path(broken_start["code_path"])
    bs_path = code_path if code_path.is_absolute() else project_root / code_path
    env.current_code = bs_path.read_text(encoding="utf-8")
    state.broken_code = env.current_code

    excerpt = broken_start.get("log_excerpt") or ""
    fallback_section = (
        f"Error log excerpt:\n```\n{excerpt[:1500]}\n```"
        if excerpt
        else "(No error log captured for this starting code — infer from code + task spec.)"
    )

    # Optional precheck: run broken_start once to get fresh failure facts.
    excerpt_section = fallback_section
    if deps.section_enabled(config, "precheck", default=False):
        try:
            workdir = Path(config["build"]["workdir"])
            sanitizer_cfg = (config.get("debug_tools") or {}).get("sanitizers")
            precheck = deps.run_precheck(
                config=config,
                task_stem=task_stem,
                dataset_dir=dataset_dir,
                input_meta=input_meta,
                workdir=workdir,
                broken_code=env.current_code,
                sanitizer_config=sanitizer_cfg,
            )
            if precheck:
                excerpt_section = deps.format_precheck_prompt_section(
                    precheck,
                    fallback_excerpt=excerpt,
                    fallback_category=broken_start.get("error_category", "unknown"),
                )
        except Exception as e:
            logger.warning(f"  [agent] precheck failed (using fallback): {e}")

    return FIRST_TURN_DEBUG.format(
        prompt=prompt_text,
        broken_code=env.current_code,
        error_category=broken_start.get("error_category", "unknown"),
        error_excerpt=excerpt_section,
    )


def _pre_turn_context_mgmt(
    messages: list[dict],
    ctx_cfg: ContextConfig,
    summarizer,
    base_system: str,
    in_locate_phase: bool = False,
) -> None:
    """Compact context, log layer activity, and (re)attach the budget
    nudge to the system message. Mutates messages in place."""
    toks, budget_status = check_budget(messages, ctx_cfg)
    stats = manage_context(messages, ctx_cfg, summarizer=summarizer)
    if stats.layers_applied:
        extra = ""
        if summarizer is not None:
            st = getattr(summarizer, "state", {})
            extra = (
                f" [summarizer calls={st.get('calls', 0)} "
                f"hits={st.get('cache_hits', 0)} "
                f"fallback={st.get('fallbacks', 0)}]"
            )
        logger.info(
            f"  [ctx] {toks}→{stats.tokens_after} toks, "
            f"applied: {','.join(stats.layers_applied)}{extra}"
        )
    nudge = get_budget_nudge(budget_status, in_locate_phase=in_locate_phase)
    messages[0]["content"] = base_system + nudge if nudge else base_system


def _call_llm_or_record_api_error(
    client, config: dict, messages: list[dict], tools: list[dict],
    state: _LoopState, task_stem: str, input_meta: dict,
    tool_choice: str = "auto",
    max_tokens: int | str | None = None,
    temperature: float | None = None,
    role: str = "agent",
) -> dict | None:
    """Wrap call_llm_with_tools. On exception, stamp an api_error
    iteration record + set state.stop_reason and return None. Otherwise
    return the response dict. Token usage is accumulated under ``role``."""
    try:
        resp = deps.call_llm_with_tools(
            client,
            config,
            messages,
            tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        _accumulate_token_usage(
            state, role, messages, resp, _chars_per_token(config)
        )
        return resp
    except Exception as e:
        import traceback
        logger.error(f"  agent: API error: {e}\n{traceback.format_exc()}")
        _record_api_error_failure(
            state, task_stem, input_meta, e, role=role
        )
        return None


def _call_json_action_or_record_api_error(
    client, config: dict, messages: list[dict],
    state: _LoopState, task_stem: str, input_meta: dict,
    max_tokens: int | str | None = None,
    temperature: float | None = None,
    response_format: bool = True,
    role: str = "agent",
) -> dict | None:
    try:
        resp = deps.call_llm_json_action(
            client,
            config,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
        )
        _accumulate_token_usage(
            state, role, messages, resp, _chars_per_token(config)
        )
        return resp
    except Exception as e:
        import traceback
        logger.error(f"  agent: JSON action API error: {e}\n{traceback.format_exc()}")
        _record_api_error_failure(
            state, task_stem, input_meta, e, role=role
        )
        return None


def _append_assistant_message(
    messages: list[dict], resp: dict, asst_content_cap: int,
) -> None:
    """Append the assistant message with <think>-stripped content and the
    tool_calls (when present, so the backend can match them next turn)."""
    asst_msg: dict = {
        "role": "assistant",
        "content": _sanitize_assistant_content(resp["content"], max_chars=asst_content_cap),
    }
    if resp["tool_calls"]:
        asst_msg["tool_calls"] = resp["tool_calls"]
    messages.append(asst_msg)


def _handle_no_op_turn(state: _LoopState, messages: list[dict]) -> bool:
    """Model returned no tool call. First time: prod it. Second time:
    abort the task. Returns True if the main loop should break."""
    state.consecutive_no_op_turns += 1
    if state.consecutive_no_op_turns >= 2:
        logger.warning("  agent: 2 consecutive no-op turns, aborting task")
        state.stop_reason = "agent_no_op"
        return True
    messages.append({
        "role": "user",
        "content": (
            "Please continue with a tool call. If you believe the candidate "
            "is ready, call `write_solution`. If you need more info, call "
            "a read_* or run_* tool."
        ),
    })
    return False


def _handle_locate_bug(
    tc: dict, result, state: _LoopState, messages: list[dict],
    json_action: bool = False,
) -> None:
    """Record the localisation claim and flip the loop into FIX phase
    (apply_edit / write_solution appear in the next tool list)."""
    state.located_this_iteration = True
    state.pending_location = {
        "file": result.meta["file"],
        "line": result.meta["line"],
        "bug_description": result.meta["bug_description"],
        "evidence": result.meta["evidence"],
    }
    if json_action:
        _append_action_result_message(
            messages, tc["function"]["name"], result.to_message_content(), result.ok
        )
    else:
        messages.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": result.to_message_content(),
        })


def _dispatch_tool_calls(
    tool_calls: list[dict],
    env: ToolEnv,
    state: _LoopState,
    messages: list[dict],
    max_tool_calls: int,
    task_stem: str,
    dataset_dir: Path,
    input_meta: dict,
    config: dict,
    output_dir: Path,
    client,
    prompt_text: str,
    json_action: bool = False,
) -> str | None:
    """Dispatch each tool call in a turn. Returns a stop_reason that
    should terminate the main loop, or None to continue."""
    for tc in tool_calls:
        if state.tool_calls_used >= max_tool_calls:
            exhausted = (
                f"[tool budget exhausted: "
                f"{state.tool_calls_used}/{max_tool_calls}]"
            )
            if json_action:
                _append_action_result_message(
                    messages, tc["function"]["name"], exhausted, ok=False
                )
            else:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": exhausted,
                })
            return "tool_budget_exhausted"

        state.tool_calls_used += 1
        name = tc["function"]["name"]
        args_str = tc["function"]["arguments"]
        try:
            parsed_args = json.loads(args_str) if isinstance(args_str, str) and args_str else {}
        except json.JSONDecodeError:
            parsed_args = {}

        full_file_fallback_submit = (
            name == "write_solution"
            and isinstance(parsed_args.get("code"), str)
            and bool(parsed_args.get("code", "").strip())
        )
        if _full_file_submissions(config) and name == "apply_edit":
            result_text = (
                "[action rejected] This run requires full-file code submissions. "
                "Do not use `apply_edit`. Call `write_solution` with a complete "
                "revised `code` file containing the whole solution."
            )
            if json_action:
                _append_action_result_message(messages, name, result_text, ok=False)
            else:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_text,
                })
            continue
        if (
            _full_file_submissions(config)
            and name == "write_solution"
            and not full_file_fallback_submit
        ):
            result_text = (
                "[action rejected] This run requires full-file code submissions. "
                "`write_solution` must include a non-empty `code` argument with "
                "the complete revised solution file, not only a rationale."
            )
            if json_action:
                _append_action_result_message(messages, name, result_text, ok=False)
            else:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_text,
                })
            continue
        if state.force_read_code_reason and name != "read_code" and not full_file_fallback_submit:
            result_text = (
                "[action rejected] The previous apply_edit failed because its "
                "old_string did not match the current candidate. You must call "
                "`read_code` first with `{ \"full_file\": true }`, then create "
                "a new exact old_string from that fresh full-file code. "
                "Alternatively, call `write_solution` with a complete revised "
                "`code` file as a full-file fallback."
            )
            if json_action:
                _append_action_result_message(messages, name, result_text, ok=False)
            else:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_text,
                })
            continue
        if (
            state.force_read_code_reason
            and name == "read_code"
            and state.force_read_code_full_file
            and not bool(parsed_args.get("full_file"))
        ):
            result_text = (
                "[action rejected] A range read is not enough after "
                "`apply_edit` old_string mismatch. Call `read_code` with "
                "`{ \"full_file\": true }` so the next `old_string` can be "
                "copied from the complete current candidate."
            )
            if json_action:
                _append_action_result_message(messages, name, result_text, ok=False)
            else:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_text,
                })
            continue

        signature = _tool_call_signature(name, args_str)
        result = call_tool(name, args_str, env, iteration=state.iteration + 1)

        if name == "read_code" and result.ok and state.force_read_code_reason:
            if (
                not state.force_read_code_full_file
                or bool(result.meta.get("full_file"))
            ):
                state.force_read_code_reason = None
                state.force_read_code_full_file = False

        if name == "locate_bug" and result.ok and result.meta.get("located"):
            _handle_locate_bug(tc, result, state, messages, json_action=json_action)
            continue

        if name == "write_solution" and result.ok and result.meta.get("submit"):
            if result.meta.get("full_replacement"):
                state.applied_edit_this_iteration = True
                state.force_read_code_reason = None
                state.force_read_code_full_file = False
                state.patch_actions_this_iteration.append(
                    _compact_patch_action(name, parsed_args, result)
                )
            stop = _handle_submission(
                tc, result, state, env, messages,
                task_stem, dataset_dir, input_meta, config, output_dir,
                client, prompt_text, json_action=json_action,
            )
            if stop is not None:
                return stop
            continue

        if name == "apply_edit" and result.ok and result.meta.get("edited"):
            state.applied_edit_this_iteration = True
            state.patch_actions_this_iteration.append(
                _compact_patch_action(name, parsed_args, result)
            )
        elif name == "apply_edit" and not result.ok:
            state.patch_actions_this_iteration.append(
                _compact_patch_action(name, parsed_args, result)
            )
            if result.error and "old_string not found" in result.error:
                count = state.failed_action_counts.get(signature, 0) + 1
                state.failed_action_counts[signature] = count
                if count >= 3:
                    state.force_read_code_reason = (
                        "`apply_edit` failed repeatedly with the same "
                        "old_string. Read the full current file, abandon this "
                        "exact edit, and construct a different minimal patch "
                        "from fresh full-file text."
                    )
                else:
                    state.force_read_code_reason = (
                        "`apply_edit` failed: old_string was not found in the "
                        "current candidate."
                    )
                state.force_read_code_full_file = True

        # Regular tool — relay the result back to the model.
        if json_action:
            _append_action_result_message(
                messages, name, result.to_message_content(), result.ok
            )
        else:
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result.to_message_content(),
            })
    return None


def _auto_submit_after_apply_edit(
    env: ToolEnv,
    state: _LoopState,
    messages: list[dict],
    max_tool_calls: int,
    task_stem: str,
    dataset_dir: Path,
    input_meta: dict,
    config: dict,
    output_dir: Path,
    client,
    prompt_text: str,
    json_action: bool = False,
) -> str | None:
    """Submit the staged patch when coder edited but omitted write_solution."""
    if _full_file_submissions(config):
        return None
    if not state.applied_edit_this_iteration:
        return None
    logger.info("  [blackboard coder] submit guard: auto write_solution after apply_edit")
    auto_tc = {
        "id": f"auto_write_solution_{state.iteration + 1}",
        "function": {
            "name": "write_solution",
            "arguments": json.dumps({
                "rationale": "submit guard after successful apply_edit",
            }),
        },
    }
    return _dispatch_tool_calls(
        [auto_tc], env, state, messages,
        max_tool_calls,
        task_stem, dataset_dir, input_meta, config, output_dir,
        client, prompt_text,
        json_action=json_action,
    )


def _write_tool_trace(
    output_dir: Path,
    task_stem: str,
    env: ToolEnv,
    state: _LoopState,
    *,
    stop_reason: str | None = None,
) -> None:
    trace_dir = output_dir / "tool_trace"
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f"{task_stem}.jsonl"
    with open(trace_path, "w", encoding="utf-8") as f:
        for entry in env.trace:
            f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
    display_stop = stop_reason or state.stop_reason
    logger.info(
        f"  [agent] {len(state.records)} iters, {len(env.trace)} tool calls, "
        f"stop={display_stop}, trace={trace_path}"
    )
