"""Legacy ReAct-style workflow entry point for Cudeubgger."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from ..agent_tools import schemas_for_config
from .blackboard import (
    run_task_blackboard_agent,
    run_task_multi_role_no_blackboard_agent,
)
from .common import (
    _LoopState,
    _action_to_tool_call,
    _append_action_result_message,
    _build_ctx_config,
    _extract_json_action,
    _forced_action_context,
    _full_file_submissions,
    _json_action_request,
    _json_action_enabled,
    _maybe_make_summarizer,
    _max_patch_submissions,
    _patch_submission_limit_stop_reason,
    _sanitize_assistant_content,
    _token_usage_summary,
    _tool_decision_overrides,
    _tool_names,
    _tools_for_phase,
)
from .prompts import AGENT_SYSTEM_PROMPT, JSON_ACTION_PROTOCOL_PROMPT
from .runtime import (
    _append_assistant_message,
    _build_tool_env,
    _call_json_action_or_record_api_error,
    _call_llm_or_record_api_error,
    _dispatch_tool_calls,
    _handle_no_op_turn,
    _pre_turn_context_mgmt,
    _seed_broken_start,
    _write_tool_trace,
)
from ..task_registry import load_input_meta, load_prompt_text

logger = logging.getLogger("cuda_debugger_exp")

def run_task_agent(
    task_stem: str,
    dataset_dir: Path,
    output_dir: Path,
    config: dict,
    client,
    broken_start: dict | None = None,
) -> tuple[list[dict], str]:
    """Run the agent loop for a single task. Returns (records, stop_reason).

    DEBUG-ONLY: requires `broken_start`. env.current_code is seeded from the
    broken candidate before iter 1. By default every revision goes through
    locate_bug -> write_solution(code=complete_file). Legacy patch mode can
    still use locate_bug -> apply_edit -> write_solution(). Generation mode
    is not supported and raises ValueError.
    """
    logger.info(f"=== [agent] Starting task: {task_stem} ===")
    workflow = str((config.get("agent", {}) or {}).get("workflow", "react")).lower()
    if workflow in {"blackboard", "multi_agent", "multi-agent"}:
        return run_task_blackboard_agent(
            task_stem, dataset_dir, output_dir, config, client,
            broken_start=broken_start,
        )
    if workflow in {
        "multi_role_no_blackboard",
        "multi-role-no-blackboard",
        "no_blackboard",
        "no-blackboard",
    }:
        return run_task_multi_role_no_blackboard_agent(
            task_stem, dataset_dir, output_dir, config, client,
            broken_start=broken_start,
        )

    if not broken_start:
        raise ValueError(
            "run_task_agent requires `broken_start` — the agent loop is "
            "debug-only (no generation mode). Run with --mode debug and "
            "--v2-task-list to supply broken_start entries."
        )

    input_meta = load_input_meta(task_stem, dataset_dir)
    prompt_text = load_prompt_text(task_stem, dataset_dir)

    agent_cfg = config.get("agent", {}) or {}
    ctx_cfg = _build_ctx_config(agent_cfg)
    env = _build_tool_env(task_stem, dataset_dir, input_meta, config, agent_cfg)
    summarizer = _maybe_make_summarizer(agent_cfg, client, config)

    max_patch_submissions = _max_patch_submissions(config)
    max_tool_calls = int(agent_cfg.get("max_tool_calls_per_task", 40))
    asst_content_cap = int(agent_cfg.get("assistant_content_cap", 2000))
    allowed_tools: list[str] | None = agent_cfg.get("tools")
    full_file_mode = _full_file_submissions(config)
    if full_file_mode and allowed_tools:
        allowed_tools = [t for t in allowed_tools if t != "apply_edit"]
    tool_choice = str(agent_cfg.get("tool_choice", "auto"))
    use_json_action = _json_action_enabled(config)
    json_response_format = bool(agent_cfg.get("json_action_response_format", True))
    tool_decision_max_tokens, tool_decision_temperature = (
        _tool_decision_overrides(agent_cfg)
    )
    base_system = (
        AGENT_SYSTEM_PROMPT + "\n\n" + JSON_ACTION_PROTOCOL_PROMPT
        if use_json_action
        else AGENT_SYSTEM_PROMPT
    )

    state = _LoopState()
    state.stop_reason = _patch_submission_limit_stop_reason(config)
    first_user = _seed_broken_start(
        broken_start, prompt_text, env, state,
        config, task_stem, dataset_dir, input_meta,
    )
    messages: list[dict] = [
        {"role": "system", "content": base_system},
        {"role": "user", "content": first_user},
    ]

    while state.iteration < max_patch_submissions:
        in_locate_phase = (
            bool(env.current_code.strip()) and not state.located_this_iteration
        )
        _pre_turn_context_mgmt(
            messages, ctx_cfg, summarizer, base_system,
            in_locate_phase=in_locate_phase,
        )

        in_locate_phase = (
            bool(env.current_code.strip()) and not state.located_this_iteration
        )
        tools = (
            schemas_for_config(
                ["read_code", "write_solution"] if full_file_mode else ["read_code"],
                require_full_solution_code=full_file_mode,
            )
            if state.force_read_code_reason
            else _tools_for_phase(
                allowed_tools,
                in_locate_phase,
                require_full_solution_code=full_file_mode,
            )
        )
        request_max_tokens = tool_decision_max_tokens
        request_temperature = tool_decision_temperature

        t0 = time.time()
        if use_json_action:
            action_request = _json_action_request(
                tools,
                in_locate_phase,
                extra_context=_forced_action_context(state),
            )
            request_messages = messages + [
                {"role": "user", "content": action_request}
            ]
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
            )
            if resp is None:
                break
            raw_content = resp.get("content") or ""
            resolved_max_tokens = resp.get("request_max_tokens", request_max_tokens)
            cap_note = (
                f", max_tokens={resolved_max_tokens}"
                if resolved_max_tokens is not None
                else ""
            )
            logger.info(
                f"  agent turn {len(env.trace) + 1}: api={time.time() - t0:.1f}s, "
                f"json_action=yes, content={'yes' if raw_content else 'none'}, "
                f"finish={resp.get('finish_reason')}{cap_note}"
            )
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
                    call_id=f"json_action_{len(env.trace) + 1}",
                )
            except ValueError as e:
                state.consecutive_no_op_turns += 1
                _append_action_result_message(
                    messages,
                    "parse_json_action",
                    f"[invalid JSON action] {e}\nRaw response:\n{raw_content[:1000]}",
                    ok=False,
                )
                if state.consecutive_no_op_turns >= 2:
                    logger.warning("  agent: 2 consecutive invalid JSON actions, aborting task")
                    state.stop_reason = "agent_action_parse_error"
                    break
                continue

            state.consecutive_no_op_turns = 0
            stop = _dispatch_tool_calls(
                [tc], env, state, messages,
                max_tool_calls, task_stem, dataset_dir, input_meta,
                config, output_dir, client, prompt_text,
                json_action=True,
            )
            if stop is not None:
                state.stop_reason = stop
                break
            continue

        resp = _call_llm_or_record_api_error(
            client, config, messages, tools, state, task_stem, input_meta,
            tool_choice=tool_choice,
            max_tokens=request_max_tokens,
            temperature=request_temperature,
        )
        if resp is None:
            break  # api_error path already stamped record + stop_reason
        resolved_max_tokens = resp.get("request_max_tokens", request_max_tokens)
        cap_note = (
            f", max_tokens={resolved_max_tokens}"
            if resolved_max_tokens is not None
            else ""
        )
        logger.info(
            f"  agent turn {len(env.trace) + 1}: api={time.time() - t0:.1f}s, "
            f"tool_calls={len(resp['tool_calls'])}, "
            f"content={'yes' if resp['content'] else 'none'}"
            f"{cap_note}"
        )

        _append_assistant_message(messages, resp, asst_content_cap)

        if not resp["tool_calls"]:
            if _handle_no_op_turn(state, messages):
                break
            continue
        state.consecutive_no_op_turns = 0

        stop = _dispatch_tool_calls(
            resp["tool_calls"], env, state, messages,
            max_tool_calls,
            task_stem, dataset_dir, input_meta, config, output_dir,
            client, prompt_text,
        )
        if stop is not None:
            state.stop_reason = stop
            break
        if state.iteration >= max_patch_submissions:
            state.stop_reason = _patch_submission_limit_stop_reason(config)
            break

    if state.records:
        state.records[-1]["stop_reason"] = state.stop_reason
        state.records[-1]["token_usage"] = _token_usage_summary(state)

    _write_tool_trace(output_dir, task_stem, env, state)
    return state.records, state.stop_reason


# ---------------------------------------------------------------------------
# Record builder — mirror of run_experiment._make_record
# ---------------------------------------------------------------------------
