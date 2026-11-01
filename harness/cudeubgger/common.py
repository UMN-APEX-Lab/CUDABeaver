"""Shared protocol, state, and configuration helpers for Cudeubgger."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..agent_tools import schemas_for_config
from ..context_manager import ContextConfig, estimate_tokens
from ..context_summary import make_summarizer
from .prompts import ACTION_RESULT_TEMPLATE, JSON_ACTION_OUTPUT_DISCIPLINE

logger = logging.getLogger("cuda_debugger_exp")

_THINK_TAG_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)
# Some vLLM-served reasoning models (e.g. gemma-4 hybrid) leak the thinking
# into `content` (reasoning_content empty) wrapped in channel markers like
# `<|channel>thought ... <channel|>` instead of <think> tags.
_CHANNEL_THOUGHT_RE = re.compile(
    r"<\|?channel\|?>\s*thought\b.*?<\|?channel\|?>", re.DOTALL | re.IGNORECASE
)


def _strip_thinking(text: str) -> str:
    """Remove leaked thinking blocks (both <think>…</think> and
    <|channel>thought…<channel|>) from model content."""
    text = _THINK_TAG_RE.sub("", text)
    text = _CHANNEL_THOUGHT_RE.sub("", text)
    return text


def _sanitize_assistant_content(content, max_chars: int = 2000):
    """Strip <think> blocks from assistant content and cap length.

    The thinking chain is the model's scratchpad — it must NOT be sent
    back to the model on the next turn (it would balloon context and
    confuse the model into "continuing" stale reasoning). After stripping
    we cap the remaining text — visible content over a couple thousand
    chars is almost always leaked reasoning, not a deliberate response.
    Returns None when nothing useful is left.
    """
    if not content:
        return content
    original_len = len(content)
    stripped = _strip_thinking(content).strip()
    if not stripped:
        return None
    if len(stripped) > max_chars:
        half = max_chars // 2
        stripped = (
            stripped[:half] + "\n... [content truncated] ...\n" + stripped[-half:]
        )
    if len(stripped) != original_len:
        logger.info(
            f"  [agent] sanitised assistant content: {original_len} → {len(stripped)} chars"
        )
    return stripped


# Tools that exist only in the FIX phase (after a bug is located).
# `apply_edit` is a patching action; `write_solution` is a submission.
# Both modify the candidate, so both must wait for locate_bug.
_FIX_PHASE_TOOLS = {"apply_edit", "write_solution"}


def _full_file_submissions(config: dict) -> bool:
    """Whether code-changing submissions must replace the whole solution file."""
    agent_cfg = config.get("agent", {}) or {}
    bb_cfg = agent_cfg.get("blackboard") or {}
    if "full_file_submissions" in bb_cfg:
        return bool(bb_cfg.get("full_file_submissions"))
    return bool(agent_cfg.get("full_file_submissions", True))


def _tools_for_phase(
    allowed_tools: list[str] | None,
    in_locate_phase: bool,
    *,
    require_full_solution_code: bool = False,
) -> list[dict]:
    """Tool schemas for the current phase.

    In the LOCATE phase the candidate-modifying tools (`apply_edit` and
    `write_solution`) are withheld entirely — the model cannot patch or
    submit a fix until it has called `locate_bug`. This makes "locate
    first" a structural property of the loop rather than something
    enforced reactively by rejecting un-located submissions.
    """
    if require_full_solution_code:
        allowed_tools = (
            [t for t in allowed_tools if t != "apply_edit"]
            if allowed_tools
            else None
        )
    schemas = schemas_for_config(
        allowed_tools,
        require_full_solution_code=require_full_solution_code,
    )
    if require_full_solution_code:
        schemas = [s for s in schemas if s["function"]["name"] != "apply_edit"]
    if in_locate_phase:
        return [s for s in schemas if s["function"]["name"] not in _FIX_PHASE_TOOLS]
    return schemas


def _tool_names(tool_schemas: list[dict]) -> set[str]:
    return {s["function"]["name"] for s in tool_schemas}


def _compact_tool_schema(tool_schemas: list[dict]) -> str:
    compact: list[dict] = []
    for schema in tool_schemas:
        fn = schema["function"]
        params = fn.get("parameters") or {}
        props = params.get("properties") or {}
        compact.append({
            "tool": fn["name"],
            "description": (fn.get("description") or "")[:220],
            "required_args": params.get("required") or [],
            "args": {
                key: {
                    k: (v[:140] if k == "description" and isinstance(v, str) else v)
                    for k, v in spec.items()
                    if k in {"type", "enum", "minimum", "description"}
                }
                for key, spec in props.items()
            },
        })
    return json.dumps(compact, ensure_ascii=False)


def _json_action_request(
    tool_schemas: list[dict],
    in_locate_phase: bool,
    extra_context: str | None = None,
) -> str:
    phase = "LOCATE" if in_locate_phase else "FIX"
    context_block = f"\n\nDecision constraints:\n{extra_context}" if extra_context else ""
    return (
        f"ACTION REQUEST ({phase} phase)\n"
        "Return exactly one JSON object matching this schema:\n"
        '{"tool":"one_available_tool_name","args":{}}\n\n'
        "Available tools for this turn:\n"
        f"{_compact_tool_schema(tool_schemas)}\n\n"
        "Choose the single next action. If the previous action failed, you "
        "must change strategy based on the failure result instead of repeating "
        "the same action."
        f"{context_block}\n"
        "No prose. No markdown.\n\n"
        f"{JSON_ACTION_OUTPUT_DISCIPLINE}"
    )


def _json_action_enabled(config: dict) -> bool:
    action_protocol = str(
        (config.get("agent", {}) or {}).get("action_protocol", "native")
    ).lower()
    return action_protocol in {"json", "json_action", "action_json"}


def _max_patch_submissions(config: dict) -> int:
    agent_cfg = config.get("agent", {}) or {}
    for key in ("max_patch_submissions", "max_patch_attempts", "max_submissions"):
        value = agent_cfg.get(key)
        if value is not None:
            return int(value)
    return int((config.get("experiment", {}) or {}).get("max_iterations", 1))


def _patch_submission_limit_stop_reason(config: dict) -> str:
    agent_cfg = config.get("agent", {}) or {}
    if any(agent_cfg.get(k) is not None for k in (
        "max_patch_submissions",
        "max_patch_attempts",
        "max_submissions",
    )):
        return "max_patch_submissions"
    return "max_iterations"


def _scoped_views_enabled(config: dict) -> bool:
    """Whether blackboard roles receive role-scoped board projections.

    Default False preserves the legacy behavior where every role sees the
    full board dump. When True, each role gets only the slice it needs
    (see ``ErrorBlackboard.view_for``) — primarily to cut the coder's
    redundant token cost.
    """
    bb_cfg = (config.get("agent", {}) or {}).get("blackboard") or {}
    return bool(bb_cfg.get("scoped_views", False)) if isinstance(bb_cfg, dict) else False


def _structured_reports_enabled(config: dict) -> bool:
    """Whether analyzer/executor finish with a structured report object
    instead of free-text prose.

    Default False keeps the legacy free-text ``final_report``. When True,
    each role emits a bounded ``{summary, key_points, ruled_out}`` object
    (see ``_normalize_structured_report``) — smaller and machine-consumable
    for the verifier. Only takes effect on the JSON-action protocol.
    """
    bb_cfg = (config.get("agent", {}) or {}).get("blackboard") or {}
    return (
        bool(bb_cfg.get("structured_reports", False))
        if isinstance(bb_cfg, dict)
        else False
    )


def _normalize_structured_report(raw) -> dict:
    """Coerce a role's ``final_report`` payload into a bounded dict.

    Defensive on purpose: the summary and list sizes are capped regardless
    of what the model emitted, so a verbose role cannot bloat downstream
    context. A non-dict payload (model ignored the contract) degrades to a
    summary-only report.
    """
    def _bounded_list(value, max_items: int, max_chars: int) -> list[str]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        return [_truncate_text(str(item), max_chars) for item in value[:max_items]]

    if not isinstance(raw, dict):
        return {
            "summary": _truncate_text(str(raw or ""), 400),
            "key_points": [],
            "ruled_out": [],
        }
    return {
        "summary": _truncate_text(str(raw.get("summary") or ""), 400),
        "key_points": _bounded_list(raw.get("key_points"), 4, 200),
        "ruled_out": _bounded_list(raw.get("ruled_out"), 4, 200),
    }


def _chars_per_token(config: dict) -> float:
    agent_cfg = config.get("agent", {}) or {}
    try:
        return float(agent_cfg.get("chars_per_token", 4.0))
    except (TypeError, ValueError):
        return 4.0


def _usage_field(usage, key: str) -> int:
    if usage is None:
        return 0
    if isinstance(usage, dict):
        value = usage.get(key)
    else:
        value = getattr(usage, key, None)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _accumulate_token_usage(
    state: "_LoopState",
    role: str,
    request_messages: list[dict],
    resp=None,
    chars_per_token: float = 4.0,
) -> None:
    """Accumulate one LLM call's token cost into ``state.token_usage[role]``.

    Always records an estimated prompt-token count from the request
    messages (chars/4) so the metric is consistent across endpoints. When
    ``resp`` carries a ``usage`` object/dict, the real prompt/completion/total
    counts are added too.
    """
    bucket = state.token_usage.setdefault(
        role,
        {
            "calls": 0,
            "est_prompt_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    )
    bucket["calls"] += 1
    try:
        bucket["est_prompt_tokens"] += estimate_tokens(
            request_messages, max(chars_per_token, 1.0)
        )
    except Exception:  # estimation must never break the loop
        pass
    usage = resp.get("usage") if isinstance(resp, dict) else None
    if usage is not None:
        bucket["prompt_tokens"] += _usage_field(usage, "prompt_tokens")
        bucket["completion_tokens"] += _usage_field(usage, "completion_tokens")
        bucket["total_tokens"] += _usage_field(usage, "total_tokens")


def _token_usage_summary(state: "_LoopState") -> dict:
    """Per-role usage plus a grand total, for the manifest record."""
    totals = {
        "calls": 0,
        "est_prompt_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    for bucket in state.token_usage.values():
        for key in totals:
            totals[key] += int(bucket.get(key, 0) or 0)
    return {"per_role": dict(state.token_usage), "total": totals}


def _hard_stop_on_stagnation(config: dict) -> bool:
    """Whether agent stagnation signals should terminate the task early.

    Agent debugging is governed by patch submission budget. Stagnation
    signals are still useful telemetry, but treating same-category failures
    as a hard stop is too aggressive for logic errors where the assertion can
    stay identical while the localization hypothesis improves.
    """
    agent_cfg = config.get("agent", {}) or {}
    return bool(agent_cfg.get("hard_stop_stagnation", False))


def _max_coder_non_submission_rounds(config: dict) -> int:
    agent_cfg = config.get("agent", {}) or {}
    bb_cfg = agent_cfg.get("blackboard") or {}
    value = (
        bb_cfg.get("max_coder_non_submission_rounds")
        if isinstance(bb_cfg, dict)
        else None
    )
    if value is None:
        value = agent_cfg.get("max_coder_non_submission_rounds", 3)
    return max(1, int(value))


def _is_recoverable_coder_stop(reason: str | None) -> bool:
    return reason in {
        "coder_no_submission",
        "coder_no_op",
        "coder_action_parse_error",
        "agent_repeated_failed_action",
    }


def _record_coder_non_submission(state: _LoopState, reason: str) -> None:
    state.non_submission_rounds += 1
    state.attempt_history.append({
        "iteration": state.iteration + 1,
        "rationale": "coder did not submit a patch",
        "located": state.pending_location,
        "patch_actions": list(state.patch_actions_this_iteration),
        "classification": {
            "status": "not_submitted",
            "category": reason,
            "five_category": "agent_workflow",
            "error_type": reason,
        },
    })
    state.located_this_iteration = False
    state.pending_location = None
    state.applied_edit_this_iteration = False
    state.patch_actions_this_iteration = []


def _extract_json_action(text: str | None) -> dict:
    if not text or not text.strip():
        raise ValueError("empty JSON action")
    raw = _strip_thinking(text).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(raw):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(raw[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("no JSON object found")


def _action_to_tool_call(action: dict, allowed_tool_names: set[str], call_id: str) -> dict:
    tool = action.get("tool") or action.get("name") or action.get("action")
    if not isinstance(tool, str) or not tool:
        raise ValueError("JSON action missing string 'tool'")
    if tool not in allowed_tool_names:
        allowed = ", ".join(sorted(allowed_tool_names))
        raise ValueError(f"tool {tool!r} is not available in this phase; allowed: {allowed}")
    args = action.get("args", {})
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ValueError("JSON action 'args' must be an object")
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": tool,
            "arguments": json.dumps(args, ensure_ascii=False),
        },
    }


def _append_action_result_message(
    messages: list[dict],
    tool_name: str,
    content: str,
    ok: bool = True,
) -> None:
    messages.append({
        "role": "user",
        "content": ACTION_RESULT_TEMPLATE.format(
            tool=tool_name,
            ok=str(ok).lower(),
            body=content,
        ),
    })


# ---------------------------------------------------------------------------
# Loop state + helpers
#
# `run_task_agent` (below) is intentionally thin — it just sequences these.
# All mutable per-task state lives on `_LoopState`; the helpers either
# return a value (response, stop_reason hint) or mutate state / messages /
# env in place.
# ---------------------------------------------------------------------------


@dataclass
class _LoopState:
    """Mutable per-task state carried through the agent loop."""

    iteration: int = 0
    tool_calls_used: int = 0
    consecutive_no_op_turns: int = 0
    located_this_iteration: bool = False
    pending_location: dict | None = None
    stop_reason: str = "max_iterations"
    records: list[dict] = field(default_factory=list)
    stagnation_history: list = field(default_factory=list)
    applied_edit_this_iteration: bool = False
    # Original broken_start code, captured at seed time and never mutated.
    # Used by debug-assist hooks (patch-scope analysis, failed-patch
    # verifier) to compare candidate against baseline.
    broken_code: str = ""
    patch_actions_this_iteration: list[dict] = field(default_factory=list)
    attempt_history: list[dict] = field(default_factory=list)
    failed_action_counts: dict[str, int] = field(default_factory=dict)
    force_read_code_reason: str | None = None
    force_read_code_full_file: bool = False
    non_submission_rounds: int = 0
    # The single fix direction the verifier handed the coder for the current
    # attempt; logged into the tried-strategy registry on submission (gap 6).
    current_fix_direction: str | None = None
    # Cumulative per-role token usage for the whole task. Keyed by role name
    # ("analyzer"/"executor"/"verifier"/"coder"/"agent"); each value is a dict
    # of {calls, est_prompt_tokens, prompt_tokens, completion_tokens,
    # total_tokens}. `est_prompt_tokens` is always populated (chars/4 estimate
    # of the request); the real `*_tokens` are filled in only when the endpoint
    # returns a usage object.
    token_usage: dict = field(default_factory=dict)


@dataclass
class ErrorBlackboard:
    """Shared evidence board for the blackboard workflow.

    The board is intentionally plain JSON-serialisable data: analyzer,
    executor, verifier, and coder all see the same object rendered into
    their prompts, and the final manifest record stores the same shape for
    later audit.
    """

    task_stem: str
    suspected_lines: list[dict] = field(default_factory=list)
    error_type: str | None = None
    static_flags: list[dict] = field(default_factory=list)
    runtime_trace: dict = field(default_factory=dict)
    # analyzer/executor reports are free-text str by default, or a structured
    # {summary, key_points, ruled_out} dict when structured_reports is on.
    analyzer_report: str | dict | None = None
    executor_report: str | dict | None = None
    verifier_report: str | None = None
    fix_suggestions: list[str] = field(default_factory=list)
    # The single concrete fix direction the verifier hands the coder (gap 4).
    next_fix: str | None = None

    def add_suspected_line(self, line: dict) -> None:
        key = (
            line.get("source"),
            line.get("file"),
            line.get("line"),
            line.get("bug_description") or line.get("root_cause"),
        )
        for existing in self.suspected_lines:
            ekey = (
                existing.get("source"),
                existing.get("file"),
                existing.get("line"),
                existing.get("bug_description") or existing.get("root_cause"),
            )
            if ekey == key:
                return
        self.suspected_lines.append(line)

    def to_dict(self) -> dict:
        return {
            "task_stem": self.task_stem,
            "error_type": self.error_type,
            "suspected_lines": self.suspected_lines,
            "static_flags": self.static_flags,
            "runtime_trace": self.runtime_trace,
            "analyzer_report": self.analyzer_report,
            "executor_report": self.executor_report,
            "verifier_report": self.verifier_report,
            "fix_suggestions": self.fix_suggestions,
            "next_fix": self.next_fix,
        }

    def to_prompt_block(self, max_chars: int | None = 14000) -> str:
        text = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        if max_chars is None or int(max_chars) <= 0:
            return text
        return _truncate_text(text, int(max_chars))

    # -- role-scoped projections (gated by agent.blackboard.scoped_views) --

    def _compact_static_flags(self) -> list[dict]:
        return [
            {k: f.get(k) for k in ("source", "file", "line", "flag", "severity")}
            for f in self.static_flags
        ]

    def _runtime_summary(self, excerpt_chars: int = 800) -> dict:
        """runtime_trace with tool-result excerpts trimmed (keeps evidence,
        sheds the bulk). Used by the verifier view."""
        summary = {
            "initial_observation": self.runtime_trace.get("initial_observation"),
            "tool_results": [
                {
                    "source": r.get("source"),
                    "tool": r.get("tool"),
                    "ok": r.get("ok"),
                    "error": r.get("error"),
                    "meta": r.get("meta"),
                    "content_excerpt": _truncate_text(
                        r.get("content_excerpt") or "", excerpt_chars
                    ),
                }
                for r in (self.runtime_trace.get("tool_results") or [])
            ],
        }
        return summary

    def _verified_suspects(self) -> list[dict]:
        """Verifier-confirmed suspected lines, falling back to all suspects
        if the verifier produced none (e.g. it failed to emit valid JSON)."""
        verified = [
            s for s in self.suspected_lines if s.get("source") == "verifier"
        ]
        return verified or self.suspected_lines

    def _latest_runtime(self) -> dict | None:
        """The most recent build/test/sanitizer result, or the initial
        observation if no runtime tool ran. Coder only needs the latest."""
        results = self.runtime_trace.get("tool_results") or []
        if results:
            r = results[-1]
            return {
                "tool": r.get("tool"),
                "ok": r.get("ok"),
                "error": r.get("error"),
                "content_excerpt": _truncate_text(r.get("content_excerpt") or "", 1500),
            }
        return self.runtime_trace.get("initial_observation")

    def view_for(self, role: str, max_chars: int | None = 14000) -> str:
        """Render only the slice of the board a given role needs.

        Unknown roles fall back to the full dump. The coder view is
        deliberately strict: only the verified bug report, fix suggestions,
        and the latest runtime result — not the analyzer/executor/verifier
        free-text reports, raw static flags, or full runtime trace.
        """
        if role == "executor":
            view = {
                "task_stem": self.task_stem,
                "error_type": self.error_type,
                "suspected_lines": self.suspected_lines,
                "static_flags": self._compact_static_flags(),
                "initial_observation": self.runtime_trace.get("initial_observation"),
            }
        elif role == "verifier":
            view = {
                "task_stem": self.task_stem,
                "error_type": self.error_type,
                "suspected_lines": self.suspected_lines,
                "static_flags": self.static_flags,
                "runtime_trace": self._runtime_summary(),
                "analyzer_report": self.analyzer_report,
                "executor_report": self.executor_report,
            }
        elif role == "coder":
            view = {
                "task_stem": self.task_stem,
                "error_type": self.error_type,
                "suspected_lines": self._verified_suspects(),
                "next_fix": self.next_fix,
                "latest_runtime": self._latest_runtime(),
            }
        else:
            return self.to_prompt_block(max_chars)
        text = json.dumps(view, indent=2, ensure_ascii=False)
        if max_chars is None or int(max_chars) <= 0:
            return text
        return _truncate_text(text, int(max_chars))


@dataclass
class _RoleResult:
    # str for free-text reports; dict for structured reports
    # ({summary, key_points, ruled_out}) when structured_reports is on.
    report: str | dict = ""
    stop_reason: str | None = None


def _truncate_text(text: str | None, max_chars: int) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n... [truncated] ...\n" + text[-half:]


def _compact_patch_action(name: str, args: dict, result) -> dict:
    return {
        "tool": name,
        "args": {
            k: _truncate_text(v, 600) if isinstance(v, str) else v
            for k, v in (args or {}).items()
        },
        "ok": bool(result.ok),
        "error": result.error,
        "content_excerpt": _truncate_text(result.content, 1000),
    }


def _tool_call_signature(name: str, args_str: str) -> str:
    try:
        args = json.loads(args_str) if args_str else {}
    except json.JSONDecodeError:
        args = args_str
    return json.dumps({"tool": name, "args": args}, sort_keys=True, ensure_ascii=False)


def _attempt_history_prompt(state: _LoopState, max_items: int = 3) -> str:
    if not state.attempt_history:
        return "[]"
    return _truncate_text(
        json.dumps(state.attempt_history[-max_items:], indent=2, ensure_ascii=False),
        10000,
    )


def _tried_strategies_prompt(state: _LoopState) -> str | None:
    """Render an explicit registry of already-attempted (cause, fix) pairs so
    the verifier can avoid repeating a failed strategy (gap 6).

    Unlike the same-signature ``_repeated_failure_alert``, this lists *every*
    failed attempt's believed root cause + fix direction + outcome, regardless
    of whether failure signatures match.
    """
    entries = []
    for attempt in state.attempt_history:
        cls = attempt.get("classification") or {}
        if cls.get("status") == "passed" or cls.get("five_category") == "passed":
            continue
        located = attempt.get("located") or {}
        entries.append({
            "iteration": attempt.get("iteration"),
            "believed_cause": (
                located.get("bug_description")
                or located.get("root_cause")
                or attempt.get("rationale")
            ),
            "fix_direction": attempt.get("fix_direction"),
            "outcome": cls.get("category") or cls.get("five_category"),
        })
    if not entries:
        return None
    return (
        "ALREADY TRIED (do not repeat any of these strategies):\n"
        + json.dumps(entries, indent=2, ensure_ascii=False)
    )


def _normalize_failure_excerpt(text: str | None) -> str:
    text = text or ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    interesting: list[str] = []
    for line in lines:
        low = line.lower()
        if (
            "assert" in low
            or "error" in low
            or "failed" in low
            or "aborted" in low
            or "illegal memory" in low
            or "miscompare" in low
        ):
            interesting.append(line)
    if not interesting:
        interesting = lines[:4]
    excerpt = "\n".join(interesting[:6])
    excerpt = re.sub(r"/tmp/[^\s]+", "/tmp/...", excerpt)
    excerpt = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", excerpt)
    return _truncate_text(excerpt, 700)


def _attempt_failure_signature(attempt: dict) -> dict | None:
    cls = attempt.get("classification") or {}
    if cls.get("status") == "passed" or cls.get("five_category") == "passed":
        return None
    excerpt = (
        attempt.get("test_output_excerpt")
        or attempt.get("build_output_excerpt")
        or cls.get("logic_error_detail")
        or ""
    )
    return {
        "five_category": cls.get("five_category"),
        "category": cls.get("category"),
        "error_type": cls.get("error_type"),
        "build_ok": cls.get("build_ok"),
        "excerpt": _normalize_failure_excerpt(excerpt),
    }


def _repeated_failure_alert(state: _LoopState) -> str | None:
    if len(state.attempt_history) < 2:
        return None
    recent = state.attempt_history[-2:]
    sigs = [_attempt_failure_signature(item) for item in recent]
    if not sigs[0] or not sigs[1] or sigs[0] != sigs[1]:
        return None

    patch_summary = [
        {
            "iteration": item.get("iteration"),
            "rationale": item.get("rationale"),
            "located": item.get("located"),
            "patch_actions": item.get("patch_actions"),
        }
        for item in recent
    ]
    lines = [
        "REPEATED FAILURE ALERT:",
        "The last two submitted patches failed with the same failure signature.",
        json.dumps(sigs[-1], indent=2, ensure_ascii=False),
        "",
        "This means the previous localization/fix hypothesis is not working. "
        "For this iteration, do not make a small/no-op edit on top of the same "
        "failed idea. Re-read the failure, identify a materially different root "
        "cause, and either revert the ineffective patch actions or replace the "
        "whole failed region with a new minimal fix.",
        "",
        "Recent failed patch actions:",
        _truncate_text(json.dumps(patch_summary, indent=2, ensure_ascii=False), 5000),
    ]
    if state.broken_code:
        lines.extend([
            "",
            "Original broken candidate excerpt for rollback comparison:",
            "```cuda",
            _truncate_text(state.broken_code, 6000),
            "```",
        ])
    return "\n".join(lines)


def _performance_feedback_text(perf: dict, gate: dict) -> str:
    lines = [
        "Correctness tests passed, but the performance gate failed.",
        f"Required speedup: {gate.get('min_speedup', 0.3):.3f}x",
    ]
    if "speedup" in perf:
        lines.append(f"Speedup: {float(perf['speedup']):.3f}x")
    if "candidate_runtime_ms" in perf:
        lines.append(f"Candidate runtime: {perf['candidate_runtime_ms']:.6g} ms")
    if "reference_runtime_ms" in perf:
        lines.append(f"Reference runtime: {perf['reference_runtime_ms']:.6g} ms")
    if perf.get("timing_method"):
        lines.append(f"Timing method: {perf['timing_method']}")
    if perf.get("skipped"):
        lines.append(f"Measurement skipped: {perf['skipped']}")
    if perf.get("error"):
        lines.append("Performance measurement error:")
        lines.append(_truncate_text(str(perf["error"]), 4000))
    return "\n".join(lines)


def _solution_artifact_suffix(input_meta: dict) -> str:
    suffix = Path(input_meta.get("solution_file", "solution.cu")).suffix
    return suffix or ".cu"


def _performance_gpu_id(config: dict) -> int:
    build_cfg = config.get("build", {}) or {}
    exp_cfg = config.get("experiment", {}) or {}
    value = build_cfg.get("gpu_id", exp_cfg.get("gpu_id", 0))
    return int(value)


def _forced_action_context(state: _LoopState) -> str | None:
    if state.force_read_code_reason:
        if state.force_read_code_full_file:
            return (
                f"{state.force_read_code_reason}\n"
                "Prefer `read_code` with arguments `{\"full_file\": true}`. "
                "After that full-file read, either use `apply_edit` with "
                "verbatim text, use `apply_edit` with `start_line`/`end_line`, "
                "or call `write_solution` with a complete revised `code` file "
                "as a full-file fallback. If you already have the complete "
                "revised file, you may call `write_solution(code=...)` now."
            )
        return (
            f"{state.force_read_code_reason}\n"
            "Read the current candidate code and use exact text or "
            "start_line/end_line before attempting another surgical patch; "
            "or submit a complete revised file with `write_solution(code=...)`."
        )
    return None


def _join_decision_constraints(*parts: str | None) -> str | None:
    items = [p.strip() for p in parts if p and p.strip()]
    return "\n".join(items) if items else None


def _json_role_action_request(
    tool_schemas: list[dict],
    role_name: str,
    allow_final_report: bool = True,
    extra_context: str | None = None,
    structured_report: bool = False,
) -> str:
    pseudo = ""
    if allow_final_report and structured_report:
        # Flat args (no nested object) — easier for the model to emit reliably,
        # especially under thinking-mode token pressure.
        pseudo = (
            "\nYou may also finish this role with a structured report:\n"
            '{"tool":"final_report","args":{'
            '"summary":"<=1 sentence","key_points":["<=4 terse evidence bullets"],'
            '"ruled_out":["<=4 rejected hypotheses"]}}\n'
        )
    elif allow_final_report:
        pseudo = (
            "\nYou may also finish this role with:\n"
            '{"tool":"final_report","args":{"report":"compact evidence summary"}}\n'
        )
    context_block = f"\nDecision constraints:\n{extra_context}\n" if extra_context else ""
    return (
        f"ACTION REQUEST ({role_name})\n"
        "Return exactly one compact JSON object.\n"
        '{"tool":"one_available_tool_name","args":{}}\n\n'
        "Available real tools for this turn:\n"
        f"{_compact_tool_schema(tool_schemas)}\n"
        f"{pseudo}"
        f"{context_block}"
        "If the previous action failed, use the failure result to choose a "
        "different next action. No prose. No markdown.\n\n"
        f"{JSON_ACTION_OUTPUT_DISCIPLINE}"
    )


def _build_ctx_config(agent_cfg: dict) -> ContextConfig:
    return ContextConfig(
        tool_result_cap_chars=int(agent_cfg.get("tool_result_cap_chars", 8000)),
        keep_recent_tool_exchanges=int(agent_cfg.get("keep_recent_tool_exchanges", 6)),
        compact_old_iterations=bool(agent_cfg.get("compact_old_iterations", True)),
        token_budget=int(agent_cfg.get("token_budget", 80_000)),
        budget_warn_ratio=float(agent_cfg.get("budget_warn_ratio", 0.8)),
        budget_force_ratio=float(agent_cfg.get("budget_force_ratio", 0.95)),
        chars_per_token=float(agent_cfg.get("chars_per_token", 4.0)),
        summary_target_chars=int(agent_cfg.get("summary_target_chars", 4000)),
    )


def _maybe_make_summarizer(agent_cfg: dict, client, config: dict):
    if not bool(agent_cfg.get("use_llm_summary", True)):
        return None
    return make_summarizer(
        client, config,
        max_calls=int(agent_cfg.get("max_summary_calls", 40)),
        max_output_tokens=int(agent_cfg.get("summary_max_tokens", 4096)),
    )


def _tool_decision_overrides(agent_cfg: dict) -> tuple[int | str | None, float | None]:
    max_tokens_cfg = agent_cfg.get("tool_decision_max_tokens", 2048)
    if max_tokens_cfg is None:
        max_tokens: int | str | None = None
    elif isinstance(max_tokens_cfg, str) and not max_tokens_cfg.strip().isdigit():
        max_tokens = max_tokens_cfg
    else:
        max_tokens = int(max_tokens_cfg)
    temperature_cfg = agent_cfg.get("tool_decision_temperature")
    temperature = None if temperature_cfg is None else float(temperature_cfg)
    return max_tokens, temperature
