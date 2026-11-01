"""
Context manager for the agent loop.

The agent loop runs an open-ended sequence:

    system + user
    → assistant.tool_call[A] → tool[A]
    → assistant.tool_call[B] → tool[B]
    → ...
    → assistant.tool_call[write_solution] → tool[write_solution]
    → user (build/test feedback)
    → assistant.tool_call[C] → tool[C] → ...

Without bounding, the message list grows ~unbounded across iterations and
the token bill explodes. Four layers of defence (each can be tuned via
config.agent.context):

  1. **Per-tool-result char cap** — already enforced inside agent_tools
     and diagnostics; this module only enforces it again as a safety net.
  2. **In-iteration sliding window** — within the messages produced since
     the last write_solution, keep only the most recent K tool exchanges
     verbatim; older ones get folded into a one-line summary.
  3. **Cross-iteration compaction** — once write_solution lands, all prior
     tool exchanges from previous iterations get replaced with a single
     "prev-attempt summary" message listing what was tried.
  4. **Token-budget soft cap** — track approximate token count of the
     message list; once over a threshold, inject a "budget low, submit a
     fix soon" nudge to the system message AND aggressively compact.

A coarse token estimator is used (4 chars/token); this matches OpenAI's
empirical rule of thumb for English+code and avoids dragging in tiktoken
as a hard dep. Off by ~15% but it's the *trend* that matters, not the
absolute number.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger("cuda_debugger_exp")

# A summariser is `(text, kind, target_chars) -> str`. When None is passed
# to the compaction functions, they fall back to head+tail truncation
# (L1) / one-line digests (L2/L3) — the pre-LLM-summary behaviour, kept so
# unit tests and LLM-less runs still work.
Summarizer = Callable[[str, str, int], str]

# Maps a tool name to the summary "kind" used by context_summary.
_TOOL_SUMMARY_KIND = {
    "read_code": "code",
    "read_test_file": "code",
    "read_prompt": "code",
    "run_static_analysis": "diagnostic",
    "run_compute_sanitizer": "diagnostic",
    "build_only": "build_log",
    "test_only": "build_log",
    "write_solution": "build_log",
}

ACTION_RESULT_MARKER = "[DETERMINISTIC ACTION RESULT]"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ContextConfig:
    # Layer 1 — last-resort cap for a *runaway* tool result. This MUST be
    # >= the largest per-tool cap in agent_tools.RESULT_CAPS (currently
    # 8000 for read_code / read_test_file), otherwise L1 re-truncates
    # legitimately-sized file reads every turn — the model then sees its
    # own earlier read mangled with a "[middle truncated]" marker and
    # re-reads the same file in a loop. L1 should only fire on something
    # truly pathological; bounding read *accumulation* is L2/L3's job.
    tool_result_cap_chars: int = 8000
    # Layer 2 — sliding window inside the current iteration
    keep_recent_tool_exchanges: int = 6
    # Layer 3 — cross-iteration compaction
    compact_old_iterations: bool = True
    # Layer 4 — token budget
    token_budget: int = 80_000
    budget_warn_ratio: float = 0.8     # nudge model when ≥80%
    budget_force_ratio: float = 0.95   # hard force-submit when ≥95%
    chars_per_token: float = 4.0       # coarse estimator
    # Target size of an LLM-produced summary (L1/L2/L3). Still well below
    # tool_result_cap_chars, so a compressed message does not re-trigger
    # L1 next turn. The summariser's hard output-token cap
    # (agent.summary_max_tokens) must sit comfortably above this value's
    # token-equivalent or summaries get cut mid-sentence.
    summary_target_chars: int = 4000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def estimate_tokens(messages: list[dict], chars_per_token: float = 4.0) -> int:
    """Rough token count from message content + role/name overhead."""
    total_chars = 0
    for m in messages:
        total_chars += len(m.get("role", "")) + 4
        content = m.get("content")
        if isinstance(content, str):
            total_chars += len(content)
        # tool calls live in `tool_calls`, not content
        for tc in m.get("tool_calls") or []:
            fn = (tc.get("function") or {})
            total_chars += len(fn.get("name", ""))
            args = fn.get("arguments", "")
            if isinstance(args, str):
                total_chars += len(args)
            elif isinstance(args, dict):
                total_chars += len(json.dumps(args))
        if m.get("tool_call_id"):
            total_chars += len(m["tool_call_id"])
    return int(total_chars / max(chars_per_token, 1.0))


def _truncate_str(s: str, cap: int) -> str:
    if len(s) <= cap:
        return s
    half = cap // 2
    return s[:half] + "\n... [middle truncated] ...\n" + s[-half:]


def _split_action_result_content(content: str) -> tuple[str, str]:
    """Return (header, body) for JSON-action result content."""
    parts = content.split("\n\n", 1)
    if len(parts) == 1:
        return content, ""
    return parts[0], parts[1]


# A "tool group" is the atomic unit L2/L3 operate on: one assistant
# message plus the contiguous run of `tool` messages that answer its
# tool_calls. A turn may carry MULTIPLE tool_calls (observed: gemma-4
# emits 3 in one assistant message), so the old (assistant, tool) pair
# model is wrong — summarising a "pair" would delete the assistant and
# only its first tool result, orphaning the rest (an orphaned tool
# message has no matching tool_call_id → API 400).


def _json_action_result_tool(content: str) -> str | None:
    if not content.startswith(ACTION_RESULT_MARKER):
        return None
    for line in content.splitlines()[1:5]:
        if line.startswith("tool="):
            return line.split("=", 1)[1].strip() or None
    return None


def _is_json_action_result(m: dict) -> bool:
    c = m.get("content")
    return (
        m.get("role") == "user"
        and isinstance(c, str)
        and c.startswith(ACTION_RESULT_MARKER)
    )


def _json_action_name_args(content: str | None) -> tuple[str, str]:
    raw = (content or "").strip()
    try:
        obj = json.loads(raw)
    except Exception:
        return "json_action", "{}"
    if not isinstance(obj, dict):
        return "json_action", "{}"
    tool = obj.get("tool") or obj.get("name") or obj.get("action") or "json_action"
    args = obj.get("args") if isinstance(obj.get("args"), dict) else {}
    return str(tool), _compact_args(args)


def _tool_groups(
    messages: list[dict], lo: int, hi: int
) -> list[tuple[int, list[int]]]:
    """Return [(assistant_idx, [tool_idx, ...]), ...] within messages[lo:hi].

    Each group is one assistant-with-tool_calls followed by its run of
    `tool` messages, or one JSON-action assistant followed by the
    deterministic action-result `user` message used by JSON-action mode.
    Deleting/folding a whole group keeps the native message list API-valid
    (no orphaned tool messages) and bounds JSON-action transcripts too.
    """
    groups: list[tuple[int, list[int]]] = []
    i = lo
    while i < hi:
        m = messages[i]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            tool_idxs: list[int] = []
            j = i + 1
            while j < hi and messages[j].get("role") == "tool":
                tool_idxs.append(j)
                j += 1
            groups.append((i, tool_idxs))
            i = max(j, i + 1)
        elif (
            m.get("role") == "assistant"
            and not m.get("tool_calls")
            and i + 1 < hi
            and _is_json_action_result(messages[i + 1])
        ):
            groups.append((i, [i + 1]))
            i += 2
        else:
            i += 1
    return groups


def _compact_args(raw) -> str:
    """Compact a tool-call arguments blob — drop huge `code` payloads."""
    if isinstance(raw, str):
        try:
            d = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return raw[:120]
    else:
        d = raw or {}
    compact = {}
    for k, v in d.items():
        if k == "code" and isinstance(v, str):
            compact[k] = f"<{len(v)} chars>"
        elif isinstance(v, str) and len(v) > 80:
            compact[k] = v[:80] + "…"
        else:
            compact[k] = v
    return json.dumps(compact, ensure_ascii=False)


def _format_group_full(messages: list[dict], asst_idx: int, tool_idxs: list[int]) -> str:
    """Verbose per-call "TOOL / RESULT" rendering — fed to the LLM
    summariser so it has the real detail to compress."""
    asst = messages[asst_idx]
    if not asst.get("tool_calls"):
        result_msg = messages[tool_idxs[0]] if tool_idxs else {}
        result = result_msg.get("content", "")
        if not isinstance(result, str):
            result = str(result)
        action_name, args = _json_action_name_args(asst.get("content"))
        name = _json_action_result_tool(result) or action_name
        return f"TOOL: {name}({args})\nRESULT:\n{result}"

    by_id = {
        messages[t].get("tool_call_id"): messages[t].get("content", "")
        for t in tool_idxs
    }
    blocks = []
    for tc in asst.get("tool_calls") or []:
        fn = tc.get("function") or {}
        name = fn.get("name", "?")
        args = _compact_args(fn.get("arguments", ""))
        result = by_id.get(tc.get("id"), "(no result captured)")
        if not isinstance(result, str):
            result = str(result)
        blocks.append(f"TOOL: {name}({args})\nRESULT:\n{result}")
    return "\n\n".join(blocks) if blocks else "(no tool calls)"


def _summarise_group_oneline(messages: list[dict], asst_idx: int, tool_idxs: list[int]) -> str:
    """One-line-per-call digest — the no-summariser fallback."""
    asst = messages[asst_idx]
    if not asst.get("tool_calls"):
        result_msg = messages[tool_idxs[0]] if tool_idxs else {}
        content = result_msg.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        result_name = _json_action_result_tool(content)
        action_name, args = _json_action_name_args(asst.get("content"))
        name = result_name or action_name
        _header, body = _split_action_result_content(content)
        first_line = next(
            (
                ln.strip()
                for ln in body.splitlines()
                if ln.strip()
            ),
            "(empty)",
        )
        return f"- `{name}({args})` → {_truncate_str(first_line, 160)}"

    by_id = {messages[t].get("tool_call_id"): messages[t] for t in tool_idxs}
    lines = []
    for tc in asst.get("tool_calls") or []:
        fn = tc.get("function") or {}
        name = fn.get("name", "?")
        args = _compact_args(fn.get("arguments", ""))
        res = by_id.get(tc.get("id"))
        first_line = "(no result)"
        if res is not None:
            c = res.get("content", "")
            if isinstance(c, str):
                first_line = next(
                    (ln.strip() for ln in c.splitlines() if ln.strip()),
                    "(empty)",
                )
        lines.append(f"- `{name}({args})` → {_truncate_str(first_line, 160)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Message-list manipulation
# ---------------------------------------------------------------------------


@dataclass
class CompactionStats:
    tokens_before: int = 0
    tokens_after: int = 0
    layers_applied: list[str] = field(default_factory=list)


def _tool_kind_for_message(messages: list[dict], tool_idx: int) -> str:
    """Resolve the summary kind of a native `tool` or JSON-action result."""
    msg = messages[tool_idx]
    if _is_json_action_result(msg):
        c = msg.get("content", "")
        name = _json_action_result_tool(c) if isinstance(c, str) else None
        return _TOOL_SUMMARY_KIND.get(name or "", "generic")

    tcid = messages[tool_idx].get("tool_call_id")
    for j in range(tool_idx - 1, -1, -1):
        m = messages[j]
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            if tc.get("id") == tcid:
                name = (tc.get("function") or {}).get("name", "")
                return _TOOL_SUMMARY_KIND.get(name, "generic")
    return "generic"


def cap_tool_results(
    messages: list[dict],
    cfg: ContextConfig,
    summarizer: Summarizer | None = None,
) -> int:
    """Layer 1 — compress any tool message whose content exceeds the cap.

    With a `summarizer`, an oversized result is replaced by an LLM-produced
    digest (kind chosen from the tool that produced it). Without one, it
    falls back to head+tail truncation.

    The resulting message is far smaller than the cap, so it will not be
    re-compressed on the next turn — no need for a per-message flag.

    Iteration-marker messages (`[BUILD/TEST RESULT] ...`) are skipped:
    summarising one would strip the marker prefix and blind L3's
    iteration detection. The current iteration's marker is live feedback
    that must stay verbatim anyway; old markers are folded whole by L3.

    Returns the number of messages that were compressed.
    """
    trimmed = 0
    for i, m in enumerate(messages):
        is_json_result = _is_json_action_result(m)
        if m.get("role") != "tool" and not is_json_result:
            continue
        if _is_marker(m):
            continue
        c = m.get("content")
        if not isinstance(c, str) or len(c) <= cfg.tool_result_cap_chars:
            continue
        if is_json_result:
            header, body = _split_action_result_content(c)
            body = body or c
            kind = _tool_kind_for_message(messages, i)
            if summarizer is not None:
                compacted = summarizer(body, kind, cfg.summary_target_chars)
            else:
                compacted = _truncate_str(body, cfg.tool_result_cap_chars)
            m["content"] = header + "\n\n" + compacted
            trimmed += 1
            continue
        if summarizer is not None:
            kind = _tool_kind_for_message(messages, i)
            m["content"] = summarizer(c, kind, cfg.summary_target_chars)
        else:
            m["content"] = _truncate_str(c, cfg.tool_result_cap_chars)
        trimmed += 1
    return trimmed


# Iteration-boundary marker. agent_loop.py replies to a write_solution
# tool call with a `tool` message whose content starts with this string.
# (It is a `tool` message, NOT a `user` message — L3 used to look for the
# wrong role and so never fired. Detection now accepts either role.)
ITERATION_MARKER = "[BUILD/TEST RESULT]"


def _is_marker(m: dict) -> bool:
    content = m.get("content")
    if not isinstance(content, str):
        return False
    if _is_json_action_result(m):
        return (
            _json_action_result_tool(content) == "write_solution"
            and ITERATION_MARKER in content
        )
    return (
        m.get("role") in ("user", "tool")
        and content.startswith(ITERATION_MARKER)
    )


def _marker_indices(messages: list[dict]) -> list[int]:
    return [i for i, m in enumerate(messages) if _is_marker(m)]


def _digest_groups(
    messages: list[dict],
    groups: list[tuple[int, list[int]]],
    kind: str,
    cfg: ContextConfig,
    summarizer: Summarizer | None,
) -> str:
    """Compress a list of tool groups into one summary block.

    With a summariser: concatenate the full groups and LLM-compress.
    Without one: fall back to the one-line-per-call digest.
    """
    if summarizer is not None:
        blob = "\n\n".join(
            _format_group_full(messages, a, ts) for a, ts in groups
        )
        return summarizer(blob, kind, cfg.summary_target_chars)
    return "\n".join(
        _summarise_group_oneline(messages, a, ts) for a, ts in groups
    )


def slide_recent_exchanges(
    messages: list[dict],
    cfg: ContextConfig,
    summarizer: Summarizer | None = None,
) -> int:
    """Layer 2 — within the CURRENT iteration, keep the last K tool groups
    verbatim and summarise the older ones into one message.

    The current iteration is everything after the last iteration marker
    (or, before any submission, from the first tool-calling assistant).
    Whole tool groups are deleted atomically — never a partial group —
    so multi-tool-call turns can't be left with orphaned tool messages.
    """
    K = max(0, cfg.keep_recent_tool_exchanges)

    markers = _marker_indices(messages)
    if markers:
        boundary = markers[-1] + 1
    else:
        boundary = len(messages)
        for i, m in enumerate(messages):
            if (
                m.get("role") == "assistant"
                and (
                    m.get("tool_calls")
                    or (
                        i + 1 < len(messages)
                        and _is_json_action_result(messages[i + 1])
                    )
                )
            ):
                boundary = i
                break

    groups = _tool_groups(messages, boundary, len(messages))
    if len(groups) <= K:
        return 0

    to_summarise = groups[:-K] if K > 0 else groups
    if not to_summarise:
        return 0

    digest = _digest_groups(messages, to_summarise, "investigation",
                            cfg, summarizer)

    drop = set()
    for a, ts in to_summarise:
        drop.add(a)
        drop.update(ts)
    insert_at = min(drop)
    for idx in sorted(drop, reverse=True):
        del messages[idx]
    messages.insert(insert_at, {
        "role": "user",
        "content": "[Earlier investigation in this iteration — summary]\n"
                   + digest,
    })
    return len(to_summarise)


def compact_old_iterations(
    messages: list[dict],
    cfg: ContextConfig,
    summarizer: Summarizer | None = None,
) -> int:
    """Layer 3 — fold each COMPLETED iteration (except the most recent
    completed one) into a single "prev-attempt" summary message.

    An iteration ends at a `[BUILD/TEST RESULT]` marker. The whole
    iteration block — investigation groups, the write_solution assistant,
    AND the marker tool message — is replaced as one unit. Folding the
    block whole (rather than leaving the marker behind) is essential: the
    marker is a `tool` message, and orphaning it from its write_solution
    assistant would make the message list API-invalid.

    The most recent COMPLETED iteration is left verbatim — the live turn
    is reacting to its build/test feedback and needs it in full.

    Returns the number of iterations compacted.
    """
    if not cfg.compact_old_iterations:
        return 0

    markers = _marker_indices(messages)
    if len(markers) <= 1:
        # 0 completed, or only the most-recent-completed one → nothing to
        # fold (that one stays verbatim).
        return 0

    # Iteration k spans (prev_marker, marker_k] inclusive. Indices 0/1
    # (system + initial task) are never touched. Fold all but the last
    # completed iteration (markers[-1]'s).
    spans: list[tuple[int, int]] = []
    prev = 1
    for k in range(len(markers) - 1):
        start = max(2, prev + 1)
        end = markers[k]  # inclusive — the marker tool message
        if start <= end:
            spans.append((start, end))
        prev = markers[k]

    compacted = 0
    for start, end in reversed(spans):
        groups = _tool_groups(messages, start, end + 1)
        if not groups:
            continue
        digest = _digest_groups(messages, groups, "attempt", cfg, summarizer)
        outcome = ""
        marker_msg = messages[end]
        if isinstance(marker_msg.get("content"), str):
            outcome = "\n".join(marker_msg["content"].splitlines()[:3])
        body = digest if not outcome else f"{digest}\nOutcome: {outcome}"
        del messages[start:end + 1]
        messages.insert(start, {
            "role": "user",
            "content": "[Prev iteration summary]\n" + body,
        })
        compacted += 1
    return compacted


def check_budget(
    messages: list[dict], cfg: ContextConfig
) -> tuple[int, str | None]:
    """Layer 4 — estimate tokens and return (tokens, status).

    status: None / "warn" / "force"
    """
    toks = estimate_tokens(messages, cfg.chars_per_token)
    if toks >= int(cfg.token_budget * cfg.budget_force_ratio):
        return toks, "force"
    if toks >= int(cfg.token_budget * cfg.budget_warn_ratio):
        return toks, "warn"
    return toks, None


def manage_context(
    messages: list[dict],
    cfg: ContextConfig,
    summarizer: Summarizer | None = None,
) -> CompactionStats:
    """Run all three compaction layers in order, returning stats.

    With a `summarizer`, oversized tool results (L1) and aged-out exchanges
    (L2/L3) are LLM-compressed; without one, they fall back to head+tail
    truncation / one-line digests. Mutates `messages` in place.
    """
    stats = CompactionStats()
    stats.tokens_before = estimate_tokens(messages, cfg.chars_per_token)

    trimmed_l1 = cap_tool_results(messages, cfg, summarizer)
    if trimmed_l1:
        stats.layers_applied.append(f"L1:cap×{trimmed_l1}")

    folded_l3 = compact_old_iterations(messages, cfg, summarizer)
    if folded_l3:
        stats.layers_applied.append(f"L3:old×{folded_l3}")

    summarised_l2 = slide_recent_exchanges(messages, cfg, summarizer)
    if summarised_l2:
        stats.layers_applied.append(f"L2:slide×{summarised_l2}")

    stats.tokens_after = estimate_tokens(messages, cfg.chars_per_token)
    return stats


def get_budget_nudge(status: str | None, in_locate_phase: bool = False) -> str | None:
    """Return a system-message nudge string for the budget-warning state."""
    if status == "warn":
        return (
            "\n\n[Context budget is at ≥80%. Wrap up your investigation soon "
            "and submit a fix via `write_solution`.]"
        )
    if status == "force":
        if in_locate_phase:
            return (
                "\n\n[Context budget is at ≥95%. You are still in LOCATE. "
                "Call `locate_bug` on the best-supported line next; once FIX "
                "phase unlocks, patch and submit without further investigation.]"
            )
        return (
            "\n\n[Context budget is at ≥95%. You MUST call `write_solution` "
            "on the next turn — further investigation is not affordable.]"
        )
    return None
