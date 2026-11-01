"""
LLM-backed summarisation for context management.

When a tool result (or a batch of aged-out exchanges) is too large to keep
verbatim, we ask the *same* agent model to compress it rather than
blind-truncating.

Why not truncate: a head+tail truncation drops the middle of a file and
leaves a "[middle truncated]" marker. The agent reads that as "my earlier
read was incomplete" and re-reads the same file in a loop (observed on
CUDA_10's first smoke run — test_main.cu read 6×). An LLM summary carries
no incompleteness signal — it reads as a deliberate, complete digest, so
the agent acts on it instead of re-fetching.

`make_summarizer(client, config)` returns a callable

    summarize(text: str, kind: str, target_chars: int) -> str

with:
  - content-hash caching (same file summarised once, reused after)
  - a call budget (falls back to truncation when exhausted / on API error)
  - kind-specific prompts that tell the model what to preserve

Kinds:
  code          — read_code / read_test_file / read_prompt output
  diagnostic    — compute-sanitizer / cuda-gdb output
  build_log     — build_only / test_only / write_solution feedback
  investigation — a batch of tool exchanges aged out of the recent window
  attempt       — one completed iteration's tool activity (cross-iter fold)
  generic       — fallback
"""

from __future__ import annotations

import hashlib
import logging
from typing import Callable

logger = logging.getLogger("cuda_debugger_exp")


SUMMARY_SYSTEM = """\
You are a context compressor for an automated CUDA debugging agent. You \
receive a chunk of context (source code, tool output, or an investigation \
log) that is too large to keep verbatim. Produce a dense, faithful summary \
the agent can still act on. Never invent facts. Never add commentary. \
Output ONLY the summary text."""


# kind -> extra "preserve exactly" instruction appended to the user prompt
_KIND_INSTRUCTIONS = {
    "code": (
        "This is CUDA/C++/Python source. PRESERVE EXACTLY: every function, "
        "kernel, and method signature; struct/class definitions; key "
        "constants and macros; launch-config expressions; and the LINE "
        "NUMBERS of anything a debugger would care about. Keep line-number "
        "references in the form `L<n>`. Drop boilerplate and repetitive "
        "blocks, but never drop a signature."
    ),
    "diagnostic": (
        "This is compute-sanitizer or cuda-gdb output. PRESERVE EXACTLY: "
        "the error kind, the faulting thread/block coordinates, the faulting "
        "address, the source file:line of the fault, the signal name, and "
        "the total error count. Drop repeated per-thread duplicates."
    ),
    "build_log": (
        "This is a compiler / test log. PRESERVE EXACTLY: every distinct "
        "error message with its file:line, the first failing assertion, and "
        "any Expected/Got/Tolerance values. Drop warnings and repeated "
        "instantiation noise unless they are the only diagnostic."
    ),
    "investigation": (
        "This is a log of tool calls the agent already made in the CURRENT "
        "debugging iteration. Summarise WHAT was investigated, WHAT was "
        "found (concrete facts: signatures seen, error coordinates, line "
        "numbers), and WHAT remains unknown. The agent will use this to "
        "avoid re-investigating."
    ),
    "attempt": (
        "This is the tool activity of one COMPLETED debugging iteration "
        "(an earlier fix attempt). Summarise: what fix was attempted, what "
        "the build/test outcome was, and the single most important finding. "
        "Two or three sentences — this is stale background, not live state."
    ),
    "generic": (
        "Summarise faithfully, preserving any error messages, identifiers, "
        "numbers, and file:line references."
    ),
}


def _build_user_prompt(text: str, kind: str, target_chars: int) -> str:
    instruction = _KIND_INSTRUCTIONS.get(kind, _KIND_INSTRUCTIONS["generic"])
    return (
        f"{instruction}\n\n"
        f"Target length: about {target_chars} characters (hard limit "
        f"{target_chars * 2}).\n\n"
        f"--- CONTENT TO SUMMARISE ---\n{text}\n--- END CONTENT ---"
    )


def _truncate(text: str, cap: int) -> str:
    """Head+tail fallback used when summarisation is unavailable."""
    if len(text) <= cap:
        return text
    half = max(1, (cap - 80) // 2)
    return (
        text[:half]
        + "\n... [middle truncated — summariser unavailable] ...\n"
        + text[-half:]
    )


def make_summarizer(
    client,
    config: dict,
    *,
    max_calls: int = 40,
    max_output_tokens: int = 4096,
) -> Callable[[str, str, int], str]:
    """Build an LLM-backed summarise() callable.

    `client` / `config` are the same objects the agent loop uses, so the
    summariser runs on the same model. A derived config lowers temperature
    for determinism.

    `max_output_tokens` caps the summary length. It must be comfortably
    above the token-equivalent of the largest `target_chars` the caller
    passes — a too-low cap silently truncates the summary mid-sentence,
    which is worse than a longer-but-complete digest.
    """
    from .llm_client import call_llm

    # Derived config: low temperature, summary-sized output. Same
    # backend/model/base_url — no extra endpoint needed.
    api = dict(config.get("api", {}))
    api["temperature"] = 0.2
    api["max_tokens"] = int(max_output_tokens)
    api["max_completion_tokens"] = int(max_output_tokens)
    # A reasoning model would burn the whole budget thinking; summarisation
    # is a shallow task, so drop reasoning_effort if present.
    api.pop("reasoning_effort", None)
    api.pop("seed", None)
    summ_config = {**config, "api": api}

    cache: dict[str, str] = {}
    state = {"calls": 0, "cache_hits": 0, "fallbacks": 0}

    def summarize(text: str, kind: str = "generic", target_chars: int = 2000) -> str:
        if not text:
            return text
        key = hashlib.sha256(
            f"{kind}|{target_chars}|{text}".encode("utf-8")
        ).hexdigest()
        if key in cache:
            state["cache_hits"] += 1
            return cache[key]

        if state["calls"] >= max_calls:
            state["fallbacks"] += 1
            logger.warning(
                f"  [summarizer] call budget {max_calls} exhausted — "
                f"falling back to truncation for {kind}"
            )
            return _truncate(text, target_chars)

        messages = [
            {"role": "system", "content": SUMMARY_SYSTEM},
            {"role": "user", "content": _build_user_prompt(text, kind, target_chars)},
        ]
        try:
            state["calls"] += 1
            raw = call_llm(client, summ_config, messages)
        except Exception as e:
            state["fallbacks"] += 1
            logger.warning(
                f"  [summarizer] LLM call failed ({type(e).__name__}: {e}) — "
                f"falling back to truncation"
            )
            return _truncate(text, target_chars)

        summary = (raw or "").strip()
        if not summary:
            state["fallbacks"] += 1
            return _truncate(text, target_chars)

        labelled = (
            f"[ctx-compressed {kind}: {len(text)}→{len(summary)} chars]\n"
            + summary
        )
        cache[key] = labelled
        return labelled

    # Expose counters for logging / tests
    summarize.state = state          # type: ignore[attr-defined]
    summarize.cache = cache          # type: ignore[attr-defined]
    return summarize
