"""
Tool registry for the agent loop.

Defines 10 tools that the LLM can call to investigate a failing CUDA task,
localise the bug, and patch it. Each tool has:
  - JSONSchema (OpenAI / vLLM tool-calling format)
  - Executor function (subprocess wrapper / file read / build hook)
  - Per-tool quota (default + override from config)

Every executor returns a `ToolResult` with:
  - ok: bool
  - content: str (already capped, ready to feed back to the model)
  - truncated: bool
  - meta: dict (returncode, file ranges, etc.)

The dispatcher (`call_tool`) records each invocation in a `ToolTrace` so
the loop can write per-task `tool_trace.jsonl` next to manifest.jsonl.

Design choice — tool surface intentionally narrow:
  - No arbitrary shell. Only the named tools below.
  - read_* tools are path-confined to the task workspace + dataset.
  - compute-sanitizer runs via diagnostics.py with a validated option allow-list.

cuda-gdb was removed from the tool set: on the benchmark's failures it
was used almost exclusively on host-side `assert()` aborts (a wrong-result
bug, not a GPU fault), where a backtrace carries no signal — it only
burned wall-clock and a tool call.
"""

from __future__ import annotations

import copy
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .diagnostics import (
    DiagResult,
    run_compute_sanitizer,
    sanitizer_available,
)
from .task_registry import resolve_task_paths

logger = logging.getLogger("cuda_debugger_exp")


# ---------------------------------------------------------------------------
# Defaults — overridable via config.agent.quotas
# ---------------------------------------------------------------------------

DEFAULT_QUOTAS = {
    "read_code": 20,
    "read_test_file": 10,
    "read_prompt": 3,
    "run_static_analysis": 4,
    "run_compute_sanitizer": 10,
    "build_only": 6,
    "test_only": 6,
    "locate_bug": None,       # mandatory step — never quota-blocked
    "apply_edit": None,       # surgical edits — unlimited within a fix
    "write_solution": None,   # bounded by iteration limit, not call count
}

# Per-tool result char caps. These are the *raw* limits at the tool layer
# — deliberately generous so a full file/log reaches the context layer.
# context_manager's L1 (tool_result_cap_chars, default 8000) then decides
# whether to LLM-summarise: results under 8KB pass through verbatim,
# larger ones get compressed into a digest. If these caps were <= L1's
# cap, L1 would never see anything worth summarising.
RESULT_CAPS = {
    "read_code": 200000,
    "read_test_file": 24000,
    "read_prompt": 12000,
    "run_static_analysis": 16000,
    "build_only": 16000,
    "test_only": 16000,
    "default": 6000,
}

MAX_READ_LINES = 400
ALLOWED_READ_SUFFIXES = (".cu", ".cuh", ".cpp", ".c", ".cc", ".h", ".hpp", ".py", ".txt", ".md", ".json", ".mk")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ToolResult:
    """What a tool returns to the agent."""

    ok: bool
    content: str
    truncated: bool = False
    meta: dict = field(default_factory=dict)
    error: str | None = None

    def to_message_content(self) -> str:
        """Render as the `content` of a tool-role message."""
        if not self.ok:
            return f"[tool error] {self.error or 'unknown'}"
        if self.truncated:
            return self.content + "\n... [output truncated] ..."
        return self.content


@dataclass
class ToolTraceEntry:
    iteration: int
    tool: str
    args: dict
    ok: bool
    truncated: bool
    content_chars: int
    error: str | None
    elapsed_ms: float

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "tool": self.tool,
            "args": self.args,
            "ok": self.ok,
            "truncated": self.truncated,
            "content_chars": self.content_chars,
            "error": self.error,
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


@dataclass
class ToolEnv:
    """Per-task execution context passed to every tool executor."""

    task_stem: str
    task_workdir: Path           # /tmp/.../CUDA_10
    dataset_dir: Path            # repo data/ (or a legacy flat dataset dir)
    testbench_dir: Path          # dataset_dir / testbench / task_stem
    prompt_path: Path            # dataset_dir / prompts / task_stem.txt
    input_meta: dict             # parsed input/<stem>.json
    solution_file: str           # e.g. solution.cu
    build_timeout: int
    test_timeout: int
    quotas: dict = field(default_factory=dict)
    used: dict = field(default_factory=dict)  # tool_name -> int
    trace: list[ToolTraceEntry] = field(default_factory=list)
    # Optional callable: env -> (build_ok, build_output) using the current
    # in-memory solution code. Filled in by agent_loop after each write_solution.
    rebuild_fn: Callable[[], tuple[bool, str]] | None = None
    rerun_test_fn: Callable[[], tuple[int | None, str]] | None = None
    # The current candidate code (mutated by write_solution).
    current_code: str = ""

    def quota_remaining(self, tool: str) -> int | None:
        cap = self.quotas.get(tool)
        if cap is None:
            return None
        return cap - self.used.get(tool, 0)


# ---------------------------------------------------------------------------
# JSONSchema definitions — exposed to the LLM via the chat-completions API
# ---------------------------------------------------------------------------


TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_code",
            "description": (
                "Read the current candidate `solution.*` file. By default, "
                "or when `full_file=true`, returns the full file. You may "
                "also pass 1-indexed inclusive `start`/`end` to inspect a "
                "specific range."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "full_file": {"type": "boolean"},
                    "start": {"type": "integer", "minimum": 1},
                    "end": {"type": "integer", "minimum": 1},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_test_file",
            "description": (
                "Read a file from the task's testbench directory (test harness, "
                "headers, reference outputs). Path is relative to the testbench "
                "root, e.g. `test/test_main.cu` or `include/helpers.h`. Use "
                "this to understand what the test expects from your kernel."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start": {"type": "integer", "minimum": 1},
                    "end": {"type": "integer", "minimum": 1},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_prompt",
            "description": (
                "Read the original task prompt (problem description). Useful "
                "to double-check signatures, tolerance, edge cases, and "
                "required outputs."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_compute_sanitizer",
            "description": (
                "Run NVIDIA Compute Sanitizer on the task's test command and "
                "return the first error block plus summary. Supports direct "
                "binaries and wrapped Python/Make/shell test commands via "
                "`--target-processes=all`. `tool` selects memcheck, "
                "racecheck, initcheck, or synccheck. Requires the candidate "
                "to have built successfully."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tool": {
                        "type": "string",
                        "enum": ["memcheck", "racecheck", "initcheck", "synccheck"],
                        "default": "memcheck",
                    },
                    "target_processes": {
                        "type": "string",
                        "enum": ["application-only", "all"],
                        "default": "all",
                        "description": "Track only the root process or root plus child processes.",
                    },
                    "target_processes_filter": {
                        "type": "string",
                        "description": "Optional process filter, e.g. `python` or `regex:python.*`.",
                    },
                    "require_cuda_init": {
                        "type": "boolean",
                        "description": "Whether the root process must initialize CUDA. Use false for wrappers.",
                    },
                    "launch_timeout": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Seconds to wait for sanitizer to connect to the target process.",
                    },
                    "print_limit": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 10,
                        "description": "Maximum number of sanitizer errors to print; 0 means unlimited.",
                    },
                    "print_level": {
                        "type": "string",
                        "enum": ["info", "warn", "error", "fatal"],
                    },
                    "kernel_name": {
                        "type": "string",
                        "description": "Kernel filter, e.g. `kernel_substring=my_kernel` or `regex:...`.",
                    },
                    "kernel_name_exclude": {
                        "type": "string",
                        "description": "Kernel exclusion filter using Compute Sanitizer filter syntax.",
                    },
                    "launch_count": {"type": "integer", "minimum": 0},
                    "launch_skip": {"type": "integer", "minimum": 0},
                    "force_blocking_launches": {"type": "boolean"},
                    "force_synchronization_limit": {"type": "integer", "minimum": 0},
                    "show_backtrace": {
                        "type": "string",
                        "enum": ["no", "host", "device", "yes"],
                    },
                    "backtrace_short": {"type": "boolean"},
                    "num_callers_host": {"type": "integer", "minimum": 0},
                    "num_callers_device": {"type": "integer", "minimum": 0},
                    "check_exit_code": {"type": "boolean"},
                    "nvtx": {"type": "boolean"},
                    "quiet": {"type": "boolean"},
                    "kill": {"type": "boolean"},
                    "max_connections": {"type": "integer", "minimum": 1},
                    "leak_check": {
                        "type": "string",
                        "enum": ["full", "no"],
                        "description": "memcheck only.",
                    },
                    "padding": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "memcheck allocation padding in bytes.",
                    },
                    "report_api_errors": {
                        "type": "string",
                        "enum": ["all", "explicit", "no"],
                        "description": "memcheck API error reporting mode.",
                    },
                    "track_stream_ordered_races": {
                        "type": "string",
                        "enum": ["all", "use-before-alloc", "use-after-free", "no"],
                        "description": "memcheck stream-ordered allocation race tracking.",
                    },
                    "check_cache_control": {"type": "boolean"},
                    "detect_missing_module_unload": {"type": "boolean"},
                    "racecheck_detect_level": {
                        "type": "string",
                        "enum": ["info", "warn", "error"],
                        "description": "racecheck minimum hazard level.",
                    },
                    "racecheck_report": {
                        "type": "string",
                        "enum": ["all", "hazard", "analysis"],
                    },
                    "racecheck_trace_sync": {"type": "boolean"},
                    "racecheck_memcpy_async": {"type": "boolean"},
                    "racecheck_deadlock_timeout": {"type": "integer", "minimum": 0},
                    "racecheck_continue_on_deadlock": {"type": "boolean"},
                    "check_api_memory_access": {
                        "type": "boolean",
                        "description": "initcheck cudaMemcpy/cudaMemset device-memory access checking.",
                    },
                    "track_unused_memory": {"type": "boolean"},
                    "unused_memory_threshold": {"type": "integer", "minimum": 0},
                    "check_optix": {"type": "boolean"},
                    "missing_barrier_init_is_fatal": {
                        "type": "boolean",
                        "description": "synccheck missing cuda::barrier init behavior.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_static_analysis",
            "description": (
                "Run lightweight static checks over the current candidate and "
                "return suspicious CUDA memory/indexing lines. This does not "
                "build or execute the code. Use it from the analyzer phase to "
                "populate static flags before choosing a line-level hypothesis."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "build_only",
            "description": (
                "Re-build the current candidate without running tests. Returns "
                "compiler output. Useful for checking a hypothesis (e.g. did "
                "fixing the syntax error unblock the next layer of errors?) "
                "without consuming an iteration."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "test_only",
            "description": (
                "Re-run the test on the most recently built candidate. Useful "
                "for flaky timing measurements — you can take a second sample "
                "without rebuilding."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "locate_bug",
            "description": (
                "Declare WHERE the bug is — a mandatory step before every "
                "`write_solution` that revises an existing candidate. You "
                "must commit to a specific file and 1-indexed line and state "
                "the concrete evidence that led you there (what you read, "
                "hand-traced, or observed — not a guess). The tool echoes "
                "back the cited line from the current candidate so you can "
                "confirm you pointed at the right place. If your first "
                "location turns out wrong, call locate_bug again with a "
                "different line before submitting."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "description": "File containing the bug, e.g. solution.cu",
                    },
                    "line": {
                        "type": "integer",
                        "description": "1-indexed line number of the buggy line",
                    },
                    "bug_description": {
                        "type": "string",
                        "description": "What is wrong at that line",
                    },
                    "evidence": {
                        "type": "string",
                        "description": (
                            "How you determined this is the bug — the read / "
                            "hand-trace / observation behind the claim"
                        ),
                    },
                },
                "required": ["file", "line", "bug_description", "evidence"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_edit",
            "description": (
                "Apply a surgical edit to the in-memory candidate `solution.*` "
                "by replacing `old_string` with `new_string`, or by replacing "
                "a 1-indexed inclusive `start_line`/`end_line` range. Does "
                "NOT build or test — it just stages the patch. Exact "
                "`old_string` matching is tried first; if that fails, the "
                "tool can fall back to a unique whitespace-tolerant whole-line "
                "match. Use `start_line`/`end_line` when exact text is brittle. "
                "Make as many edits as you need, then call `write_solution` "
                "to build and test the patched candidate."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "old_string": {
                        "type": "string",
                        "description": (
                            "Substring to replace. Exact match is attempted "
                            "first; if match_whitespace is true, a unique "
                            "whole-line whitespace-normalized match may be "
                            "used when exact matching fails. Omit when using "
                            "start_line/end_line."
                        ),
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement text (may be empty to delete).",
                    },
                    "start_line": {
                        "type": "integer",
                        "minimum": 1,
                        "description": (
                            "1-indexed first line to replace. Must be used "
                            "with end_line. Takes precedence over old_string."
                        ),
                    },
                    "end_line": {
                        "type": "integer",
                        "minimum": 1,
                        "description": (
                            "1-indexed inclusive last line to replace. Must "
                            "be used with start_line."
                        ),
                    },
                    "replace_all": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "Replace every occurrence of old_string. Only "
                            "set true when intentional (e.g. renaming a "
                            "variable everywhere)."
                        ),
                    },
                    "match_whitespace": {
                        "type": "boolean",
                        "default": True,
                        "description": (
                            "If exact old_string matching fails, allow a "
                            "unique whole-line match that ignores whitespace "
                            "differences within each line."
                        ),
                    },
                },
                "required": ["new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_solution",
            "description": (
                "Submit a correct, performance-preserving candidate for build "
                "+ test + classification. "
                "Counts as ONE iteration against the per-task iteration "
                "budget. In full-file submission mode, pass a complete "
                "revised `code` file to replace the current candidate and "
                "submit it in one step. Legacy patch-submit mode may omit "
                "`code` to commit staged `apply_edit` changes. Must be preceded by "
                "`locate_bug` in this iteration; an un-located submission is "
                "rejected."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rationale": {
                        "type": "string",
                        "description": "Short reason for this submission (recorded in trace).",
                    },
                    "code": {
                        "type": "string",
                        "description": (
                            "Complete replacement for the current solution "
                            "file. Required in full-file submission mode. "
                            "Preserve performance-critical parallel structure; "
                            "do not submit a serial, fallback, or much slower "
                            "correctness-only rewrite."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Executors
# ---------------------------------------------------------------------------


def _truncate(text: str, cap: int) -> tuple[str, bool]:
    if len(text) <= cap:
        return text, False
    half = cap // 2
    return text[:half] + "\n... [middle truncated] ...\n" + text[-half:], True


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _safe_read(path: Path, start: int | None, end: int | None, cap: int) -> ToolResult:
    if not path.exists():
        return ToolResult(ok=False, content="", error=f"file not found: {path}")
    if path.suffix and path.suffix not in ALLOWED_READ_SUFFIXES:
        return ToolResult(
            ok=False, content="",
            error=f"refusing to read {path.suffix} (allow-list: {ALLOWED_READ_SUFFIXES})",
        )
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return ToolResult(ok=False, content="", error=f"{type(e).__name__}: {e}")

    lines = text.splitlines()
    total = len(lines)
    if total == 0:
        return ToolResult(ok=True, content="", meta={"total_lines": 0, "range": [0, 0]})
    s = max(1, start or 1)
    e = min(total, end or total)
    if s > total:
        return ToolResult(
            ok=False,
            content="",
            error=f"requested start line {s} is out of range; file has {total} lines",
            meta={"total_lines": total, "range": [s, e]},
        )
    if e < s:
        return ToolResult(
            ok=False,
            content="",
            error=f"invalid line range [{s}, {e}] for file with {total} lines",
            meta={"total_lines": total, "range": [s, e]},
        )
    if e - s + 1 > MAX_READ_LINES:
        e = min(total, s + MAX_READ_LINES - 1)
    sliced = lines[s - 1:e]
    body = "\n".join(f"{s + i:5d} | {line}" for i, line in enumerate(sliced))
    content, truncated = _truncate(body, cap)
    return ToolResult(
        ok=True, content=content, truncated=truncated,
        meta={"total_lines": total, "range": [s, e]},
    )


def _tool_read_code(args: dict, env: ToolEnv) -> ToolResult:
    if not env.current_code:
        return ToolResult(ok=False, content="", error="no candidate code yet — call write_solution first")
    lines = env.current_code.splitlines()
    total = len(lines)
    if total == 0:
        return ToolResult(ok=True, content="", meta={"total_lines": 0, "range": [0, 0]})

    full_file = bool(args.get("full_file", False)) or (
        "start" not in args and "end" not in args
    )
    if full_file:
        s, e = 1, total
    else:
        start = args.get("start", 1)
        end = args.get("end", start + 50)
        s = max(1, int(start))
        e = min(total, int(end))
    if s > total:
        return ToolResult(
            ok=False,
            content="",
            error=f"requested start line {s} is out of range; candidate has {total} lines",
            meta={"total_lines": total, "range": [s, e]},
        )
    if e < s:
        return ToolResult(
            ok=False,
            content="",
            error=f"invalid line range [{s}, {e}] for candidate with {total} lines",
            meta={"total_lines": total, "range": [s, e]},
        )
    if not full_file and e - s + 1 > MAX_READ_LINES:
        e = min(total, s + MAX_READ_LINES - 1)
    body = "\n".join(f"{s + i:5d} | {line}" for i, line in enumerate(lines[s - 1:e]))
    content, truncated = _truncate(body, RESULT_CAPS["read_code"])
    return ToolResult(
        ok=True, content=content, truncated=truncated,
        meta={"total_lines": total, "range": [s, e], "full_file": full_file},
    )


def _tool_read_test_file(args: dict, env: ToolEnv) -> ToolResult:
    rel = args.get("path", "")
    if not rel:
        return ToolResult(ok=False, content="", error="missing 'path'")
    # Path confinement: resolve and assert under testbench_dir.
    candidate = (env.testbench_dir / rel).resolve()
    allowed_roots = [env.testbench_dir.resolve()]
    external_root = (env.dataset_dir / "data" / "_external").resolve()
    if external_root.exists():
        allowed_roots.append(external_root)
    try:
        if not any(_is_relative_to(candidate, root) for root in allowed_roots):
            raise ValueError
    except ValueError:
        return ToolResult(ok=False, content="",
                          error=f"path escapes testbench root: {rel}")
    return _safe_read(candidate, args.get("start"), args.get("end"),
                      cap=RESULT_CAPS["read_test_file"])


def _tool_read_prompt(args: dict, env: ToolEnv) -> ToolResult:
    return _safe_read(env.prompt_path, args.get("start"), args.get("end"),
                      cap=RESULT_CAPS["read_prompt"])


def _line_context(lines: list[str], line_no: int, radius: int = 1) -> str:
    lo = max(1, line_no - radius)
    hi = min(len(lines), line_no + radius)
    return "\n".join(f"{i:5d} | {lines[i - 1]}" for i in range(lo, hi + 1))


def _tool_run_static_analysis(args: dict, env: ToolEnv) -> ToolResult:
    """Heuristic static CUDA checks for the blackboard analyzer.

    This is intentionally conservative: it flags places worth inspecting,
    not proven bugs. The LLM still has to connect a flag to the task spec,
    runtime evidence, and a concrete line-level diagnosis.
    """
    if not env.current_code.strip():
        return ToolResult(ok=False, content="", error="no candidate code yet")

    flags: list[dict[str, Any]] = []
    lines = env.current_code.splitlines()
    guard_stack: list[int] = []

    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        lowered = stripped.lower()

        if re.search(r"\bif\s*\(", stripped) and re.search(
            r"\b(n|m|k|width|height|size|len|num|rows|cols|count|total|dim)\b",
            stripped,
            flags=re.IGNORECASE,
        ):
            guard_stack.append(idx)

        def add(flag: str, reason: str, severity: str = "medium") -> None:
            flags.append({
                "line": idx,
                "flag": flag,
                "severity": severity,
                "reason": reason,
                "code": stripped[:240],
            })

        has_thread_math = any(tok in stripped for tok in (
            "threadIdx", "blockIdx", "blockDim", "gridDim",
        ))
        has_array_access = bool(re.search(r"\w+\s*\[[^\]]+\]", stripped))
        has_write = has_array_access and bool(re.search(r"\w+\s*\[[^\]]+\]\s*(=|\+=|-=|\*=|/=)", stripped))
        recent_guard = guard_stack and idx - guard_stack[-1] <= 5

        if has_array_access and has_thread_math:
            add(
                "global_or_shared_index",
                "array access uses thread/block-derived indexing; verify mapping and bounds",
                "high" if not recent_guard else "medium",
            )
            if not recent_guard:
                add(
                    "possible_unchecked_index",
                    "no obvious nearby bounds guard in the previous few lines",
                    "high",
                )

        if "__shared__" in stripped or "extern __shared__" in stripped:
            add("shared_memory_declaration", "shared memory sizing/indexing must match block geometry")

        if has_write and "__syncthreads" not in stripped and "atomic" not in stripped.lower():
            add("memory_write", "write target may need uniqueness proof or synchronization/atomic protection")

        if "__syncthreads" in stripped:
            add("sync_boundary", "synchronization point; check all threads reach it consistently")

        if any(tok in lowered for tok in ("cudamalloc", "cudamemcpy", "malloc(", "new ", "delete", "free(")):
            add("allocation_or_copy", "allocation/copy sizes and ownership are common memory-failure sources")

        if "asm" in lowered or "mma." in lowered or "ldmatrix" in lowered or ".shared" in lowered:
            add("inline_ptx_or_sass_sensitive", "inline PTX/SASS-level operation; check constraints, alignment, and address space")

        if re.search(r"\b(float4|int4|half2|uint4|ulonglong|reinterpret_cast)\b", stripped):
            add("alignment_sensitive", "vectorized/reinterpreted access may require alignment and size divisibility")

    # De-duplicate repeated flags on the same line while preserving order.
    seen: set[tuple[int, str]] = set()
    deduped: list[dict[str, Any]] = []
    for f in flags:
        key = (int(f["line"]), str(f["flag"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(f)
    flags = deduped[:80]

    if not flags:
        return ToolResult(
            ok=True,
            content="No obvious static CUDA memory/indexing flags found.",
            meta={"flags": []},
        )

    lines_out = ["Static CUDA flags (heuristic, not proof):"]
    for f in flags[:40]:
        lines_out.append(
            f"- line {f['line']}: {f['flag']} [{f['severity']}] - {f['reason']}"
        )
        lines_out.append(f"  code: {f['code']}")
    if len(flags) > 40:
        lines_out.append(f"... {len(flags) - 40} more flags omitted ...")

    hot_lines = sorted({int(f["line"]) for f in flags[:12]})
    if hot_lines:
        lines_out.append("\nContext for top flagged lines:")
        for line_no in hot_lines:
            lines_out.append(_line_context(lines, line_no, radius=1))

    content, truncated = _truncate("\n".join(lines_out), RESULT_CAPS["run_static_analysis"])
    return ToolResult(
        ok=True,
        content=content,
        truncated=truncated,
        meta={"flags": flags},
    )


def _tool_run_compute_sanitizer(args: dict, env: ToolEnv) -> ToolResult:
    if not sanitizer_available():
        return ToolResult(ok=False, content="", error="compute-sanitizer not on PATH")
    tool = args.get("tool", "memcheck")
    options = {k: v for k, v in args.items() if k != "tool"}
    test_cmd = env.input_meta.get("test_command", "")
    result: DiagResult = run_compute_sanitizer(
        test_cmd,
        env.task_workdir,
        tool=tool,
        timeout=env.test_timeout * 3,
        options=options,
    )
    if not result.ok:
        return ToolResult(ok=False, content="", error=result.error)
    return ToolResult(
        ok=True, content=result.to_prompt_block(),
        truncated=result.truncated, meta=result.meta,
    )


def _tool_build_only(args: dict, env: ToolEnv) -> ToolResult:
    if env.rebuild_fn is None:
        return ToolResult(ok=False, content="", error="rebuild not wired (no candidate code)")
    build_ok, build_output = env.rebuild_fn()
    body = ("BUILD OK\n" if build_ok else "BUILD FAILED\n") + (build_output or "")
    content, truncated = _truncate(body, RESULT_CAPS["build_only"])
    return ToolResult(
        ok=True, content=content, truncated=truncated,
        meta={"build_ok": build_ok},
    )


def _tool_test_only(args: dict, env: ToolEnv) -> ToolResult:
    if env.rerun_test_fn is None:
        return ToolResult(ok=False, content="", error="no built binary to test")
    rc, test_output = env.rerun_test_fn()
    body = f"RETURNCODE: {rc}\n\n{test_output or ''}"
    content, truncated = _truncate(body, RESULT_CAPS["test_only"])
    return ToolResult(
        ok=True, content=content, truncated=truncated,
        meta={"returncode": rc},
    )


def _tool_locate_bug(args: dict, env: ToolEnv) -> ToolResult:
    """Record a bug-location claim and echo back the cited line.

    Does not build or test — it is a declaration. The loop reads
    `meta["located"]` to satisfy the pre-write_solution localisation
    requirement and to stamp the location onto the iteration record.
    """
    file = args.get("file", "")
    line = args.get("line")
    desc = args.get("bug_description", "")
    evidence = args.get("evidence", "")
    if not file or line is None or not desc or not evidence:
        return ToolResult(
            ok=False, content="",
            error="locate_bug requires file, line, bug_description, evidence",
        )
    try:
        line = int(line)
    except (TypeError, ValueError):
        return ToolResult(ok=False, content="", error="'line' must be an integer")

    # Echo the cited line from the current candidate so the model can
    # confirm it pointed at the right place.
    echo = ""
    is_solution = file.endswith(env.solution_file) or file == env.solution_file \
        or file.endswith("solution.cu")
    if env.current_code and is_solution:
        lines = env.current_code.splitlines()
        if 1 <= line <= len(lines):
            lo, hi = max(1, line - 3), min(len(lines), line + 3)
            echo = "\n".join(
                f"{'>>' if i == line else '  '} {i:4d} | {lines[i - 1]}"
                for i in range(lo, hi + 1)
            )
        else:
            echo = (f"(line {line} is out of range — the current candidate "
                    f"has {len(lines)} lines)")

    body = f"Bug location recorded: {file}:{line}\n{desc}"
    if echo:
        body += f"\n\nCited line in the current candidate:\n{echo}"
    return ToolResult(
        ok=True, content=body,
        meta={
            "located": True,
            "file": file,
            "line": line,
            "bug_description": desc[:500],
            "evidence": evidence[:1000],
        },
    )


def _normalise_ws_line(line: str) -> str:
    """Collapse intra-line whitespace for tolerant edit matching."""
    return re.sub(r"\s+", " ", line.strip())


def _replace_line_range(code: str, start_line: int, end_line: int, new: str) -> tuple[str, str]:
    lines = code.splitlines()
    total = len(lines)
    if start_line < 1 or end_line < 1:
        raise ValueError("start_line/end_line must be 1-indexed positive integers")
    if start_line > end_line:
        raise ValueError(f"invalid line range [{start_line}, {end_line}]")
    if start_line > total:
        raise ValueError(
            f"start_line {start_line} is out of range; candidate has {total} lines"
        )
    if end_line > total:
        raise ValueError(
            f"end_line {end_line} is out of range; candidate has {total} lines"
        )

    replacement = new.splitlines()
    had_final_newline = code.endswith("\n")
    lines[start_line - 1:end_line] = replacement
    patched = "\n".join(lines)
    if had_final_newline:
        patched += "\n"
    body = (
        f"Line-range edit applied (lines {start_line}-{end_line} replaced "
        f"with {len(replacement)} line{'s' if len(replacement) != 1 else ''})."
    )
    return patched, body


def _find_whitespace_tolerant_line_matches(code: str, old: str) -> list[tuple[int, int]]:
    old_lines = old.splitlines()
    if not old_lines:
        return []
    norm_old = [_normalise_ws_line(line) for line in old_lines]
    code_lines = code.splitlines()
    if len(old_lines) > len(code_lines):
        return []
    matches: list[tuple[int, int]] = []
    width = len(old_lines)
    for idx in range(0, len(code_lines) - width + 1):
        window = code_lines[idx:idx + width]
        if [_normalise_ws_line(line) for line in window] == norm_old:
            matches.append((idx + 1, idx + width))
    return matches


def _tool_apply_edit(args: dict, env: ToolEnv) -> ToolResult:
    """Surgical replace on `env.current_code` — does not build or test.

    Exact `old_string` replacement is preferred. If exact matching fails,
    the tool can fall back to unique whole-line whitespace-tolerant matching,
    or the caller can bypass string matching with start_line/end_line.
    """
    old = args.get("old_string", "")
    new = args.get("new_string", "")
    replace_all = bool(args.get("replace_all", False))
    match_whitespace = bool(args.get("match_whitespace", True))
    if not isinstance(new, str):
        return ToolResult(ok=False, content="", error="apply_edit 'new_string' must be a string")
    if not env.current_code.strip():
        return ToolResult(
            ok=False, content="",
            error="no candidate to edit — env.current_code is empty (use write_solution for the first version)",
        )

    has_start = args.get("start_line") is not None
    has_end = args.get("end_line") is not None
    match_strategy = "exact"
    n_replaced = 1
    if has_start or has_end:
        if not (has_start and has_end):
            return ToolResult(
                ok=False, content="",
                error="apply_edit line-range mode requires both start_line and end_line",
            )
        try:
            start_line = int(args.get("start_line"))
            end_line = int(args.get("end_line"))
            env.current_code, body = _replace_line_range(
                env.current_code, start_line, end_line, new
            )
        except (TypeError, ValueError) as e:
            return ToolResult(ok=False, content="", error=str(e))
        match_strategy = "line_range"
        line_num = start_line
    else:
        if not isinstance(old, str) or not old:
            return ToolResult(
                ok=False, content="",
                error="apply_edit requires non-empty 'old_string' unless start_line/end_line are provided",
            )
        count = env.current_code.count(old)
        if count == 0 and match_whitespace:
            matches = _find_whitespace_tolerant_line_matches(env.current_code, old)
            if len(matches) == 1 or (replace_all and matches):
                selected = matches if replace_all else matches[:1]
                for start_line, end_line in reversed(selected):
                    env.current_code, _ = _replace_line_range(
                        env.current_code, start_line, end_line, new
                    )
                n_replaced = len(selected)
                match_strategy = "whitespace_tolerant_lines"
                line_num = selected[0][0]
                body = (
                    "Edit applied via whitespace-tolerant line match "
                    f"({n_replaced} occurrence{'s' if n_replaced != 1 else ''} replaced)."
                )
            elif len(matches) > 1:
                return ToolResult(
                    ok=False, content="",
                    error=(
                        "old_string not found as an exact match, and the "
                        f"whitespace-tolerant line match found {len(matches)} "
                        "occurrences. Add start_line/end_line or more context."
                    ),
                )
            else:
                return ToolResult(
                    ok=False, content="",
                    error=(
                        "old_string not found in candidate. Whitespace-tolerant "
                        "line matching also failed; use read_code and then "
                        "start_line/end_line or write_solution(code=...) as a "
                        "full-file fallback."
                    ),
                )
        elif count == 0:
            return ToolResult(
                ok=False, content="",
                error=(
                    "old_string not found in candidate. Use read_code and then "
                    "start_line/end_line, enable match_whitespace, or use "
                    "write_solution(code=...) as a full-file fallback."
                ),
            )
        else:
            if count > 1 and not replace_all:
                return ToolResult(
                    ok=False, content="",
                    error=(
                        f"old_string appears {count} times in candidate. Add more "
                        f"surrounding context to make it unique, or set "
                        f"replace_all=true to patch every occurrence."
                    ),
                )

            n_replaced = count if replace_all else 1
            env.current_code = (
                env.current_code.replace(old, new)
                if replace_all
                else env.current_code.replace(old, new, 1)
            )
            body = (
                f"Edit applied ({n_replaced} occurrence"
                f"{'s' if n_replaced != 1 else ''} replaced)."
            )
            pos = env.current_code.find(new) if new else -1
            line_num = env.current_code[:pos].count("\n") + 1 if pos >= 0 else 1

    # Surface the edited region for the model to confirm placement.
    if new or match_strategy == "line_range":
        new_lines = new.count("\n")
        lines = env.current_code.splitlines()
        lo = max(1, line_num - 3)
        hi = min(len(lines), line_num + new_lines + 3)
        window = "\n".join(
            f"{i:5d} | {lines[i - 1]}" for i in range(lo, hi + 1)
        )
        body += f"\nEdited region:\n{window}"
    return ToolResult(
        ok=True, content=body,
        meta={
            "edited": True,
            "replace_all": replace_all,
            "occurrences": n_replaced,
            "match_strategy": match_strategy,
        },
    )


def _tool_write_solution(args: dict, env: ToolEnv) -> ToolResult:
    """Special-cased — actual submission is handled by agent_loop, not here.

    Usually commits whatever `env.current_code` currently holds (i.e. the
    broken_start plus all `apply_edit` patches staged this iteration). As a
    fallback for brittle surgical matching, `code` may contain a complete
    replacement for the solution file. The dispatcher returns `meta["submit"]`
    and the loop intercepts it for build+test.

    Rejects if `env.current_code` is empty (the agent loop is debug-only,
    so a missing candidate is a configuration error, not something the
    model can recover from).
    """
    rationale = args.get("rationale", "")
    code = args.get("code")
    full_replacement = False
    if code is not None:
        if not isinstance(code, str) or not code.strip():
            return ToolResult(
                ok=False, content="",
                error="write_solution 'code' must be a non-empty full solution file",
            )
        env.current_code = code
        full_replacement = True
    if not env.current_code.strip():
        return ToolResult(
            ok=False, content="",
            error=(
                "nothing to submit — env.current_code is empty. This is "
                "a configuration error (no broken_start seeded); the "
                "agent loop is debug-only."
            ),
        )
    return ToolResult(
        ok=True,
        content=(
            "(submission queued — full-file replacement, build+test pending)"
            if full_replacement
            else "(submission queued — patch-submit, build+test pending)"
        ),
        meta={
            "submit": True,
            "rationale": rationale[:500],
            "full_replacement": full_replacement,
        },
    )


EXECUTORS: dict[str, Callable[[dict, ToolEnv], ToolResult]] = {
    "read_code": _tool_read_code,
    "read_test_file": _tool_read_test_file,
    "read_prompt": _tool_read_prompt,
    "run_static_analysis": _tool_run_static_analysis,
    "run_compute_sanitizer": _tool_run_compute_sanitizer,
    "build_only": _tool_build_only,
    "test_only": _tool_test_only,
    "locate_bug": _tool_locate_bug,
    "apply_edit": _tool_apply_edit,
    "write_solution": _tool_write_solution,
}


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def call_tool(
    name: str,
    args_json: str | dict,
    env: ToolEnv,
    iteration: int,
) -> ToolResult:
    """Dispatch a tool call, enforce quotas, record a trace entry."""
    if name not in EXECUTORS:
        return ToolResult(ok=False, content="", error=f"unknown tool: {name}")

    # Parse args
    if isinstance(args_json, str):
        try:
            args = json.loads(args_json) if args_json else {}
        except json.JSONDecodeError as e:
            return ToolResult(ok=False, content="", error=f"invalid JSON args: {e}")
    else:
        args = dict(args_json or {})

    # Quota check
    remaining = env.quota_remaining(name)
    if remaining is not None and remaining <= 0:
        return ToolResult(
            ok=False, content="",
            error=f"quota exhausted for {name} (limit {env.quotas[name]})",
        )

    # Execute
    t0 = time.time()
    try:
        result = EXECUTORS[name](args, env)
    except Exception as e:
        result = ToolResult(ok=False, content="", error=f"{type(e).__name__}: {e}")
    elapsed_ms = (time.time() - t0) * 1000

    # Accounting
    env.used[name] = env.used.get(name, 0) + 1
    env.trace.append(ToolTraceEntry(
        iteration=iteration,
        tool=name,
        args=args,
        ok=result.ok,
        truncated=result.truncated,
        content_chars=len(result.content or ""),
        error=result.error,
        elapsed_ms=elapsed_ms,
    ))
    return result


def make_env(
    task_stem: str,
    task_workdir: Path,
    dataset_dir: Path,
    input_meta: dict,
    build_timeout: int,
    test_timeout: int,
    quotas_override: dict | None = None,
) -> ToolEnv:
    quotas = dict(DEFAULT_QUOTAS)
    quotas.update(quotas_override or {})
    try:
        task_paths = resolve_task_paths(task_stem, dataset_dir)
        testbench_dir = task_paths.testbench_path
        prompt_path = task_paths.prompt_path
    except FileNotFoundError:
        # Unit tests and ad-hoc tool harnesses often pass a minimal dataset
        # directory with only prompts/testbench. Preserve that lightweight path.
        testbench_dir = dataset_dir / "testbench" / task_stem
        prompt_path = dataset_dir / "prompts" / f"{task_stem}.txt"
    return ToolEnv(
        task_stem=task_stem,
        task_workdir=task_workdir,
        dataset_dir=dataset_dir,
        testbench_dir=testbench_dir,
        prompt_path=prompt_path,
        input_meta=input_meta,
        solution_file=input_meta.get("solution_file", "solution.cu"),
        build_timeout=build_timeout,
        test_timeout=test_timeout,
        quotas=quotas,
    )


def schemas_for_config(
    allowed_tools: list[str] | None = None,
    *,
    require_full_solution_code: bool = False,
) -> list[dict]:
    """Return the JSONSchema list filtered to `allowed_tools` (or all)."""
    if not allowed_tools:
        selected = TOOL_SCHEMAS
    else:
        allowed = set(allowed_tools)
        selected = [s for s in TOOL_SCHEMAS if s["function"]["name"] in allowed]
    schemas = [copy.deepcopy(s) for s in selected]
    if require_full_solution_code:
        for schema in schemas:
            fn = schema["function"]
            if fn["name"] != "write_solution":
                continue
            params = fn.setdefault("parameters", {"type": "object", "properties": {}})
            params["required"] = ["code"]
            fn["description"] = (
                "Submit a complete revised solution file for build + test + "
                "classification. The `code` argument is required and must be "
                "the full replacement file, not a diff or snippet. Counts as "
                "ONE iteration against the per-task iteration budget. Must be "
                "preceded by `locate_bug` in this iteration."
            )
            props = params.setdefault("properties", {})
            if "code" in props:
                props["code"]["description"] = (
                    "Required complete replacement for the current solution "
                    "file. Include the whole file, not just edited lines."
                )
    return schemas
