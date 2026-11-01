"""
Error-aware feedback prompt builder.

Instead of a single FEEDBACK_TEMPLATE for all error types,
selects the most effective prompt based on the error category.
Extracts targeted information (error line, expected vs actual, etc.)
to help the LLM focus on the right fix.
"""

from __future__ import annotations

import re


def build_feedback(
    classification: dict,
    build_output: str,
    test_output: str | None,
    previous_code: str,
    level: int = 3,
    ncu_data: dict | None = None,
    mode: str = "aware",
    *,
    sanitizer_summary: dict | None = None,
    repair_mode: str | None = None,
    minimal_patch: bool = False,
) -> str | None:
    """Build a feedback message at the requested signal level (Exp 3).

    Level matrix:
      L0 — return None (caller resets messages; no carryover)
      L1 — G only: previous code, no errors
      L2 — G+B: previous code + build error (skip test info)
      L3 — G+B+T: full error-aware feedback (current default)
      L4 — G+B+T+P: L3 + ncu profiler metrics (if provided)

    Mode (only honoured at level=3, no-op elsewhere):
      "aware" — current 6-way category dispatch (default)
      "raw"   — bypass dispatch; dump build+test via _feedback_generic
    """
    if mode not in ("aware", "raw"):
        raise ValueError(
            f"build_feedback: mode must be 'aware' or 'raw', got {mode!r}"
        )
    if level == 0:
        return None
    if level == 1:
        return _append_agent_context(
            _feedback_l1_g_only(previous_code),
            sanitizer_summary=sanitizer_summary,
            repair_mode=repair_mode,
            minimal_patch=minimal_patch,
        )
    if level == 2:
        return _append_agent_context(
            _feedback_l2_gb(classification, build_output, previous_code),
            sanitizer_summary=sanitizer_summary,
            repair_mode=repair_mode,
            minimal_patch=minimal_patch,
        )
    if level == 4:
        l3_msg = _feedback_l3(classification, build_output, test_output, previous_code)
        if ncu_data:
            l3_msg = l3_msg + "\n\n## Profiler Metrics (ncu)\n" + _format_ncu(ncu_data)
        return _append_agent_context(
            l3_msg,
            sanitizer_summary=sanitizer_summary,
            repair_mode=repair_mode,
            minimal_patch=minimal_patch,
        )
    # Level 3
    if mode == "raw":
        msg = _feedback_generic(build_output, test_output, previous_code)
    else:
        msg = _feedback_l3(classification, build_output, test_output, previous_code)
    return _append_agent_context(
        msg,
        sanitizer_summary=sanitizer_summary,
        repair_mode=repair_mode,
        minimal_patch=minimal_patch,
    )


def _append_agent_context(
    feedback: str | None,
    *,
    sanitizer_summary: dict | None,
    repair_mode: str | None,
    minimal_patch: bool,
) -> str | None:
    if feedback is None:
        return None
    additions = []
    if repair_mode:
        additions.extend(["", "## Repair Mode", str(repair_mode)])
    sanitizer_text = _format_sanitizer_summary(sanitizer_summary)
    if sanitizer_text:
        additions.extend(["", sanitizer_text])
    if minimal_patch:
        additions.extend([
            "",
            "## Patch Scope",
            "Make the smallest change that addresses the current failure. Avoid unrelated rewrites.",
        ])
    if not additions:
        return feedback
    return feedback.rstrip() + "\n" + "\n".join(additions)


def _format_sanitizer_summary(sanitizer_summary: dict | None) -> str:
    if not sanitizer_summary or not sanitizer_summary.get("triggered"):
        return ""

    lines = ["## Sanitizer Diagnostics"]
    real_hits = sanitizer_summary.get("real_hit_tools") or []
    hit_tools = sanitizer_summary.get("hit_tools") or []
    timed_out = sanitizer_summary.get("timed_out_tools") or []
    unavailable = sanitizer_summary.get("unavailable_tools") or []
    if real_hits:
        lines.append(f"- real_hits: {', '.join(map(str, real_hits))}")
    elif hit_tools:
        lines.append(f"- hit_tools: {', '.join(map(str, hit_tools))}")
    else:
        lines.append("- hits: none")
    if timed_out:
        lines.append(f"- timed_out: {', '.join(map(str, timed_out))}")
    if unavailable:
        lines.append(f"- unavailable: {', '.join(map(str, unavailable))}")

    for tool, info in (sanitizer_summary.get("results") or {}).items():
        if not isinstance(info, dict):
            continue
        lines.append(f"- tool: {tool}")
        if info.get("summary"):
            lines.append(f"  summary: {_one_line(str(info.get('summary')), 500)}")
        if info.get("error"):
            lines.append(f"  error: {_one_line(str(info.get('error')), 300)}")
        finding = info.get("primary_finding")
        if finding:
            lines.append(f"  primary_finding: {_one_line(str(finding), 500)}")
        output = info.get("output") or ""
        if output:
            lines.extend([
                "  output_excerpt:",
                "  ```",
                _truncate(str(output), 1200),
                "  ```",
            ])
    return "\n".join(lines)


def _one_line(text: str, max_len: int) -> str:
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _feedback_l1_g_only(previous_code: str) -> str:
    """L1: previous code only, no error info."""
    return "\n".join([
        "Your previous attempt did not pass. Try a different approach.",
        "",
        "## Your Previous Code",
        "```cuda",
        _truncate(previous_code, 2000),
        "```",
        "",
        "Output ONLY the corrected content of `solution.cu`.",
    ])


def _feedback_l2_gb(
    classification: dict, build_output: str, previous_code: str
) -> str:
    """L2: previous code + build error (skip test feedback)."""
    build_ok = classification.get("build_ok", False)
    if not build_ok:
        # Build failed — show compiler errors
        error_lines = []
        for line in build_output.splitlines():
            if re.search(r"\berror\b", line, re.IGNORECASE) and not re.search(
                r"\bwarning\b", line, re.IGNORECASE
            ):
                error_lines.append(line.strip())
        error_text = "\n".join(error_lines[:5]) if error_lines else build_output[:1000]
        parts = [
            "Your code failed to compile.",
            "",
            "## Compiler Errors",
            "```",
            error_text,
            "```",
        ]
    else:
        # Build succeeded but failed at test — L2 intentionally omits T.
        parts = [
            "Your code compiled but the test failed. Try a different approach.",
        ]

    parts += [
        "",
        "## Your Previous Code",
        "```cuda",
        _truncate(previous_code, 2000),
        "```",
        "",
        "Output ONLY the corrected content of `solution.cu`.",
    ]
    return "\n".join(parts)


def _format_ncu(ncu: dict) -> str:
    """Render ncu metric dict for prompt (compact, ~6 metrics + runtime)."""
    lines = []
    if "achieved_occupancy_pct" in ncu:
        lines.append(f"  Achieved Occupancy: {ncu['achieved_occupancy_pct']:.1f}%")
    if "dram_throughput_pct" in ncu:
        lines.append(f"  DRAM Throughput: {ncu['dram_throughput_pct']:.1f}% of peak")
    if "sm_efficiency_pct" in ncu:
        lines.append(f"  SM Efficiency: {ncu['sm_efficiency_pct']:.1f}%")
    if "branch_stall_pct" in ncu:
        lines.append(f"  Branch Stall: {ncu['branch_stall_pct']:.1f}%")
    if "warp_eligibility_pct" in ncu:
        lines.append(f"  Warp Eligibility: {ncu['warp_eligibility_pct']:.1f}%")
    if "long_scoreboard_pct" in ncu:
        lines.append(f"  Long Scoreboard Stall: {ncu['long_scoreboard_pct']:.1f}%")
    if "candidate_runtime_ms" in ncu and "reference_runtime_ms" in ncu:
        cand, ref = ncu["candidate_runtime_ms"], ncu["reference_runtime_ms"]
        ratio = cand / ref if ref > 0 else float("inf")
        lines.append(
            f"  Runtime: candidate {cand:.3f} ms vs reference {ref:.3f} ms ({ratio:.2f}x)"
        )
    return "\n".join(lines) if lines else "  (no metrics collected)"


def _feedback_l3(
    classification: dict,
    build_output: str,
    test_output: str | None,
    previous_code: str,
) -> str:
    """L3: error-aware feedback. 6-way dispatch: compile / memory / timeout / perf / wrong_result / environment."""
    category = classification.get(
        "category", classification.get("five_category", "unknown")
    )
    if category in ("buildability_bug", "compile_error", "integration_bug"):
        return _feedback_compile(build_output, previous_code)
    elif category in ("illegal_memory_access", "out_of_memory", "memory_crash"):
        return _feedback_memory(test_output or "", previous_code)
    elif category == "timeout":
        return _feedback_timeout(previous_code)
    elif category == "perf_below_threshold":
        return _feedback_perf(test_output or "", previous_code, classification)
    elif category in ("functional_correctness_bug", "logic_error", "wrong_result"):
        return _feedback_wrong_result(test_output or "", previous_code)
    elif category == "environment_dependency_bug":
        return _feedback_environment(build_output, test_output or "", previous_code)
    else:
        return _feedback_generic(build_output, test_output, previous_code)


def _feedback_compile(build_output: str, code: str) -> str:
    """Extract specific error lines and surrounding code context."""
    error_lines = []
    for line in build_output.splitlines():
        if re.search(r"\berror\b", line, re.IGNORECASE) and not re.search(r"\bwarning\b", line, re.IGNORECASE):
            error_lines.append(line.strip())

    # Try to extract line number from first error
    code_context = ""
    if error_lines:
        m = re.search(r"solution\.\w+\((\d+)\)", error_lines[0])
        if m:
            line_num = int(m.group(1))
            code_context = _get_code_window(code, line_num, window=5)

    error_text = "\n".join(error_lines[:5]) if error_lines else build_output[:1000]

    parts = [
        "Your code failed to compile. Fix the compilation error below.",
        "",
        "## Compiler Errors",
        "```",
        error_text,
        "```",
    ]

    if code_context:
        parts += [
            "",
            "## Code Around Error",
            "```cuda",
            code_context,
            "```",
        ]

    parts += [
        "",
        "## Your Previous Code",
        "```cuda",
        _truncate(code, 2000),
        "```",
        "",
        "Fix the compilation error. Output ONLY the corrected content of `solution.cu`.",
    ]

    return "\n".join(parts)


def _feedback_memory(test_output: str, code: str) -> str:
    """Extract crash message and suggest common causes."""
    crash_line = ""
    for line in test_output.splitlines():
        if any(p in line.lower() for p in ["segmentation fault", "illegal memory", "out of memory", "double free"]):
            crash_line = line.strip()
            break

    return "\n".join([
        "Your code compiled but crashed with a memory error at runtime.",
        "",
        "## Error",
        f"```",
        crash_line or test_output[:500],
        "```",
        "",
        "## Common Causes",
        "- Out-of-bounds array access (check grid/block dimensions vs array sizes)",
        "- Shared memory overflow (check `__shared__` allocations fit in 48KB/96KB)",
        "- Uninitialized device pointers (check cudaMalloc before kernel launch)",
        "- Race condition on global memory (check atomics or synchronization)",
        "",
        "## Your Previous Code",
        "```cuda",
        _truncate(code, 2000),
        "```",
        "",
        "Fix the memory error. Output ONLY the corrected content of `solution.cu`.",
    ])


def _feedback_timeout(code: str) -> str:
    """Suggest performance / hang causes."""
    return "\n".join([
        "Your code compiled but execution timed out.",
        "",
        "## Likely Causes",
        "- Infinite loop (check loop termination conditions and convergence criteria)",
        "- Excessive grid size (check grid dimensions — too many blocks cause launch overhead)",
        "- Synchronization deadlock (`__syncthreads()` inside divergent branch)",
        "- O(N^2) algorithm where O(N log N) or O(N) is needed",
        "",
        "## Your Previous Code",
        "```cuda",
        _truncate(code, 2000),
        "```",
        "",
        "Fix the timeout issue. Output ONLY the corrected content of `solution.cu`.",
    ])


def _feedback_wrong_result(test_output: str, code: str) -> str:
    """Extract assertion / expected-vs-actual from test output."""
    assertion_line = ""
    expected_actual = ""

    for line in test_output.splitlines():
        if "assertion" in line.lower() or "assert" in line.lower():
            assertion_line = line.strip()
        if "expected" in line.lower() and "actual" in line.lower():
            expected_actual = line.strip()
        if "mismatch" in line.lower():
            expected_actual = expected_actual or line.strip()

    parts = [
        "Your code compiled and ran without crashes, but produced wrong results.",
        "",
        "## Test Failure",
        "```",
    ]

    if assertion_line:
        parts.append(assertion_line)
    if expected_actual:
        parts.append(expected_actual)
    if not assertion_line and not expected_actual:
        parts.append(test_output[:500])
    parts.append("```")

    parts += [
        "",
        "## Check These",
        "- Algorithm logic (reduction order, boundary conditions, off-by-one)",
        "- Numerical precision (float accumulation order, FMA vs separate mul+add)",
        "- Memory layout (row-major vs column-major, stride calculation, padding)",
        "- Index mapping (threadIdx/blockIdx to array indices)",
        "",
        "## Your Previous Code",
        "```cuda",
        _truncate(code, 2000),
        "```",
        "",
        "Fix the logic error. Output ONLY the corrected content of `solution.cu`.",
    ]

    return "\n".join(parts)


def _feedback_perf(test_output: str, code: str, classification: dict | None = None) -> str:
    """Correctness OK but below the configured performance gate."""
    gate = (classification or {}).get("performance_gate") or {}
    threshold = gate.get("min_speedup")
    speedups = [float(m) for m in re.findall(r"Speedup:\s*([\d.]+)x", test_output or "")]
    if speedups:
        mean_sp = sum(speedups) / len(speedups)
        per_size = ", ".join(f"{s:.3f}x" for s in speedups)
        perf_summary = f"Mean speedup: {mean_sp:.3f}x  (per-size: {per_size})"
    elif gate.get("speedup") is not None:
        perf_summary = f"Speedup: {gate['speedup']:.3f}x"
    elif threshold is not None:
        perf_summary = f"Speedup below {float(threshold):.3f}x reference."
    else:
        perf_summary = "Speedup below the configured performance gate."

    requirement = (
        f"It must reach >= {float(threshold):.3f}x reference speedup to pass."
        if threshold is not None
        else "It must reach the configured reference speedup threshold to pass."
    )

    return "\n".join([
        f"Your code is correct but too slow. {requirement}",
        "",
        "## Performance Result",
        "```",
        perf_summary,
        "```",
        "",
        "## Likely Bottlenecks (focus your rewrite here)",
        "- Global memory: ensure coalesced loads/stores; use `float4`/`__ldg` where applicable",
        "- Shared memory: tile to reuse data; watch for bank conflicts (stride 32)",
        "- Occupancy: tune block size (typically 128–256), reduce per-thread register/shared usage",
        "- Compute throughput: prefer FMA; replace divisions with reciprocals; vectorize inner loop",
        "- Tensor cores: for matmul/conv, switch to wmma/mma fragments where shapes allow",
        "- Launch config: minimize wave imbalance (grid divisible by SM count × waves)",
        "",
        "## Your Previous Code",
        "```cuda",
        _truncate(code, 2000),
        "```",
        "",
        "Rewrite for higher throughput. Output ONLY the optimized content of `solution.cu`.",
    ])


def _feedback_environment(build_output: str, test_output: str, code: str) -> str:
    """Environment issues — unlikely to be fixable by the model."""
    combined = build_output + "\n" + test_output
    return "\n".join([
        "Your code encountered an environment error (GPU architecture mismatch or missing dependency).",
        "This may not be fixable by changing your code.",
        "",
        "## Error",
        "```",
        combined[:500],
        "```",
        "",
        "## Your Previous Code",
        "```cuda",
        _truncate(code, 1500),
        "```",
        "",
        "Try a different approach if possible. Output ONLY the corrected content of `solution.cu`.",
    ])


def _feedback_generic(build_output: str, test_output: str | None, code: str) -> str:
    """Fallback for unclassified errors."""
    error_log = build_output
    if test_output:
        error_log += "\n" + test_output
    if len(error_log) > 3000:
        error_log = error_log[:1500] + "\n...[truncated]...\n" + error_log[-1500:]

    return "\n".join([
        "Your previous solution did not pass. Here is the feedback:",
        "",
        "## Build / Test Output",
        "```",
        error_log,
        "```",
        "",
        "## Your Previous Code",
        "```cuda",
        _truncate(code, 2000),
        "```",
        "",
        "Please fix the code. Output ONLY the corrected content of `solution.cu`.",
    ])


def _get_code_window(code: str, line_num: int, window: int = 5) -> str:
    """Extract lines around the error."""
    lines = code.splitlines()
    start = max(0, line_num - window - 1)
    end = min(len(lines), line_num + window)
    result = []
    for i in range(start, end):
        marker = ">>> " if i == line_num - 1 else "    "
        result.append(f"{marker}{i + 1:4d} | {lines[i]}")
    return "\n".join(result)


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    half = max_len // 2
    return text[:half] + "\n... [truncated] ...\n" + text[-half:]
