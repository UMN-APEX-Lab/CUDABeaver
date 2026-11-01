"""
Runtime-diagnostics helpers — a wrapper around compute-sanitizer.

Used by the agent tool layer (read-only investigation). Every entry point
returns a normalised `DiagResult` and never raises into the caller —
failures degrade to an "unavailable" reason string.

The subprocess is launched through `run_pg` so SIGKILL hits the whole
process group on timeout (compute-sanitizer forks helpers).
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .subprocess_util import run_pg

# Hard cap on what we expose back to the LLM. compute-sanitizer emits a
# huge transcript; the model only needs the first failing site to act on.
SANITIZER_OUTPUT_CAP = 2000
SANITIZER_TIMEOUT_DEFAULT = 180

SANITIZER_TOOLS = ("memcheck", "racecheck", "initcheck", "synccheck")
TARGET_PROCESSES = ("application-only", "all")
PRINT_LEVELS = ("info", "warn", "error", "fatal")
BACKTRACE_MODES = ("no", "host", "device", "yes")
LEAK_CHECK_MODES = ("full", "no")
REPORT_API_ERRORS = ("all", "explicit", "no")
STREAM_RACE_MODES = ("all", "use-before-alloc", "use-after-free", "no")
RACECHECK_DETECT_LEVELS = ("info", "warn", "error")
RACECHECK_REPORTS = ("all", "hazard", "analysis")


def _bool_choice(value: bool) -> str:
    return "yes" if bool(value) else "no"


def _enum_value(name: str, value: str | None, allowed: tuple[str, ...]) -> str | None:
    if value is None:
        return None
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
    return value


def _int_value(name: str, value, *, minimum: int | None = None) -> int | None:
    if value is None:
        return None
    ivalue = int(value)
    if minimum is not None and ivalue < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {ivalue}")
    return ivalue


def _add_option(args: list[str], name: str, value=None) -> None:
    if value is None:
        args.append(name)
    else:
        args.append(f"{name}={value}")


def _build_sanitizer_options(
    *,
    tool: str,
    wrapped_command: bool,
    options: dict | None,
) -> tuple[list[str], dict]:
    """Validate user-facing options and render compute-sanitizer flags.

    The agent controls only sanitizer flags. The executable remains the
    benchmark's own test_command, so we can expose a broad option surface
    without giving the model arbitrary shell access.
    """
    opts = dict(options or {})
    rendered: list[str] = [f"--tool={tool}"]
    normalized: dict = {"tool": tool}

    def pop_bool(key: str, default: bool | None = None) -> bool | None:
        if key not in opts:
            return default
        return bool(opts.pop(key))

    def pop_int(key: str, default=None, minimum: int | None = None):
        raw = opts.pop(key, default)
        return _int_value(key, raw, minimum=minimum)

    def pop_enum(key: str, allowed: tuple[str, ...], default=None):
        raw = opts.pop(key, default)
        return _enum_value(key, raw, allowed)

    target_processes = pop_enum("target_processes", TARGET_PROCESSES, "all")
    require_cuda_init = pop_bool("require_cuda_init", not wrapped_command)
    print_limit = pop_int("print_limit", 10, minimum=0)
    error_exitcode = pop_int("error_exitcode", 99, minimum=1)
    launch_timeout = pop_int("launch_timeout", None, minimum=0)
    print_level = pop_enum("print_level", PRINT_LEVELS, None)
    show_backtrace = pop_enum("show_backtrace", BACKTRACE_MODES, None)
    check_exit_code = pop_bool("check_exit_code", None)
    force_blocking_launches = pop_bool("force_blocking_launches", None)
    force_sync_limit = pop_int("force_synchronization_limit", None, minimum=0)
    launch_count = pop_int("launch_count", None, minimum=0)
    launch_skip = pop_int("launch_skip", None, minimum=0)
    num_callers_host = pop_int("num_callers_host", None, minimum=0)
    num_callers_device = pop_int("num_callers_device", None, minimum=0)
    max_connections = pop_int("max_connections", None, minimum=1)
    nvtx = pop_bool("nvtx", None)
    quiet = pop_bool("quiet", None)
    kill = pop_bool("kill", None)
    backtrace_short = pop_bool("backtrace_short", None)

    _add_option(rendered, "--target-processes", target_processes)
    _add_option(rendered, "--require-cuda-init", _bool_choice(require_cuda_init))
    _add_option(rendered, "--print-limit", print_limit)
    _add_option(rendered, "--error-exitcode", error_exitcode)
    normalized.update(
        target_processes=target_processes,
        require_cuda_init=require_cuda_init,
        print_limit=print_limit,
        error_exitcode=error_exitcode,
    )

    for key, value, flag in (
        ("target_processes_filter", opts.pop("target_processes_filter", None), "--target-processes-filter"),
        ("kernel_name", opts.pop("kernel_name", None), "--kernel-name"),
        ("kernel_name_exclude", opts.pop("kernel_name_exclude", None), "--kernel-name-exclude"),
        ("launch_timeout", launch_timeout, "--launch-timeout"),
        ("print_level", print_level, "--print-level"),
        ("show_backtrace", show_backtrace, "--show-backtrace"),
        ("force_synchronization_limit", force_sync_limit, "--force-synchronization-limit"),
        ("launch_count", launch_count, "--launch-count"),
        ("launch_skip", launch_skip, "--launch-skip"),
        ("num_callers_host", num_callers_host, "--num-callers-host"),
        ("num_callers_device", num_callers_device, "--num-callers-device"),
        ("max_connections", max_connections, "--max-connections"),
    ):
        if value is not None:
            _add_option(rendered, flag, value)
            normalized[key] = value

    for key, value, flag in (
        ("check_exit_code", check_exit_code, "--check-exit-code"),
        ("nvtx", nvtx, "--nvtx"),
        ("backtrace_short", backtrace_short, "--backtrace-short"),
    ):
        if value is not None:
            _add_option(rendered, flag, _bool_choice(value))
            normalized[key] = value

    if force_blocking_launches:
        _add_option(rendered, "--force-blocking-launches")
        normalized["force_blocking_launches"] = True
    if quiet:
        _add_option(rendered, "--quiet")
        normalized["quiet"] = True
    if kill:
        _add_option(rendered, "--kill")
        normalized["kill"] = True

    if tool == "memcheck":
        for key, value, flag in (
            ("leak_check", pop_enum("leak_check", LEAK_CHECK_MODES, "no"), "--leak-check"),
            ("padding", pop_int("padding", None, minimum=0), "--padding"),
            ("report_api_errors", pop_enum("report_api_errors", REPORT_API_ERRORS, None), "--report-api-errors"),
            ("track_stream_ordered_races", pop_enum("track_stream_ordered_races", STREAM_RACE_MODES, None), "--track-stream-ordered-races"),
        ):
            if value is not None:
                _add_option(rendered, flag, value)
                normalized[key] = value
        for key, flag in (
            ("check_cache_control", "--check-cache-control"),
            ("detect_missing_module_unload", "--detect-missing-module-unload"),
        ):
            value = pop_bool(key, None)
            if value:
                _add_option(rendered, flag)
                normalized[key] = True
    if tool == "racecheck":
        for key, value, flag in (
            ("racecheck_detect_level", pop_enum("racecheck_detect_level", RACECHECK_DETECT_LEVELS, None), "--racecheck-detect-level"),
            ("racecheck_report", pop_enum("racecheck_report", RACECHECK_REPORTS, None), "--racecheck-report"),
            ("racecheck_deadlock_timeout", pop_int("racecheck_deadlock_timeout", None, minimum=0), "--racecheck-deadlock-timeout"),
        ):
            if value is not None:
                _add_option(rendered, flag, value)
                normalized[key] = value
        for key, flag in (
            ("racecheck_trace_sync", "--racecheck-trace-sync"),
            ("racecheck_memcpy_async", "--racecheck-memcpy-async"),
            ("racecheck_continue_on_deadlock", "--racecheck-continue-on-deadlock"),
        ):
            value = pop_bool(key, None)
            if value is not None:
                _add_option(rendered, flag, _bool_choice(value))
                normalized[key] = value

    if tool == "initcheck":
        for key, flag in (
            ("check_api_memory_access", "--check-api-memory-access"),
            ("track_unused_memory", "--track-unused-memory"),
            ("check_optix", "--check-optix"),
        ):
            value = pop_bool(key, None)
            if value is not None:
                if flag in {"--track-unused-memory", "--check-optix"}:
                    if value:
                        _add_option(rendered, flag)
                else:
                    _add_option(rendered, flag, _bool_choice(value))
                normalized[key] = value
        unused_threshold = pop_int("unused_memory_threshold", None, minimum=0)
        if unused_threshold is not None:
            _add_option(rendered, "--unused-memory-threshold", unused_threshold)
            normalized["unused_memory_threshold"] = unused_threshold

    if tool == "synccheck":
        missing_barrier = pop_bool("missing_barrier_init_is_fatal", None)
        if missing_barrier is not None:
            _add_option(rendered, "--missing-barrier-init-is-fatal", _bool_choice(missing_barrier))
            normalized["missing_barrier_init_is_fatal"] = missing_barrier

    if opts:
        unknown = ", ".join(sorted(opts))
        raise ValueError(f"unsupported compute-sanitizer option(s): {unknown}")

    return rendered, normalized


@dataclass
class DiagResult:
    """Normalised diagnostic output."""

    ok: bool
    tool: str
    summary: str
    truncated: bool = False
    raw_excerpt: str = ""
    error: str | None = None
    meta: dict = field(default_factory=dict)

    def to_prompt_block(self) -> str:
        """Render for inclusion in a prompt or tool result."""
        header = f"## {self.tool}"
        if not self.ok:
            return f"{header}\n[unavailable: {self.error or 'unknown error'}]"
        body = self.summary
        if self.truncated:
            body += "\n... [output truncated] ..."
        return f"{header}\n{body}"


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------


def sanitizer_available() -> bool:
    return shutil.which("compute-sanitizer") is not None


# ---------------------------------------------------------------------------
# compute-sanitizer
# ---------------------------------------------------------------------------


def run_compute_sanitizer(
    test_command: str,
    workdir: Path,
    *,
    tool: str = "memcheck",
    timeout: int = SANITIZER_TIMEOUT_DEFAULT,
    env: dict | None = None,
    options: dict | None = None,
) -> DiagResult:
    """Run `compute-sanitizer --tool=<tool> <test_command>` and summarise.

    `test_command` is the shell string used by the test runner (e.g.
    "./test.out"). Wrapped commands are supported: the default options use
    `--target-processes=all`, and wrapped commands default
    `--require-cuda-init=no` so a shell/make/python launcher can be the root
    process while CUDA work happens in a child or imported extension.
    """
    if tool not in SANITIZER_TOOLS:
        return DiagResult(
            ok=False, tool=f"compute-sanitizer/{tool}", summary="",
            error=f"unsupported tool {tool!r}; expected one of {SANITIZER_TOOLS}",
        )
    if not sanitizer_available():
        return DiagResult(
            ok=False, tool=f"compute-sanitizer/{tool}", summary="",
            error="compute-sanitizer not found on PATH",
        )
    if not str(test_command or "").strip():
        return DiagResult(
            ok=False, tool=f"compute-sanitizer/{tool}", summary="",
            error="empty test command; nothing to run under compute-sanitizer",
        )

    wrapped_command = _looks_like_wrapped_command(test_command)
    try:
        option_args, normalized_options = _build_sanitizer_options(
            tool=tool,
            wrapped_command=wrapped_command,
            options=options,
        )
    except Exception as e:
        return DiagResult(
            ok=False, tool=f"compute-sanitizer/{tool}", summary="",
            error=f"invalid sanitizer option: {e}",
        )

    prefix = " ".join(["compute-sanitizer"] + [shlex.quote(arg) for arg in option_args])
    cmd = f"{prefix} {test_command}"
    try:
        proc = run_pg(cmd, shell=True, cwd=str(workdir), timeout=timeout, env=env)
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        summary, truncated = _summarise_sanitizer(combined)
        # error_exitcode=99 → sanitizer found a violation. rc==0 → clean run.
        # Anything else → wrapped command failed for unrelated reasons.
        meta = {
            "returncode": proc.returncode,
            "command": cmd,
            "wrapped_command": wrapped_command,
            "options": normalized_options,
        }
        return DiagResult(
            ok=True,
            tool=f"compute-sanitizer/{tool}",
            summary=summary,
            truncated=truncated,
            raw_excerpt=combined[:200],
            meta=meta,
        )
    except subprocess.TimeoutExpired:
        return DiagResult(
            ok=False, tool=f"compute-sanitizer/{tool}", summary="",
            error=f"sanitizer timed out after {timeout}s",
        )
    except Exception as e:
        return DiagResult(
            ok=False, tool=f"compute-sanitizer/{tool}", summary="",
            error=f"{type(e).__name__}: {e}",
        )


def _summarise_sanitizer(output: str) -> tuple[str, bool]:
    """Pick the first reported error block + tail summary.

    compute-sanitizer's transcript looks like:

        ========= COMPUTE-SANITIZER
        ========= <ErrorKind> of size N bytes        ← block start
        =========     at 0x70 in kernel(...)
        =========     by thread (33,0,0) in block (0,0,0)
        =========     Host Frame:... at solution.cu:42
        =========                                    ← blank divider
        ========= <ErrorKind> ...                    ← second block (skip)
        =========
        ========= ERROR SUMMARY: N errors

    We surface only the first block (subsequent ones are usually the same
    bug repeated per thread) plus the ERROR SUMMARY tail.
    """
    if not output.strip():
        return "(empty sanitizer output)", False

    summary_tail: str | None = None
    # The "block start" line is any `========= <non-space>` that is NOT the
    # COMPUTE-SANITIZER header or the ERROR SUMMARY footer.
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in output.splitlines():
        if line.startswith("========= ERROR SUMMARY"):
            summary_tail = line
            continue
        if line == "=========" or line == "========= ":
            # Divider: close current block (if any).
            if current:
                blocks.append(current)
                current = []
            continue
        if line.startswith("========= COMPUTE-SANITIZER"):
            continue
        if line.startswith("========= "):
            # Could be a new block header OR a continuation line of the
            # current block (indented frames also start with =========).
            # Heuristic: a header line has non-space immediately after the
            # `========= ` prefix; continuation lines start with whitespace.
            after = line[len("========= "):]
            is_header = bool(after) and not after[0].isspace()
            if is_header and current:
                # Start of next block — close the previous one.
                blocks.append(current)
                current = []
            current.append(line)
    if current:
        blocks.append(current)

    if not blocks:
        body = summary_tail or "no errors detected"
        return body[:SANITIZER_OUTPUT_CAP], False

    first = blocks[0][:30]  # cap to ~30 frames
    body = "\n".join(first)
    if summary_tail:
        body += "\n" + summary_tail
    truncated = len(body) > SANITIZER_OUTPUT_CAP
    if truncated:
        half = SANITIZER_OUTPUT_CAP // 2
        body = body[:half] + "\n... [middle truncated] ...\n" + body[-half:]
    return body, truncated


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _looks_like_wrapped_command(cmd: str) -> bool:
    """Detect commands that aren't a direct CUDA binary.

    compute-sanitizer wrapping a python interpreter or make recipe usually
    produces useless output (host-side fault, not the kernel). The agent
    should rebuild against a direct test binary for these tasks.
    """
    head = cmd.strip().split()
    if not head:
        return True
    first = head[0]
    # Direct binary like ./test.out or /abs/path/test.out
    if first.startswith("./") or first.startswith("/"):
        return False
    if first.endswith(".out"):
        return False
    # Bare commands likely to be wrappers
    wrapper_tokens = {"python", "python3", "make", "bash", "sh", "env"}
    return first in wrapper_tokens or any(t in cmd for t in (" python ", " make ", " bash "))
