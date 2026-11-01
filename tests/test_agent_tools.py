#!/usr/bin/env python3
"""Unit tests for agent_tools — schema integrity, executors, quota enforcement.

Subprocess-touching tools (sanitizer, build_only/test_only) are tested
through their hooks: build_only/test_only delegate to env.rebuild_fn /
env.rerun_test_fn which the loop sets, so we exercise the dispatch path
with stub functions and rely on test_diagnostics.py for the actual
subprocess wrappers.
"""

import json
from pathlib import Path

import pytest

from harness import agent_tools
from harness.agent_tools import (
    TOOL_SCHEMAS,
    ToolEnv,
    ToolResult,
    call_tool,
    make_env,
    schemas_for_config,
)


# ---------------------------------------------------------------------------
# Schema integrity
# ---------------------------------------------------------------------------


def test_schemas_have_required_shape():
    seen = set()
    for s in TOOL_SCHEMAS:
        assert s["type"] == "function"
        fn = s["function"]
        assert "name" in fn
        assert "description" in fn
        assert "parameters" in fn
        assert fn["name"] not in seen, "duplicate tool name"
        seen.add(fn["name"])
    expected = {
        "read_code", "read_test_file", "read_prompt",
        "run_static_analysis", "run_compute_sanitizer",
        "build_only", "test_only", "locate_bug", "apply_edit", "write_solution",
    }
    assert seen == expected


def test_schemas_for_config_filters():
    sub = schemas_for_config(["read_code", "write_solution"])
    names = {s["function"]["name"] for s in sub}
    assert names == {"read_code", "write_solution"}


def test_schemas_for_config_returns_all_when_unset():
    sub = schemas_for_config(None)
    assert len(sub) == len(TOOL_SCHEMAS)


# ---------------------------------------------------------------------------
# Fixtures — minimal ToolEnv for executor tests
# ---------------------------------------------------------------------------


def _make_env(tmp_path: Path) -> ToolEnv:
    workdir = tmp_path / "ws" / "CUDA_42"
    workdir.mkdir(parents=True)
    dataset = tmp_path / "dataset"
    (dataset / "prompts").mkdir(parents=True)
    (dataset / "testbench" / "CUDA_42" / "test").mkdir(parents=True)
    (dataset / "testbench" / "CUDA_42" / "include").mkdir(parents=True)
    (dataset / "prompts" / "CUDA_42.txt").write_text("Write a kernel that does X.\n")
    (dataset / "testbench" / "CUDA_42" / "test" / "test_main.cu").write_text(
        "// test harness\nint main() { return 0; }\n"
    )
    (dataset / "testbench" / "CUDA_42" / "include" / "helpers.h").write_text(
        "#define TOLERANCE 1e-5\n"
    )
    return make_env(
        task_stem="CUDA_42",
        task_workdir=workdir,
        dataset_dir=dataset,
        input_meta={
            "task_id": "CUDA/42",
            "build_command": "nvcc solution.cu -o test.out",
            "test_command": "./test.out",
            "solution_file": "solution.cu",
        },
        build_timeout=60,
        test_timeout=30,
    )


# ---------------------------------------------------------------------------
# read_code — reads from in-memory candidate
# ---------------------------------------------------------------------------


def test_read_code_returns_lines(tmp_path):
    env = _make_env(tmp_path)
    env.current_code = "\n".join(f"line {i}" for i in range(1, 21))
    r = call_tool("read_code", {"start": 5, "end": 8}, env, iteration=1)
    assert r.ok
    assert "5 | line 5" in r.content
    assert "8 | line 8" in r.content
    assert "9 | line 9" not in r.content


def test_read_code_empty_candidate_errors(tmp_path):
    env = _make_env(tmp_path)
    r = call_tool("read_code", {"start": 1, "end": 5}, env, iteration=1)
    assert not r.ok
    assert "no candidate" in (r.error or "").lower()


def test_read_code_caps_huge_range(tmp_path):
    env = _make_env(tmp_path)
    env.current_code = "\n".join(f"line {i}" for i in range(1, 3000))
    r = call_tool("read_code", {"start": 1, "end": 2500}, env, iteration=1)
    assert r.ok
    from harness.agent_tools import MAX_READ_LINES
    assert r.meta["range"][1] - r.meta["range"][0] + 1 <= MAX_READ_LINES


def test_read_code_out_of_range_reports_error(tmp_path):
    env = _make_env(tmp_path)
    env.current_code = "line 1\nline 2\n"
    r = call_tool("read_code", {"start": 500, "end": 510}, env, iteration=1)
    assert not r.ok
    assert "out of range" in (r.error or "").lower()


# ---------------------------------------------------------------------------
# read_test_file — path confinement
# ---------------------------------------------------------------------------


def test_read_test_file_in_bounds(tmp_path):
    env = _make_env(tmp_path)
    r = call_tool("read_test_file", {"path": "test/test_main.cu"}, env, iteration=1)
    assert r.ok
    assert "int main" in r.content


def test_read_test_file_rejects_path_traversal(tmp_path):
    env = _make_env(tmp_path)
    # Create a file outside the testbench
    secret = tmp_path / "secret.txt"
    secret.write_text("secret data")
    r = call_tool(
        "read_test_file", {"path": "../../secret.txt"}, env, iteration=1
    )
    assert not r.ok
    assert "escapes" in (r.error or "").lower()


def test_read_test_file_rejects_disallowed_suffix(tmp_path):
    env = _make_env(tmp_path)
    (env.testbench_dir / "bin.exe").write_bytes(b"\x00\x01\x02")
    r = call_tool("read_test_file", {"path": "bin.exe"}, env, iteration=1)
    assert not r.ok
    assert "refusing" in (r.error or "").lower()


def test_read_test_file_start_only_is_line_capped(tmp_path):
    env = _make_env(tmp_path)
    big = env.testbench_dir / "include" / "big.h"
    big.write_text("\n".join(f"line {i}" for i in range(1, 1000)))

    r = call_tool("read_test_file", {"path": "include/big.h", "start": 10}, env, iteration=1)

    assert r.ok
    from harness.agent_tools import MAX_READ_LINES
    assert r.meta["range"][1] - r.meta["range"][0] + 1 <= MAX_READ_LINES


def test_read_test_file_allows_dataset_external_symlink(tmp_path):
    env = _make_env(tmp_path)
    external = env.dataset_dir / "data" / "_external" / "cutlass"
    external.mkdir(parents=True)
    header = external / "external_header.h"
    header.write_text("// external header\n")
    refs = env.testbench_dir / "reference_sources"
    refs.mkdir()
    link = refs / "external_header.h"
    try:
        link.symlink_to(header)
    except OSError as exc:  # pragma: no cover - platform/filesystem defensive path
        pytest.skip(f"symlink unavailable: {exc}")

    r = call_tool("read_test_file", {"path": "reference_sources/external_header.h"}, env, iteration=1)

    assert r.ok
    assert "external header" in r.content


def test_read_prompt_returns_text(tmp_path):
    env = _make_env(tmp_path)
    r = call_tool("read_prompt", {}, env, iteration=1)
    assert r.ok
    assert "Write a kernel" in r.content


def test_run_static_analysis_flags_memory_write(tmp_path):
    env = _make_env(tmp_path)
    env.current_code = (
        "extern \"C\" __global__ void k(float* out) {\n"
        "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "  out[i] = 1.0f;\n"
        "}\n"
    )
    r = call_tool("run_static_analysis", {}, env, iteration=1)
    assert r.ok
    assert r.meta["flags"]
    assert any(f["flag"] == "memory_write" and f["line"] == 3 for f in r.meta["flags"])


# ---------------------------------------------------------------------------
# locate_bug — localisation declaration
# ---------------------------------------------------------------------------


def test_locate_bug_records_claim_and_echoes_line(tmp_path):
    env = _make_env(tmp_path)
    env.current_code = "\n".join(f"line {i}" for i in range(1, 21))
    r = call_tool(
        "locate_bug",
        {"file": "solution.cu", "line": 7,
         "bug_description": "off-by-one in stride",
         "evidence": "hand-traced thread 0 of block 1"},
        env, iteration=2,
    )
    assert r.ok
    assert r.meta["located"] is True
    assert r.meta["line"] == 7
    assert r.meta["file"] == "solution.cu"
    # echoes the cited line from the candidate, marked with >>
    assert ">>    7 | line 7" in r.content


def test_locate_bug_requires_all_fields(tmp_path):
    env = _make_env(tmp_path)
    r = call_tool("locate_bug", {"file": "solution.cu", "line": 3}, env, iteration=2)
    assert not r.ok
    assert "requires" in (r.error or "").lower()


def test_locate_bug_non_integer_line(tmp_path):
    env = _make_env(tmp_path)
    r = call_tool(
        "locate_bug",
        {"file": "solution.cu", "line": "seven",
         "bug_description": "x", "evidence": "y"},
        env, iteration=2,
    )
    assert not r.ok
    assert "integer" in (r.error or "").lower()


def test_locate_bug_line_out_of_range(tmp_path):
    env = _make_env(tmp_path)
    env.current_code = "a\nb\nc\n"
    r = call_tool(
        "locate_bug",
        {"file": "solution.cu", "line": 999,
         "bug_description": "x", "evidence": "y"},
        env, iteration=2,
    )
    assert r.ok  # still recorded — the model's claim, even if it missed
    assert "out of range" in r.content


# ---------------------------------------------------------------------------
# apply_edit — surgical patch on env.current_code
# ---------------------------------------------------------------------------


def test_apply_edit_unique_match_replaces(tmp_path):
    env = _make_env(tmp_path)
    env.current_code = "int main() {\n  int x = 0;\n  return x;\n}\n"
    r = call_tool(
        "apply_edit",
        {"old_string": "int x = 0;", "new_string": "int x = 42;"},
        env, iteration=2,
    )
    assert r.ok
    assert r.meta == {
        "edited": True,
        "replace_all": False,
        "occurrences": 1,
        "match_strategy": "exact",
    }
    assert "int x = 42;" in env.current_code
    assert "int x = 0;" not in env.current_code
    assert "Edited region" in r.content


def test_apply_edit_whitespace_tolerant_line_match(tmp_path):
    env = _make_env(tmp_path)
    env.current_code = "def f():\n    x = foo(1,\n          2)\n    return x\n"
    r = call_tool(
        "apply_edit",
        {
            "old_string": "    x = foo(1,\n           2)",
            "new_string": "    x = foo(3,\n          4)",
        },
        env,
        iteration=2,
    )
    assert r.ok
    assert r.meta["match_strategy"] == "whitespace_tolerant_lines"
    assert "foo(3," in env.current_code
    assert "foo(1," not in env.current_code


def test_apply_edit_line_range_replaces(tmp_path):
    env = _make_env(tmp_path)
    env.current_code = "a\nb\nc\nd\n"
    r = call_tool(
        "apply_edit",
        {"start_line": 2, "end_line": 3, "new_string": "B\nC"},
        env,
        iteration=2,
    )
    assert r.ok
    assert r.meta["match_strategy"] == "line_range"
    assert env.current_code == "a\nB\nC\nd\n"


def test_apply_edit_old_string_not_found(tmp_path):
    env = _make_env(tmp_path)
    env.current_code = "ok\n"
    r = call_tool("apply_edit",
                  {"old_string": "missing", "new_string": "x"},
                  env, iteration=2)
    assert not r.ok
    assert "not found" in (r.error or "").lower()


def test_apply_edit_non_unique_without_replace_all_errors(tmp_path):
    env = _make_env(tmp_path)
    env.current_code = "foo\nfoo\nfoo\n"
    r = call_tool("apply_edit",
                  {"old_string": "foo", "new_string": "bar"},
                  env, iteration=2)
    assert not r.ok
    assert "3 times" in (r.error or "") or "unique" in (r.error or "").lower()
    assert env.current_code == "foo\nfoo\nfoo\n"  # unchanged


def test_apply_edit_replace_all(tmp_path):
    env = _make_env(tmp_path)
    env.current_code = "foo\nfoo\nfoo\n"
    r = call_tool("apply_edit",
                  {"old_string": "foo", "new_string": "bar", "replace_all": True},
                  env, iteration=2)
    assert r.ok
    assert env.current_code == "bar\nbar\nbar\n"
    assert r.meta["occurrences"] == 3
    assert r.meta["replace_all"] is True


def test_apply_edit_empty_candidate_errors(tmp_path):
    env = _make_env(tmp_path)
    env.current_code = ""
    r = call_tool("apply_edit",
                  {"old_string": "x", "new_string": "y"},
                  env, iteration=2)
    assert not r.ok
    assert "empty" in (r.error or "").lower()


def test_apply_edit_requires_non_empty_old_string(tmp_path):
    env = _make_env(tmp_path)
    env.current_code = "code\n"
    r = call_tool("apply_edit",
                  {"old_string": "", "new_string": "x"},
                  env, iteration=2)
    assert not r.ok


# ---------------------------------------------------------------------------
# write_solution — sentinel meta.submit, two modes
# ---------------------------------------------------------------------------


def test_write_solution_patch_submit_commits_current_code(tmp_path):
    """write_solution is patch-submit only: it commits env.current_code
    (which holds broken_start + any staged apply_edit changes)."""
    env = _make_env(tmp_path)
    env.current_code = "patched candidate"
    r = call_tool("write_solution", {"rationale": "ship it"}, env, iteration=2)
    assert r.ok
    assert r.meta["submit"] is True
    assert r.meta["rationale"] == "ship it"


def test_write_solution_accepts_full_file_fallback_code(tmp_path):
    """`code` is a full-file fallback for brittle apply_edit matching."""
    env = _make_env(tmp_path)
    env.current_code = "existing"
    r = call_tool(
        "write_solution",
        {"code": "I rewrote the file", "rationale": "fallback"},
        env, iteration=2,
    )
    assert r.ok
    assert r.meta["submit"] is True
    assert r.meta["full_replacement"] is True
    assert r.meta["rationale"] == "fallback"
    assert env.current_code == "I rewrote the file"


def test_write_solution_rejects_when_no_candidate(tmp_path):
    """No broken_start seeded → can't submit. This is a config error in
    practice (debug mode always seeds), but we guard against it."""
    env = _make_env(tmp_path)
    env.current_code = ""
    r = call_tool("write_solution", {}, env, iteration=1)
    assert not r.ok
    assert "nothing to submit" in (r.error or "").lower()


# ---------------------------------------------------------------------------
# build_only / test_only — delegate to env hooks
# ---------------------------------------------------------------------------


def test_build_only_uses_hook(tmp_path):
    env = _make_env(tmp_path)
    env.rebuild_fn = lambda: (True, "0 errors, 0 warnings")
    r = call_tool("build_only", {}, env, iteration=1)
    assert r.ok
    assert "BUILD OK" in r.content


def test_build_only_without_hook(tmp_path):
    env = _make_env(tmp_path)
    r = call_tool("build_only", {}, env, iteration=1)
    assert not r.ok


def test_test_only_uses_hook(tmp_path):
    env = _make_env(tmp_path)
    env.rerun_test_fn = lambda: (0, "all checks passed")
    r = call_tool("test_only", {}, env, iteration=1)
    assert r.ok
    assert "RETURNCODE: 0" in r.content


# ---------------------------------------------------------------------------
# Quota enforcement
# ---------------------------------------------------------------------------


def test_quota_blocks_after_limit(tmp_path):
    env = _make_env(tmp_path)
    env.current_code = "x\n"
    env.quotas["read_code"] = 2
    assert call_tool("read_code", {"start": 1, "end": 1}, env, 1).ok
    assert call_tool("read_code", {"start": 1, "end": 1}, env, 1).ok
    r = call_tool("read_code", {"start": 1, "end": 1}, env, 1)
    assert not r.ok
    assert "quota" in (r.error or "").lower()


def test_quota_none_is_unlimited(tmp_path):
    env = _make_env(tmp_path)
    env.current_code = "x\n"
    env.quotas["read_code"] = None
    for _ in range(50):
        assert call_tool("read_code", {"start": 1, "end": 1}, env, 1).ok


# ---------------------------------------------------------------------------
# Trace recording
# ---------------------------------------------------------------------------


def test_trace_records_every_call(tmp_path):
    env = _make_env(tmp_path)
    env.current_code = "x\n"
    call_tool("read_code", {"start": 1, "end": 1}, env, iteration=3)
    call_tool("read_prompt", {}, env, iteration=3)
    assert len(env.trace) == 2
    assert env.trace[0].tool == "read_code"
    assert env.trace[1].tool == "read_prompt"
    assert env.trace[0].iteration == 3
    assert env.trace[0].elapsed_ms >= 0


def test_invalid_json_args(tmp_path):
    env = _make_env(tmp_path)
    r = call_tool("read_code", "{not valid json", env, 1)
    assert not r.ok
    assert "invalid JSON" in (r.error or "")


def test_unknown_tool(tmp_path):
    env = _make_env(tmp_path)
    r = call_tool("rm_rf", {}, env, 1)
    assert not r.ok
    assert "unknown" in (r.error or "").lower()
