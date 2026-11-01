#!/usr/bin/env python3
"""Unit tests for role-scoped blackboard views and token accumulation.

Pure logic — no LLM, GPU, or nvcc required.
"""

import json

from harness.cudeubgger.common import (
    ErrorBlackboard,
    _LoopState,
    _accumulate_token_usage,
    _json_role_action_request,
    _normalize_structured_report,
    _scoped_views_enabled,
    _structured_reports_enabled,
    _token_usage_summary,
)


def _populated_board() -> ErrorBlackboard:
    """A board with content from every role, mirroring a real round."""
    board = ErrorBlackboard(task_stem="cuda_demo")
    board.error_type = "logic_error"
    # analyzer-sourced suspect + verifier-confirmed suspect
    board.add_suspected_line({
        "source": "analyzer",
        "file": "solution.cu",
        "line": 12,
        "bug_description": "missing __syncthreads before reduction",
        "evidence": "static",
    })
    board.add_suspected_line({
        "source": "verifier",
        "file": "solution.cu",
        "line": 12,
        "confidence": "high",
        "root_cause": "race on shared memory: no barrier",
        "evidence": ["racecheck hazard"],
    })
    board.static_flags = [{
        "source": "analyzer",
        "file": "solution.cu",
        "line": 12,
        "flag": "missing_sync",
        "severity": "high",
        "reason": "shared write before barrier",
        "code": "sdata[tid] += ...",
    }]
    board.runtime_trace = {
        "initial_observation": {"build_ok": True, "test_returncode": 1},
        "tool_results": [
            {
                "source": "executor",
                "tool": "run_compute_sanitizer",
                "ok": True,
                "error": None,
                "meta": {"tool": "racecheck"},
                "content_excerpt": "X" * 2000,
            },
        ],
    }
    board.analyzer_report = "ANALYZER FREE TEXT " * 50
    board.executor_report = "EXECUTOR FREE TEXT " * 50
    board.verifier_report = "VERIFIER FREE TEXT " * 50
    board.fix_suggestions = ["add __syncthreads() before the reduction loop"]
    board.next_fix = "add __syncthreads() before the reduction loop"
    return board


def test_coder_view_is_strict():
    board = _populated_board()
    view = json.loads(board.view_for("coder"))

    # Included: distilled verifier output the coder acts on.
    assert view["error_type"] == "logic_error"
    assert view["next_fix"]                     # single fix direction (gap 4)
    assert view["suspected_lines"]
    # Only the verifier-confirmed suspect survives.
    assert all(s.get("source") == "verifier" for s in view["suspected_lines"])
    assert "latest_runtime" in view

    # Excluded: free-text reports and raw evidence the coder does not need.
    assert "analyzer_report" not in view
    assert "executor_report" not in view
    assert "verifier_report" not in view
    assert "static_flags" not in view
    assert "runtime_trace" not in view
    assert "fix_suggestions" not in view        # replaced by next_fix


def test_coder_view_smaller_than_full_dump():
    board = _populated_board()
    assert len(board.view_for("coder")) < len(board.to_prompt_block())


def test_executor_view_compacts_static_flags():
    board = _populated_board()
    view = json.loads(board.view_for("executor"))
    assert view["static_flags"]
    flag = view["static_flags"][0]
    # Heavy fields dropped; only the locating fields kept.
    assert set(flag.keys()) <= {"source", "file", "line", "flag", "severity"}
    assert "reason" not in flag
    assert "code" not in flag


def test_verifier_view_trims_runtime_excerpt():
    board = _populated_board()
    view = json.loads(board.view_for("verifier"))
    excerpt = view["runtime_trace"]["tool_results"][0]["content_excerpt"]
    # 2000-char excerpt is trimmed for the verifier view.
    assert len(excerpt) < 2000
    # Verifier still sees the evidence reports it must merge.
    assert view["analyzer_report"]
    assert view["executor_report"]


def test_unknown_role_falls_back_to_full_dump():
    board = _populated_board()
    assert board.view_for("orchestrator") == board.to_prompt_block()


def test_blackboard_full_dump_can_be_unbounded():
    board = _populated_board()
    board.verifier_report = "V" * 20_000

    default_dump = board.to_prompt_block()
    unbounded_dump = board.to_prompt_block(max_chars=None)

    assert len(default_dump) < len(unbounded_dump)
    assert "... [truncated] ..." in default_dump
    assert "V" * 1000 in unbounded_dump


def test_verified_suspects_fall_back_when_no_verifier_lines():
    board = ErrorBlackboard(task_stem="cuda_demo")
    board.add_suspected_line({
        "source": "analyzer", "file": "solution.cu", "line": 5,
        "bug_description": "off-by-one",
    })
    view = json.loads(board.view_for("coder"))
    # No verifier line present -> fall back to all suspects rather than empty.
    assert len(view["suspected_lines"]) == 1


def test_scoped_views_config_gate():
    assert _scoped_views_enabled({"agent": {"blackboard": {"scoped_views": True}}})
    assert not _scoped_views_enabled({"agent": {"blackboard": {"scoped_views": False}}})
    assert not _scoped_views_enabled({"agent": {"blackboard": {}}})
    assert not _scoped_views_enabled({})


def test_token_accumulation_estimate_and_usage():
    state = _LoopState()
    messages = [{"role": "user", "content": "x" * 400}]  # ~100 est tokens @4 cpt

    # First call: real usage present.
    _accumulate_token_usage(
        state, "coder", messages,
        {"usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}},
        chars_per_token=4.0,
    )
    # Second call: no usage object (estimate only).
    _accumulate_token_usage(state, "coder", messages, None, chars_per_token=4.0)

    bucket = state.token_usage["coder"]
    assert bucket["calls"] == 2
    assert bucket["est_prompt_tokens"] > 0
    assert bucket["prompt_tokens"] == 120
    assert bucket["completion_tokens"] == 30
    assert bucket["total_tokens"] == 150


def test_token_accumulation_handles_usage_object():
    """usage may be an SDK object (attributes) rather than a dict."""
    class _Usage:
        prompt_tokens = 10
        completion_tokens = 5
        total_tokens = 15

    state = _LoopState()
    _accumulate_token_usage(
        state, "verifier", [{"role": "user", "content": "hi"}],
        {"usage": _Usage()}, chars_per_token=4.0,
    )
    assert state.token_usage["verifier"]["total_tokens"] == 15


def test_structured_reports_config_gate():
    assert _structured_reports_enabled(
        {"agent": {"blackboard": {"structured_reports": True}}})
    assert not _structured_reports_enabled(
        {"agent": {"blackboard": {"structured_reports": False}}})
    assert not _structured_reports_enabled({"agent": {"blackboard": {}}})
    assert not _structured_reports_enabled({})


def test_normalize_structured_report_caps_and_shapes():
    raw = {
        "summary": "S" * 1000,
        "key_points": ["a", "b", "c", "d", "e", "f"],  # >4
        "ruled_out": "single hypothesis",               # str -> list
        "extra": "ignored field",
    }
    out = _normalize_structured_report(raw)
    assert set(out.keys()) == {"summary", "key_points", "ruled_out"}
    assert len(out["summary"]) <= 440            # 400 cap + truncation marker
    assert len(out["key_points"]) == 4           # capped to 4
    assert out["ruled_out"] == ["single hypothesis"]


def test_normalize_structured_report_non_dict_degrades_to_summary():
    out = _normalize_structured_report("just a string")
    assert out["summary"] == "just a string"
    assert out["key_points"] == []
    assert out["ruled_out"] == []
    # None payload -> empty report, never raises.
    empty = _normalize_structured_report(None)
    assert empty == {"summary": "", "key_points": [], "ruled_out": []}


def test_role_action_request_mentions_structured_schema():
    schemas = [{"function": {"name": "locate_bug", "description": "x",
                             "parameters": {}}}]
    free = _json_role_action_request(schemas, "analyzer", structured_report=False)
    structured = _json_role_action_request(schemas, "analyzer", structured_report=True)
    assert "compact evidence summary" in free
    assert "summary" in structured and "ruled_out" in structured
    assert "compact evidence summary" not in structured


def test_structured_report_stored_as_dict_serialises_in_views():
    board = _populated_board()
    board.analyzer_report = {
        "summary": "missing barrier", "key_points": ["line 12"], "ruled_out": []}
    # verifier view embeds the structured dict; full dump still serialises.
    view = json.loads(board.view_for("verifier"))
    assert view["analyzer_report"]["summary"] == "missing barrier"
    json.loads(board.to_prompt_block())  # must not raise


def test_tried_strategies_prompt_lists_failed_attempts_only():
    from harness.cudeubgger.common import _LoopState, _tried_strategies_prompt
    st = _LoopState()
    st.attempt_history = [
        {"iteration": 1, "located": {"bug_description": "missing barrier"},
         "fix_direction": "add __syncthreads",
         "classification": {"category": "wrong_result", "five_category": "logic_error"}},
        {"iteration": 2, "located": {"bug_description": "OOB write"},
         "fix_direction": "guard index",
         "classification": {"five_category": "passed"}},  # passed -> excluded
    ]
    out = _tried_strategies_prompt(st)
    assert "ALREADY TRIED" in out
    assert "add __syncthreads" in out and "missing barrier" in out
    assert "guard index" not in out  # passed attempt is not a "tried failure"


def test_tried_strategies_prompt_empty_is_none():
    from harness.cudeubgger.common import _LoopState, _tried_strategies_prompt
    assert _tried_strategies_prompt(_LoopState()) is None


def test_route_decision_parses_and_defaults(monkeypatch):
    from harness.cudeubgger import blackboard as bb
    from harness.cudeubgger.common import _LoopState
    cfg = {"agent": {}}

    monkeypatch.setattr(bb.deps, "call_llm", lambda *a, **k:
                        '{"next_action":"retry_fix","diagnosis":"coder_error","reason":"typo"}')
    out = bb._run_route_decision(None, cfg, _LoopState(), "code",
                                 {"line": 1}, "fix dir", {"classification": {}}, None)
    assert out["next_action"] == "retry_fix"
    assert out["diagnosis"] == "coder_error"

    # Unparseable -> safe default (reanalyze).
    monkeypatch.setattr(bb.deps, "call_llm", lambda *a, **k: "not json at all")
    out2 = bb._run_route_decision(None, cfg, _LoopState(), "code", None, None, None, None)
    assert out2["next_action"] == "reanalyze"

    # Invalid next_action value -> safe default (reanalyze).
    monkeypatch.setattr(bb.deps, "call_llm", lambda *a, **k: '{"next_action":"banana"}')
    out3 = bb._run_route_decision(None, cfg, _LoopState(), "code", None, None, None, None)
    assert out3["next_action"] == "reanalyze"

    # API error -> safe default (reanalyze), never raises.
    def _boom(*a, **k):
        raise RuntimeError("endpoint down")
    monkeypatch.setattr(bb.deps, "call_llm", _boom)
    out4 = bb._run_route_decision(None, cfg, _LoopState(), "code", None, None, None, None)
    assert out4["next_action"] == "reanalyze"


def test_token_usage_summary_totals():
    state = _LoopState()
    msgs = [{"role": "user", "content": "abcd"}]
    _accumulate_token_usage(state, "analyzer", msgs,
                            {"usage": {"prompt_tokens": 10, "completion_tokens": 2,
                                       "total_tokens": 12}})
    _accumulate_token_usage(state, "coder", msgs,
                            {"usage": {"prompt_tokens": 20, "completion_tokens": 4,
                                       "total_tokens": 24}})
    summary = _token_usage_summary(state)
    assert set(summary["per_role"].keys()) == {"analyzer", "coder"}
    assert summary["total"]["calls"] == 2
    assert summary["total"]["total_tokens"] == 36
    assert summary["total"]["prompt_tokens"] == 30
