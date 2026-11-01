#!/usr/bin/env python3
"""End-to-end agent_loop tests with a mock LLM client and stubbed compile/test.

These tests exercise the full loop logic — tool dispatch, write_solution
build/test handling, context management, stagnation, iter accounting —
without touching a real LLM, GPU, or nvcc.
"""

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Mock LLM client — programmable response queue
# ---------------------------------------------------------------------------


class MockChoiceMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.reasoning_content = None
        self.reasoning = None


class MockToolCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.type = "function"

        class _Fn:
            pass
        self.function = _Fn()
        self.function.name = name
        self.function.arguments = arguments


class MockChoice:
    def __init__(self, message, finish_reason="stop"):
        self.message = message
        self.finish_reason = finish_reason


class MockResponse:
    def __init__(self, message):
        self.choices = [MockChoice(message)]


class MockChatCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise RuntimeError("MockClient ran out of scripted responses")
        return self.responses.pop(0)


class MockClient:
    """Mimics the OpenAI client surface used by call_llm_with_tools."""

    def __init__(self, responses):
        class _Chat:
            pass
        self.chat = _Chat()
        self.chat.completions = MockChatCompletions(responses)


def _resp(content=None, tool_calls=None):
    return MockResponse(MockChoiceMessage(content=content, tool_calls=tool_calls or []))


def _tc(name, args: dict, call_id=None):
    cid = call_id or f"call_{name}_{id(args)}"
    return MockToolCall(cid, name, json.dumps(args))


# ---------------------------------------------------------------------------
# Stub compile/test — patch the agent_loop module
# ---------------------------------------------------------------------------


@pytest.fixture
def stubbed_loop(tmp_path, monkeypatch):
    """Build a minimal dataset + workspace and stub out compile/test."""
    dataset = tmp_path / "dataset"
    (dataset / "input").mkdir(parents=True)
    (dataset / "prompts").mkdir(parents=True)
    (dataset / "testbench" / "CUDA_42" / "test").mkdir(parents=True)
    (dataset / "input" / "CUDA_42.json").write_text(json.dumps({
        "task_id": "CUDA/42",
        "build_command": "nvcc solution.cu -o test.out",
        "test_command": "./test.out",
        "solution_file": "solution.cu",
        "difficulty": "medium",
        "level": "compute_eval",
    }))
    (dataset / "prompts" / "CUDA_42.txt").write_text("Write a kernel.\n")
    (dataset / "testbench" / "CUDA_42" / "test" / "test_main.cu").write_text(
        "int main() { return 0; }\n"
    )
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    workdir = tmp_path / "ws"
    workdir.mkdir()

    base_config = {
        "experiment": {"max_iterations": 5},
        "build": {
            "timeout": 30, "test_timeout": 30,
            "workdir": str(workdir),
        },
        "api": {"model": "mock-model"},
        "agent": {
            "max_tool_calls_per_task": 20,
            "tools": None,  # all
            "full_file_submissions": False,
            "token_budget": 50_000,
            # These tests exercise loop mechanics with a scripted MockClient;
            # the summariser would consume scripted responses. Summarisation
            # is covered by test_context_summary / test_context_manager.
            "use_llm_summary": False,
        },
    }

    # Programmable compile/test results — tests can append to these lists
    compile_results = []   # list[(build_ok, build_output)]
    test_results = []      # list[(returncode, test_output)]

    def fake_compile(code, stem, ds, meta, wd, timeout):
        if compile_results:
            return compile_results.pop(0)
        return (True, "0 errors")

    def fake_run_test(stem, meta, wd, timeout):
        if test_results:
            return test_results.pop(0)
        return (0, "all checks passed")

    import harness.agent_loop as agent_loop
    monkeypatch.setattr(agent_loop, "compile_code", fake_compile)
    monkeypatch.setattr(agent_loop, "run_test", fake_run_test)

    # The agent loop is debug-only — every task needs a broken_start.
    # Use an absolute path so `repo_root / code_path` resolves to it.
    broken_file = tmp_path / "broken_v0.cu"
    broken_file.write_text("seed_broken_v0\n")
    broken_start = {
        "code_path": str(broken_file),
        "error_category": "buildability_bug",
        "log_excerpt": "seeded broken code for tests",
    }

    return {
        "broken_start": broken_start,
        "dataset": dataset,
        "output_dir": output_dir,
        "workdir": workdir,
        "config": base_config,
        "compile_results": compile_results,
        "test_results": test_results,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_agent_passes_on_first_submission(stubbed_loop):
    """Happy path (debug mode): read_prompt → locate + apply_edit + write_solution → passes."""
    responses = [
        _resp(tool_calls=[_tc("read_prompt", {})]),                # LOCATE phase: investigate
        _patch_turn("seed_broken_v0", "ok_code", locate_line=1, rationale="wrote it"),
    ]
    client = MockClient(responses)
    stubbed_loop["compile_results"].append((True, "ok"))
    stubbed_loop["test_results"].append((0, "all checks passed"))

    from harness.agent_loop import run_task_agent
    records, stop = run_task_agent(
        task_stem="CUDA_42",
        dataset_dir=stubbed_loop["dataset"],
        output_dir=stubbed_loop["output_dir"],
        config=stubbed_loop["config"],
        client=client,
        broken_start=stubbed_loop["broken_start"],
    )
    assert stop == "passed"
    assert len(records) == 1
    assert records[0]["status"] == "passed"
    assert records[0]["iteration"] == 1
    assert records[0]["method"] == "agent"
    trace_file = stubbed_loop["output_dir"] / "tool_trace" / "CUDA_42.jsonl"
    assert trace_file.exists()
    tools = [json.loads(line)["tool"] for line in trace_file.read_text().strip().splitlines()]
    # read_prompt, locate_bug, apply_edit, write_solution
    assert tools == ["read_prompt", "locate_bug", "apply_edit", "write_solution"]


def test_agent_accepts_project_relative_broken_start_path(stubbed_loop):
    """Legacy task lists store broken_start code_path relative to repo root."""
    rel_broken_start = dict(stubbed_loop["broken_start"])
    rel_broken_start["code_path"] = "tests/data/agent_relative_broken.cu"
    responses = [
        _patch_turn("relative_seed_v0", "ok_code", locate_line=1, rationale="relative path"),
    ]
    client = MockClient(responses)
    stubbed_loop["compile_results"].append((True, "ok"))
    stubbed_loop["test_results"].append((0, "all checks passed"))

    from harness.agent_loop import run_task_agent
    records, stop = run_task_agent(
        task_stem="CUDA_42",
        dataset_dir=stubbed_loop["dataset"],
        output_dir=stubbed_loop["output_dir"],
        config=stubbed_loop["config"],
        client=client,
        broken_start=rel_broken_start,
    )

    assert stop == "passed"
    assert len(records) == 1


def test_agent_can_force_required_tool_choice(stubbed_loop):
    """agent.tool_choice is passed through to the tool-calling API."""
    stubbed_loop["config"]["agent"]["tool_choice"] = "required"
    responses = [
        _patch_turn("seed_broken_v0", "ok_code", locate_line=1),
    ]
    client = MockClient(responses)
    stubbed_loop["compile_results"].append((True, "ok"))
    stubbed_loop["test_results"].append((0, "all checks passed"))

    from harness.agent_loop import run_task_agent
    records, stop = run_task_agent(
        task_stem="CUDA_42",
        dataset_dir=stubbed_loop["dataset"],
        output_dir=stubbed_loop["output_dir"],
        config=stubbed_loop["config"],
        client=client,
        broken_start=stubbed_loop["broken_start"],
    )

    assert stop == "passed"
    assert len(records) == 1
    assert client.chat.completions.calls[0]["tool_choice"] == "required"


def test_agent_caps_tool_decision_tokens(stubbed_loop):
    """Tool-decision turns use the configured tool token cap."""
    stubbed_loop["config"]["api"].update({
        "backend": "vllm",
        "max_tokens": 8192,
        "temperature": 0,
    })
    stubbed_loop["config"]["agent"]["tool_decision_max_tokens"] = 1024
    responses = [
        _resp(tool_calls=[_locate(1)]),
        _resp(tool_calls=[
            _tc("apply_edit", {
                "old_string": "seed_broken_v0",
                "new_string": "ok_code",
            }),
            _tc("write_solution", {}),
        ]),
    ]
    client = MockClient(responses)
    stubbed_loop["compile_results"].append((True, "ok"))
    stubbed_loop["test_results"].append((0, "all checks passed"))

    from harness.agent_loop import run_task_agent
    records, stop = run_task_agent(
        task_stem="CUDA_42",
        dataset_dir=stubbed_loop["dataset"],
        output_dir=stubbed_loop["output_dir"],
        config=stubbed_loop["config"],
        client=client,
        broken_start=stubbed_loop["broken_start"],
    )

    assert stop == "passed"
    assert len(records) == 1
    assert client.chat.completions.calls[0]["max_tokens"] == 1024
    assert client.chat.completions.calls[0]["temperature"] == 0
    assert client.chat.completions.calls[1]["max_tokens"] == 1024
    assert client.chat.completions.calls[1]["temperature"] == 0


def test_agent_json_action_protocol_executes_deterministically(stubbed_loop):
    """JSON action mode avoids native tool calls and still uses the dispatcher."""
    stubbed_loop["config"]["api"].update({
        "backend": "vllm",
        "max_tokens": 8192,
        "temperature": 0,
    })
    stubbed_loop["config"]["agent"].update({
        "action_protocol": "json",
        "tool_decision_max_tokens": 2048,
    })
    responses = [
        _resp(content='{"tool":"read_prompt","args":{}}'),
        _resp(content=json.dumps({
            "tool": "locate_bug",
            "args": {
                "file": "solution.cu",
                "line": 1,
                "bug_description": "bad seed",
                "evidence": "prompt + code",
            },
        })),
        _resp(content=json.dumps({
            "tool": "apply_edit",
            "args": {
                "old_string": "seed_broken_v0",
                "new_string": "ok_code",
            },
        })),
        _resp(content='{"tool":"write_solution","args":{}}'),
    ]
    client = MockClient(responses)
    stubbed_loop["compile_results"].append((True, "ok"))
    stubbed_loop["test_results"].append((0, "all checks passed"))

    from harness.agent_loop import run_task_agent
    records, stop = run_task_agent(
        task_stem="CUDA_42",
        dataset_dir=stubbed_loop["dataset"],
        output_dir=stubbed_loop["output_dir"],
        config=stubbed_loop["config"],
        client=client,
        broken_start=stubbed_loop["broken_start"],
    )

    assert stop == "passed"
    assert len(records) == 1
    assert [c["max_tokens"] for c in client.chat.completions.calls] == [
        2048, 2048, 2048, 2048,
    ]
    for call in client.chat.completions.calls:
        assert "tools" not in call
        assert "tool_choice" not in call
        assert call["temperature"] == 0
        assert call["response_format"] == {"type": "json_object"}
        assert all(m["role"] != "tool" for m in call["messages"])


def _locate(line=1):
    """A locate_bug tool call with all required fields filled."""
    return _tc("locate_bug", {
        "file": "solution.cu", "line": line,
        "bug_description": "suspect line", "evidence": "hand-traced",
    })


def _patch_turn(old, new, locate_line=1, rationale=""):
    """One canonical FIX-flow turn: locate + apply_edit + write_solution()."""
    args = {"rationale": rationale} if rationale else {}
    return _resp(tool_calls=[
        _locate(locate_line),
        _tc("apply_edit", {"old_string": old, "new_string": new}),
        _tc("write_solution", args),
    ])


def test_agent_iterates_on_failure(stubbed_loop):
    """Iter 1 patch fails; iter 2 patches again and passes."""
    responses = [
        # iter 1: patch the seed → broken v1, submit. Will fail.
        _patch_turn("seed_broken_v0", "broken_v1", locate_line=1),
        # After failure, model investigates again, locates, patches, submits.
        _resp(tool_calls=[_tc("read_code", {"start": 1, "end": 5})]),
        _patch_turn("broken_v1", "fixed_v2", locate_line=3),
    ]
    client = MockClient(responses)
    stubbed_loop["compile_results"].append((False, "solution.cu(1): error: bad code"))
    stubbed_loop["compile_results"].append((True, "ok"))
    stubbed_loop["test_results"].append((0, "all checks passed"))

    from harness.agent_loop import run_task_agent
    records, stop = run_task_agent(
        task_stem="CUDA_42",
        dataset_dir=stubbed_loop["dataset"],
        output_dir=stubbed_loop["output_dir"],
        config=stubbed_loop["config"],
        client=client,
        broken_start=stubbed_loop["broken_start"],
    )
    assert stop == "passed"
    assert len(records) == 2
    assert records[0]["status"] == "failed"
    assert records[0]["iteration"] == 1
    assert records[1]["status"] == "passed"
    assert records[1]["iteration"] == 2
    # In debug mode every iteration carries a locate.
    assert records[0]["located"]["line"] == 1
    assert records[1]["located"]["line"] == 3


def test_failed_patch_verifier_receives_outputs_and_location(stubbed_loop, monkeypatch):
    """debug_tools.patch_verifier sees the failed build/test output and locate_bug claim."""
    stubbed_loop["config"]["agent"]["max_patch_submissions"] = 1
    stubbed_loop["config"]["debug_tools"] = {
        "enabled": True,
        "patch_verifier": {"enabled": True, "mode": "on_failed_attempt"},
    }
    captured = {}

    def fake_verify_failed_patch(**kwargs):
        captured.update(kwargs)
        return {"status": "ok", "parsed": {"patch_issue": "still wrong"}}

    import harness.agent_loop as agent_loop
    monkeypatch.setattr(agent_loop, "verify_failed_patch", fake_verify_failed_patch)

    responses = [
        _patch_turn("seed_broken_v0", "broken_v1", locate_line=7, rationale="try fix"),
    ]
    client = MockClient(responses)
    stubbed_loop["compile_results"].append((False, "solution.cu(7): error: bad code"))

    from harness.agent_loop import run_task_agent
    records, stop = run_task_agent(
        task_stem="CUDA_42",
        dataset_dir=stubbed_loop["dataset"],
        output_dir=stubbed_loop["output_dir"],
        config=stubbed_loop["config"],
        client=client,
        broken_start=stubbed_loop["broken_start"],
    )

    assert stop == "max_patch_submissions"
    assert len(records) == 1
    assert records[0]["located"]["line"] == 7
    assert captured["build_output"] == "solution.cu(7): error: bad code"
    assert captured["test_output"] is None
    assert captured["localization"]["line"] == 7


def test_agent_hits_max_patch_submissions(stubbed_loop):
    """Every submission fails; loop stops at max_patch_submissions."""
    stubbed_loop["config"]["experiment"]["max_iterations"] = 99
    stubbed_loop["config"]["agent"]["max_patch_submissions"] = 2

    responses = [
        _patch_turn("seed_broken_v0", "v1", locate_line=1),
        _patch_turn("v1", "v2", locate_line=1),
        _patch_turn("v2", "v3", locate_line=2),  # extra in case loop survives
    ]
    client = MockClient(responses)
    stubbed_loop["compile_results"].append((True, "ok"))
    stubbed_loop["compile_results"].append((True, "ok"))
    stubbed_loop["test_results"].append((1, "Expected 1, Got 0"))
    stubbed_loop["test_results"].append((1, "Expected 1, Got 0"))

    from harness.agent_loop import run_task_agent
    records, stop = run_task_agent(
        task_stem="CUDA_42",
        dataset_dir=stubbed_loop["dataset"],
        output_dir=stubbed_loop["output_dir"],
        config=stubbed_loop["config"],
        client=client,
        broken_start=stubbed_loop["broken_start"],
    )
    assert stop == "max_patch_submissions"
    assert len(records) == 2


def test_agent_stagnation_signal_does_not_stop_before_patch_budget(stubbed_loop):
    """Same-category failures are telemetry; patch budget is the stop boundary."""
    stubbed_loop["config"]["experiment"]["max_iterations"] = 99
    stubbed_loop["config"]["agent"]["max_patch_submissions"] = 4

    responses = [
        _patch_turn("seed_broken_v0", "v1", locate_line=1),
        _patch_turn("v1", "v2", locate_line=2),
        _patch_turn("v2", "v3", locate_line=3),
        _patch_turn("v3", "v4", locate_line=4),
    ]
    client = MockClient(responses)
    for _ in range(4):
        stubbed_loop["compile_results"].append((True, "ok"))
        stubbed_loop["test_results"].append((1, "Expected 1, Got 0"))

    from harness.agent_loop import run_task_agent
    records, stop = run_task_agent(
        task_stem="CUDA_42",
        dataset_dir=stubbed_loop["dataset"],
        output_dir=stubbed_loop["output_dir"],
        config=stubbed_loop["config"],
        client=client,
        broken_start=stubbed_loop["broken_start"],
    )
    assert stop == "max_patch_submissions"
    assert len(records) == 4
    assert [r["located"]["line"] for r in records] == [1, 2, 3, 4]


def test_repeated_failed_apply_edit_forces_recovery_not_task_stop(stubbed_loop):
    """Repeating one bad old_string should force a fresh read, not end the task."""
    stubbed_loop["config"]["agent"]["max_patch_submissions"] = 1
    bad_edit = _tc("apply_edit", {
        "old_string": "not_in_current_code",
        "new_string": "bad_patch",
    })
    responses = [
        _resp(tool_calls=[_locate(1)]),
        _resp(tool_calls=[bad_edit]),
        _resp(tool_calls=[_tc("read_code", {"full_file": True})]),
        _resp(tool_calls=[bad_edit]),
        _resp(tool_calls=[_tc("read_code", {"full_file": True})]),
        _resp(tool_calls=[bad_edit]),
        _resp(tool_calls=[_tc("read_code", {"full_file": True})]),
        _resp(tool_calls=[
            _tc("apply_edit", {
                "old_string": "seed_broken_v0",
                "new_string": "fixed_v1",
            }),
            _tc("write_solution", {"rationale": "fresh edit after failed matches"}),
        ]),
    ]
    client = MockClient(responses)
    stubbed_loop["compile_results"].append((True, "ok"))
    stubbed_loop["test_results"].append((0, "all checks passed"))

    from harness.agent_loop import run_task_agent
    records, stop = run_task_agent(
        task_stem="CUDA_42",
        dataset_dir=stubbed_loop["dataset"],
        output_dir=stubbed_loop["output_dir"],
        config=stubbed_loop["config"],
        client=client,
        broken_start=stubbed_loop["broken_start"],
    )

    assert stop == "passed"
    assert len(records) == 1
    assert records[0]["patch_actions"][-1]["ok"] is True
    trace = stubbed_loop["output_dir"] / "tool_trace" / "CUDA_42.jsonl"
    tools = [json.loads(line)["tool"] for line in trace.read_text().strip().splitlines()]
    assert tools == [
        "locate_bug",
        "apply_edit",
        "read_code",
        "apply_edit",
        "read_code",
        "apply_edit",
        "read_code",
        "apply_edit",
        "write_solution",
    ]


def test_agent_no_op_aborts(stubbed_loop):
    """Model returns no tool call twice in a row → abort with agent_no_op."""
    responses = [
        _resp(content="just thinking..."),
        _resp(content="still thinking..."),
    ]
    client = MockClient(responses)

    from harness.agent_loop import run_task_agent
    records, stop = run_task_agent(
        task_stem="CUDA_42",
        dataset_dir=stubbed_loop["dataset"],
        output_dir=stubbed_loop["output_dir"],
        config=stubbed_loop["config"],
        client=client,
        broken_start=stubbed_loop["broken_start"],
    )
    assert stop == "agent_no_op"
    assert records == []


def test_agent_tool_budget_caps(stubbed_loop):
    """Once max_tool_calls_per_task is hit, further calls return tool budget error."""
    stubbed_loop["config"]["agent"]["max_tool_calls_per_task"] = 2
    responses = [
        _resp(tool_calls=[_tc("read_prompt", {}, call_id="c1")]),
        _resp(tool_calls=[_tc("read_prompt", {}, call_id="c2")]),
        _resp(tool_calls=[_tc("read_prompt", {}, call_id="c3")]),
        _resp(content="ok"),
    ]
    client = MockClient(responses)

    from harness.agent_loop import run_task_agent
    records, stop = run_task_agent(
        task_stem="CUDA_42",
        dataset_dir=stubbed_loop["dataset"],
        output_dir=stubbed_loop["output_dir"],
        config=stubbed_loop["config"],
        client=client,
        broken_start=stubbed_loop["broken_start"],
    )
    assert stop == "tool_budget_exhausted"


def test_agent_writes_tool_trace(stubbed_loop):
    """Tool trace JSONL records every call with timing."""
    responses = [
        _resp(tool_calls=[_tc("read_prompt", {})]),
        _patch_turn("seed_broken_v0", "ok", rationale="done"),
    ]
    client = MockClient(responses)
    stubbed_loop["compile_results"].append((True, "ok"))
    stubbed_loop["test_results"].append((0, "passed"))

    from harness.agent_loop import run_task_agent
    run_task_agent(
        task_stem="CUDA_42",
        dataset_dir=stubbed_loop["dataset"],
        output_dir=stubbed_loop["output_dir"],
        config=stubbed_loop["config"],
        client=client,
        broken_start=stubbed_loop["broken_start"],
    )
    trace = stubbed_loop["output_dir"] / "tool_trace" / "CUDA_42.jsonl"
    lines = [json.loads(line) for line in trace.read_text().strip().splitlines()]
    assert [r["tool"] for r in lines] == [
        "read_prompt", "locate_bug", "apply_edit", "write_solution",
    ]
    assert all("elapsed_ms" in line for line in lines)
    assert all(line["iteration"] >= 1 for line in lines)


def test_blackboard_workflow_runs_agents_then_coder(stubbed_loop):
    """Blackboard workflow: analyzer/executor/verifier populate evidence,
    then coder applies a patch-only submission."""
    stubbed_loop["config"]["agent"]["workflow"] = "blackboard"
    Path(stubbed_loop["broken_start"]["code_path"]).write_text(
        "extern \"C\" __global__ void k(float* out) {\n"
        "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "  out[i] = 1.0f;\n"
        "}\n"
    )
    responses = [
        # Analyzer: call static tool + locate a suspected line, then report.
        _resp(tool_calls=[
            _tc("run_static_analysis", {}),
            _tc("locate_bug", {
                "file": "solution.cu",
                "line": 3,
                "bug_description": "unchecked global write index",
                "evidence": "static analysis flagged memory write",
            }),
        ]),
        _resp(content="static report: line 3 lacks a bounds guard"),
        # Executor: runtime observation is already supplied by the harness.
        _resp(content="runtime report: test reports wrong result"),
        # Verifier: structured line-level report.
        _resp(content=json.dumps({
            "error_type": "illegal_memory_access",
            "line_level_bug_report": [{
                "file": "solution.cu",
                "line": 3,
                "confidence": "high",
                "root_cause": "unchecked write can go out of bounds",
                "evidence": ["static flag", "runtime failure"],
            }],
            "fix_suggestions": ["guard i before writing out[i]"],
        })),
        # Coder: confirm line, patch, submit.
        _resp(tool_calls=[
            _locate(3),
            _tc("apply_edit", {
                "old_string": "  out[i] = 1.0f;",
                "new_string": "  if (i < 1) out[i] = 1.0f;",
            }),
            _tc("write_solution", {"rationale": "guard global write"}),
        ]),
    ]
    client = MockClient(responses)
    # Runtime observation before executor.
    stubbed_loop["compile_results"].append((True, "ok"))
    stubbed_loop["test_results"].append((1, "illegal memory access"))
    # Coder submission.
    stubbed_loop["compile_results"].append((True, "ok"))
    stubbed_loop["test_results"].append((0, "all checks passed"))

    from harness.agent_loop import run_task_agent
    records, stop = run_task_agent(
        task_stem="CUDA_42",
        dataset_dir=stubbed_loop["dataset"],
        output_dir=stubbed_loop["output_dir"],
        config=stubbed_loop["config"],
        client=client,
        broken_start=stubbed_loop["broken_start"],
    )

    assert stop == "passed"
    assert len(records) == 1
    assert records[0]["agent_workflow"] == "blackboard"
    board = records[0]["error_blackboard"]
    assert board["error_type"] == "illegal_memory_access"
    assert any(s["source"] == "analyzer" and s["line"] == 3 for s in board["suspected_lines"])
    assert any(s["source"] == "verifier" and s["line"] == 3 for s in board["suspected_lines"])
    assert board["static_flags"]
    trace = stubbed_loop["output_dir"] / "tool_trace" / "CUDA_42.jsonl"
    tools = [json.loads(line)["tool"] for line in trace.read_text().strip().splitlines()]
    assert tools == [
        "run_static_analysis",
        "locate_bug",
        "locate_bug",
        "apply_edit",
        "write_solution",
    ]


def test_multi_role_no_blackboard_workflow_runs_without_shared_board(stubbed_loop):
    """No-blackboard ablation keeps the roles but only passes compact reports."""
    stubbed_loop["config"]["agent"]["workflow"] = "multi_role_no_blackboard"
    Path(stubbed_loop["broken_start"]["code_path"]).write_text(
        "extern \"C\" __global__ void k(float* out) {\n"
        "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "  out[i] = 1.0f;\n"
        "}\n"
    )
    responses = [
        _resp(tool_calls=[
            _tc("run_static_analysis", {}),
            _tc("locate_bug", {
                "file": "solution.cu",
                "line": 3,
                "bug_description": "unchecked global write index",
                "evidence": "static analysis flagged memory write",
            }),
        ]),
        _resp(content="static report: line 3 lacks a bounds guard"),
        _resp(content="runtime report: test reports wrong result"),
        _resp(content=json.dumps({
            "error_type": "illegal_memory_access",
            "line_level_bug_report": [{
                "file": "solution.cu",
                "line": 3,
                "confidence": "high",
                "root_cause": "unchecked write can go out of bounds",
                "evidence": ["static flag", "runtime failure"],
            }],
            "fix_suggestions": ["guard i before writing out[i]"],
        })),
        _resp(tool_calls=[
            _locate(3),
            _tc("apply_edit", {
                "old_string": "  out[i] = 1.0f;",
                "new_string": "  if (i < 1) out[i] = 1.0f;",
            }),
            _tc("write_solution", {"rationale": "guard global write"}),
        ]),
    ]
    client = MockClient(responses)
    # Runtime observation before executor.
    stubbed_loop["compile_results"].append((True, "ok"))
    stubbed_loop["test_results"].append((1, "illegal memory access"))
    # Coder submission.
    stubbed_loop["compile_results"].append((True, "ok"))
    stubbed_loop["test_results"].append((0, "all checks passed"))

    from harness.agent_loop import run_task_agent
    records, stop = run_task_agent(
        task_stem="CUDA_42",
        dataset_dir=stubbed_loop["dataset"],
        output_dir=stubbed_loop["output_dir"],
        config=stubbed_loop["config"],
        client=client,
        broken_start=stubbed_loop["broken_start"],
    )

    assert stop == "passed"
    assert len(records) == 1
    assert records[0]["agent_workflow"] == "multi_role_no_blackboard"
    assert "error_blackboard" not in records[0]
    role_reports = records[0]["role_reports"]
    assert role_reports["error_type"] == "illegal_memory_access"
    assert role_reports["analyzer_report"] == "static report: line 3 lacks a bounds guard"
    assert role_reports["executor_report"] == "runtime report: test reports wrong result"
    assert role_reports["next_fix"] == "guard i before writing out[i]"
    trace = stubbed_loop["output_dir"] / "tool_trace" / "CUDA_42.jsonl"
    tools = [json.loads(line)["tool"] for line in trace.read_text().strip().splitlines()]
    assert tools == [
        "run_static_analysis",
        "locate_bug",
        "locate_bug",
        "apply_edit",
        "write_solution",
    ]


def test_blackboard_submit_guard_writes_after_apply_edit(stubbed_loop):
    """If blackboard coder edits but omits write_solution, submit the patch."""
    stubbed_loop["config"]["agent"]["workflow"] = "blackboard"
    stubbed_loop["config"]["agent"]["blackboard"] = {
        "max_role_turns": 1,
        "max_coder_turns": 1,
    }
    Path(stubbed_loop["broken_start"]["code_path"]).write_text("seed_broken_v0\n")

    responses = [
        _resp(content="static report"),
        _resp(content="runtime report"),
        _resp(content=json.dumps({
            "error_type": "functional_correctness_bug",
            "line_level_bug_report": [{
                "file": "solution.cu",
                "line": 1,
                "confidence": "high",
                "root_cause": "seed code is wrong",
                "evidence": ["test failed"],
            }],
            "fix_suggestions": ["replace seed with fixed code"],
        })),
        # Coder patches but forgets to call write_solution.
        _resp(tool_calls=[
            _locate(1),
            _tc("apply_edit", {
                "old_string": "seed_broken_v0",
                "new_string": "fixed_v1",
            }),
        ]),
    ]
    client = MockClient(responses)
    # Runtime observation before executor.
    stubbed_loop["compile_results"].append((True, "ok"))
    stubbed_loop["test_results"].append((1, "wrong result"))
    # Submit guard submission.
    stubbed_loop["compile_results"].append((True, "ok"))
    stubbed_loop["test_results"].append((0, "all checks passed"))

    from harness.agent_loop import run_task_agent
    records, stop = run_task_agent(
        task_stem="CUDA_42",
        dataset_dir=stubbed_loop["dataset"],
        output_dir=stubbed_loop["output_dir"],
        config=stubbed_loop["config"],
        client=client,
        broken_start=stubbed_loop["broken_start"],
    )

    assert stop == "passed"
    assert len(records) == 1
    assert records[0]["status"] == "passed"
    trace = stubbed_loop["output_dir"] / "tool_trace" / "CUDA_42.jsonl"
    tools = [json.loads(line)["tool"] for line in trace.read_text().strip().splitlines()]
    assert tools == ["locate_bug", "apply_edit", "write_solution"]


def test_blackboard_coder_must_apply_edit_after_locate(stubbed_loop):
    """After coder locates a bug, the next blackboard coder turn only offers apply_edit."""
    stubbed_loop["config"]["agent"]["workflow"] = "blackboard"
    stubbed_loop["config"]["agent"]["blackboard"] = {
        "max_role_turns": 1,
        "max_coder_turns": 2,
    }
    Path(stubbed_loop["broken_start"]["code_path"]).write_text("seed_broken_v0\n")

    responses = [
        _resp(content="static report"),
        _resp(content="runtime report"),
        _resp(content=json.dumps({
            "error_type": "functional_correctness_bug",
            "line_level_bug_report": [{
                "file": "solution.cu",
                "line": 1,
                "confidence": "high",
                "root_cause": "seed code is wrong",
                "evidence": ["test failed"],
            }],
            "fix_suggestions": ["replace seed with fixed code"],
        })),
        # First coder turn locates only.
        _resp(tool_calls=[_locate(1)]),
        # The next coder turn should be forced to patch.
        _resp(tool_calls=[
            _tc("apply_edit", {
                "old_string": "seed_broken_v0",
                "new_string": "fixed_v1",
            }),
        ]),
    ]
    client = MockClient(responses)
    stubbed_loop["compile_results"].append((True, "ok"))
    stubbed_loop["test_results"].append((1, "wrong result"))
    stubbed_loop["compile_results"].append((True, "ok"))
    stubbed_loop["test_results"].append((0, "all checks passed"))

    from harness.agent_loop import run_task_agent
    records, stop = run_task_agent(
        task_stem="CUDA_42",
        dataset_dir=stubbed_loop["dataset"],
        output_dir=stubbed_loop["output_dir"],
        config=stubbed_loop["config"],
        client=client,
        broken_start=stubbed_loop["broken_start"],
    )

    assert stop == "passed"
    assert len(records) == 1
    tool_name_sets = [
        [tool["function"]["name"] for tool in call.get("tools", [])]
        for call in client.chat.completions.calls
        if call.get("tools") is not None
    ]
    assert tool_name_sets[-1] == ["apply_edit"]
    trace = stubbed_loop["output_dir"] / "tool_trace" / "CUDA_42.jsonl"
    tools = [json.loads(line)["tool"] for line in trace.read_text().strip().splitlines()]
    assert tools == ["locate_bug", "apply_edit", "write_solution"]


def test_blackboard_coder_no_submission_recovers_next_round(stubbed_loop):
    """A coder round that fails to submit is recoverable, not a task stop."""
    stubbed_loop["config"]["agent"]["workflow"] = "blackboard"
    stubbed_loop["config"]["agent"]["blackboard"] = {
        "max_role_turns": 1,
        "max_coder_turns": 3,
        "max_coder_non_submission_rounds": 2,
    }
    Path(stubbed_loop["broken_start"]["code_path"]).write_text("seed_broken_v0\n")

    verifier_report = {
        "error_type": "functional_correctness_bug",
        "line_level_bug_report": [{
            "file": "solution.cu",
            "line": 1,
            "confidence": "high",
            "root_cause": "seed code is wrong",
            "evidence": ["test failed"],
        }],
        "fix_suggestions": ["replace seed with fixed code"],
    }
    responses = [
        _resp(content="static report 1"),
        _resp(content="runtime report 1"),
        _resp(content=json.dumps(verifier_report)),
        _resp(tool_calls=[_locate(1)]),
        _resp(tool_calls=[
            _tc("apply_edit", {
                "old_string": "not_in_current_code",
                "new_string": "bad_patch",
            }),
        ]),
        _resp(tool_calls=[_tc("read_code", {"full_file": True})]),
        _resp(content="static report 2 sees failed edit"),
        _resp(content="runtime report 2 still wrong"),
        _resp(content=json.dumps(verifier_report)),
        _resp(tool_calls=[
            _locate(1),
            _tc("apply_edit", {
                "old_string": "seed_broken_v0",
                "new_string": "fixed_v1",
            }),
        ]),
    ]
    client = MockClient(responses)
    # Runtime observation before first coder round.
    stubbed_loop["compile_results"].append((True, "ok"))
    stubbed_loop["test_results"].append((1, "wrong result"))
    # Runtime observation before recovered coder round.
    stubbed_loop["compile_results"].append((True, "ok"))
    stubbed_loop["test_results"].append((1, "wrong result"))
    # Auto-submitted recovered patch.
    stubbed_loop["compile_results"].append((True, "ok"))
    stubbed_loop["test_results"].append((0, "all checks passed"))

    from harness.agent_loop import run_task_agent
    records, stop = run_task_agent(
        task_stem="CUDA_42",
        dataset_dir=stubbed_loop["dataset"],
        output_dir=stubbed_loop["output_dir"],
        config=stubbed_loop["config"],
        client=client,
        broken_start=stubbed_loop["broken_start"],
    )

    assert stop == "passed"
    assert len(records) == 1
    assert records[0]["status"] == "passed"
    assert len(records[0]["patch_actions"]) == 1
    assert records[0]["patch_actions"][0]["ok"] is True
    trace = stubbed_loop["output_dir"] / "tool_trace" / "CUDA_42.jsonl"
    tools = [json.loads(line)["tool"] for line in trace.read_text().strip().splitlines()]
    assert tools == [
        "locate_bug",
        "apply_edit",
        "read_code",
        "locate_bug",
        "apply_edit",
        "write_solution",
    ]


def test_blackboard_json_action_protocol_runs_deterministic_tools(stubbed_loop):
    """Blackboard workflow honors JSON action mode instead of native tools."""
    stubbed_loop["config"]["agent"].update({
        "workflow": "blackboard",
        "action_protocol": "json",
        "blackboard": {
            "max_role_turns": 2,
            "max_coder_turns": 2,
        },
    })
    Path(stubbed_loop["broken_start"]["code_path"]).write_text("seed_broken_v0\n")
    responses = [
        _resp(content='{"tool":"run_static_analysis","args":{}}'),
        _resp(content='{"tool":"final_report","args":{"report":"static report"}}'),
        _resp(content='{"tool":"final_report","args":{"report":"runtime report"}}'),
        _resp(content=json.dumps({
            "error_type": "functional_correctness_bug",
            "line_level_bug_report": [{
                "file": "solution.cu",
                "line": 1,
                "confidence": "high",
                "root_cause": "seed code is wrong",
                "evidence": ["static report"],
            }],
            "fix_suggestions": ["replace seed with fixed code"],
        })),
        _resp(content=json.dumps({
            "tool": "locate_bug",
            "args": {
                "file": "solution.cu",
                "line": 1,
                "bug_description": "seed code is wrong",
                "evidence": "verified report",
            },
        })),
        _resp(content=json.dumps({
            "tool": "apply_edit",
            "args": {
                "old_string": "seed_broken_v0",
                "new_string": "fixed_v1",
            },
        })),
    ]
    client = MockClient(responses)
    stubbed_loop["compile_results"].append((True, "ok"))
    stubbed_loop["test_results"].append((1, "wrong result"))
    stubbed_loop["compile_results"].append((True, "ok"))
    stubbed_loop["test_results"].append((0, "all checks passed"))

    from harness.agent_loop import run_task_agent
    records, stop = run_task_agent(
        task_stem="CUDA_42",
        dataset_dir=stubbed_loop["dataset"],
        output_dir=stubbed_loop["output_dir"],
        config=stubbed_loop["config"],
        client=client,
        broken_start=stubbed_loop["broken_start"],
    )

    assert stop == "passed"
    assert len(records) == 1
    assert all("tools" not in call for call in client.chat.completions.calls[:3])
    assert client.chat.completions.calls[0]["response_format"] == {"type": "json_object"}
    trace = stubbed_loop["output_dir"] / "tool_trace" / "CUDA_42.jsonl"
    tools = [json.loads(line)["tool"] for line in trace.read_text().strip().splitlines()]
    assert tools == ["run_static_analysis", "locate_bug", "apply_edit", "write_solution"]


def test_blackboard_perf_gate_failure_relocalizes_and_retries(stubbed_loop, monkeypatch):
    """A correctness pass below perf threshold is a failed attempt with history."""
    stubbed_loop["config"]["experiment"]["max_iterations"] = 3
    stubbed_loop["config"]["agent"].update({
        "workflow": "blackboard",
        "performance_gate": {"enabled": True, "min_speedup": 0.7},
        "blackboard": {
            "max_role_turns": 1,
            "max_coder_turns": 1,
        },
    })
    Path(stubbed_loop["broken_start"]["code_path"]).write_text("seed_broken_v0\n")
    responses = [
        _resp(content="static report 1"),
        _resp(content="runtime report 1"),
        _resp(content=json.dumps({
            "error_type": "functional_correctness_bug",
            "line_level_bug_report": [{"file": "solution.cu", "line": 1, "root_cause": "wrong"}],
            "fix_suggestions": ["make it correct"],
        })),
        _patch_turn("seed_broken_v0", "fixed_slow", rationale="correct but slow"),
        # Router decision after the failed (perf-gate) submission -> reanalyze.
        _resp(content=json.dumps({
            "next_action": "reanalyze",
            "diagnosis": "wrong_strategy",
            "reason": "fix is correct but too slow",
        })),
        _resp(content="static report 2 sees perf history"),
        _resp(content="runtime report 2 sees perf history"),
        _resp(content=json.dumps({
            "error_type": "perf_below_threshold",
            "line_level_bug_report": [{"file": "solution.cu", "line": 1, "root_cause": "slow fix"}],
            "fix_suggestions": ["preserve correctness with faster code"],
        })),
        _patch_turn("fixed_slow", "fixed_fast", rationale="faster fix"),
    ]
    client = MockClient(responses)
    perf_results = [
        {"speedup": 0.1, "candidate_runtime_ms": 10.0, "reference_runtime_ms": 1.0},
        {"speedup": 0.8, "candidate_runtime_ms": 2.0, "reference_runtime_ms": 1.0},
    ]

    import harness.agent_loop as agent_loop
    monkeypatch.setattr(agent_loop, "measure_iter_perf", lambda *a, **k: perf_results.pop(0))

    stubbed_loop["compile_results"].append((True, "ok"))
    stubbed_loop["test_results"].append((1, "wrong result"))
    stubbed_loop["compile_results"].append((True, "ok"))
    stubbed_loop["test_results"].append((0, "all checks passed"))
    stubbed_loop["compile_results"].append((True, "ok"))
    stubbed_loop["test_results"].append((0, "all checks passed"))
    stubbed_loop["compile_results"].append((True, "ok"))
    stubbed_loop["test_results"].append((0, "all checks passed"))

    from harness.agent_loop import run_task_agent
    records, stop = run_task_agent(
        task_stem="CUDA_42",
        dataset_dir=stubbed_loop["dataset"],
        output_dir=stubbed_loop["output_dir"],
        config=stubbed_loop["config"],
        client=client,
        broken_start=stubbed_loop["broken_start"],
    )

    assert stop == "passed"
    assert len(records) == 2
    assert records[0]["status"] == "failed"
    assert records[0]["category"] == "perf_below_threshold"
    assert records[0]["performance_gate"]["passed"] is False
    assert records[0]["performance_gate"]["min_speedup"] == 0.7
    assert records[1]["status"] == "passed"
    assert any(
        "Previous Failed Attempts" in m["content"] and "fixed_slow" in m["content"]
        for call in client.chat.completions.calls
        for m in call["messages"]
        if m["role"] == "user"
    )


# ---------------------------------------------------------------------------
# _sanitize_assistant_content — strip <think> blocks, cap length
# ---------------------------------------------------------------------------


def test_sanitizer_none_and_empty_pass_through():
    from harness.agent_loop import _sanitize_assistant_content
    assert _sanitize_assistant_content(None) is None
    assert _sanitize_assistant_content("") == ""


def test_sanitizer_strips_think_block():
    from harness.agent_loop import _sanitize_assistant_content
    s = "before\n<think>I am thinking\nover multiple lines</think>\nafter"
    out = _sanitize_assistant_content(s)
    assert "thinking" not in out
    assert "I am thinking" not in out
    assert out.startswith("before")
    assert out.endswith("after")


def test_sanitizer_strips_thinking_tag_too():
    from harness.agent_loop import _sanitize_assistant_content
    s = "<thinking>secret</thinking>visible"
    out = _sanitize_assistant_content(s)
    assert out == "visible"


def test_sanitizer_strips_multiple_blocks_case_insensitive():
    from harness.agent_loop import _sanitize_assistant_content
    s = "<Think>one</Think>middle<THINK>two</THINK>end"
    out = _sanitize_assistant_content(s)
    assert "one" not in out
    assert "two" not in out
    assert "middle" in out
    assert "end" in out


def test_sanitizer_returns_none_when_only_thinking():
    from harness.agent_loop import _sanitize_assistant_content
    s = "<think>everything was thinking</think>"
    out = _sanitize_assistant_content(s)
    assert out is None


def test_sanitizer_caps_oversize_content():
    from harness.agent_loop import _sanitize_assistant_content
    s = "x" * 10000
    out = _sanitize_assistant_content(s, max_chars=200)
    assert len(out) <= 300  # cap + truncation marker
    assert "content truncated" in out


def test_sanitizer_passes_through_short_clean_content():
    from harness.agent_loop import _sanitize_assistant_content
    s = "I will call read_code next."
    assert _sanitize_assistant_content(s) == s


def test_tools_for_phase_withholds_write_solution_in_locate():
    """LOCATE phase tool list must exclude the candidate-modifying tools
    (write_solution and apply_edit); FIX phase keeps them."""
    from harness.agent_loop import _tools_for_phase
    locate = {s["function"]["name"] for s in _tools_for_phase(None, in_locate_phase=True)}
    fix = {s["function"]["name"] for s in _tools_for_phase(None, in_locate_phase=False)}
    assert "write_solution" not in locate
    assert "apply_edit" not in locate
    assert "locate_bug" in locate          # locate_bug IS available in locate phase
    assert "read_code" in locate
    assert "write_solution" in fix
    assert "apply_edit" in fix
    assert "locate_bug" in fix
    # The difference between the two phases is exactly the candidate-modifying tools.
    assert fix - locate == {"write_solution", "apply_edit"}


def test_full_file_mode_filters_apply_edit_and_requires_write_solution_code():
    """Production full-file mode exposes only full-file write_solution edits."""
    from harness.agent_loop import _tools_for_phase

    tools = _tools_for_phase(
        ["locate_bug", "apply_edit", "write_solution"],
        in_locate_phase=False,
        require_full_solution_code=True,
    )
    by_name = {s["function"]["name"]: s for s in tools}
    assert "apply_edit" not in by_name
    assert "write_solution" in by_name
    assert by_name["write_solution"]["function"]["parameters"]["required"] == ["code"]


def test_full_file_mode_rejects_patch_submit_and_accepts_complete_code(stubbed_loop):
    """Default production mode requires write_solution(code=full_file)."""
    stubbed_loop["config"]["agent"]["full_file_submissions"] = True
    responses = [
        _resp(tool_calls=[_locate(1)]),
        _resp(tool_calls=[
            _tc("apply_edit", {"old_string": "seed_broken_v0", "new_string": "bad"}),
        ]),
        _resp(tool_calls=[_tc("write_solution", {"rationale": "missing code"})]),
        _resp(tool_calls=[
            _tc("write_solution", {
                "rationale": "complete replacement",
                "code": "ok_code\n",
            }),
        ]),
    ]
    client = MockClient(responses)
    stubbed_loop["compile_results"].append((True, "ok"))
    stubbed_loop["test_results"].append((0, "all checks passed"))

    from harness.agent_loop import run_task_agent
    records, stop = run_task_agent(
        task_stem="CUDA_42",
        dataset_dir=stubbed_loop["dataset"],
        output_dir=stubbed_loop["output_dir"],
        config=stubbed_loop["config"],
        client=client,
        broken_start=stubbed_loop["broken_start"],
    )

    assert stop == "passed"
    assert len(records) == 1
    assert records[0]["iteration"] == 1
    assert stubbed_loop["compile_results"] == []
    assert stubbed_loop["test_results"] == []


def test_agent_apply_edit_then_patch_submit(stubbed_loop):
    """Patch workflow end-to-end (debug mode). The model patches the
    seeded broken_start via apply_edit, then write_solution() commits the
    in-memory candidate for build+test."""
    seed = "seed_broken_v0\n"
    # Override the fixture's broken_start with one whose code matches what
    # the test patches against.
    seed_file = stubbed_loop["broken_start"]["code_path"]
    Path(seed_file).write_text("int x = 0;\nint y = 0;\n")
    responses = [
        # iter 1 fails: patch x → 1, but test still fails
        _resp(tool_calls=[
            _locate(1),
            _tc("apply_edit", {"old_string": "int x = 0;", "new_string": "int x = 1;"}),
            _tc("write_solution", {"rationale": "first try"}),
        ]),
        # iter 2 passes: patch x → 42
        _resp(tool_calls=[
            _locate(1),
            _tc("apply_edit", {"old_string": "int x = 1;", "new_string": "int x = 42;"}),
            _tc("write_solution", {"rationale": "fixed x init"}),
        ]),
    ]
    client = MockClient(responses)
    stubbed_loop["compile_results"].append((True, "ok"))
    stubbed_loop["compile_results"].append((True, "ok"))
    stubbed_loop["test_results"].append((1, "wrong"))
    stubbed_loop["test_results"].append((0, "all passed"))

    from harness.agent_loop import run_task_agent
    records, stop = run_task_agent(
        task_stem="CUDA_42",
        dataset_dir=stubbed_loop["dataset"],
        output_dir=stubbed_loop["output_dir"],
        config=stubbed_loop["config"],
        client=client,
        broken_start=stubbed_loop["broken_start"],
    )
    assert stop == "passed"
    assert len(records) == 2
    assert records[1]["status"] == "passed"
    # iter 2 saved code reflects the apply_edit (int x = 42).
    code_path = stubbed_loop["output_dir"] / records[1]["paths"]["error_code"]
    assert "int x = 42;" in code_path.read_text()
    assert "int x = 0;" not in code_path.read_text()


def test_agent_write_rejected_without_locate(stubbed_loop):
    """Backstop: if a backend ignores the phase-filtered tool list and
    submits without locate_bug, the loop rejects it. The MockClient
    ignores tool filtering so we can drive this directly."""
    responses = [
        # turn 1 — apply_edit + submit WITHOUT locate_bug → locate gate rejects
        _resp(tool_calls=[
            _tc("apply_edit", {"old_string": "seed_broken_v0", "new_string": "v1-attempt"}),
            _tc("write_solution", {}),
        ]),
        # turn 2 — locate first, then patch + submit → accepted
        _resp(tool_calls=[
            _locate(4),
            _tc("apply_edit", {"old_string": "v1-attempt", "new_string": "v1-located"}),
            _tc("write_solution", {}),
        ]),
    ]
    client = MockClient(responses)
    stubbed_loop["compile_results"].append((True, "ok"))      # v1-located
    stubbed_loop["test_results"].append((0, "all checks passed"))

    from harness.agent_loop import run_task_agent
    records, stop = run_task_agent(
        task_stem="CUDA_42",
        dataset_dir=stubbed_loop["dataset"],
        output_dir=stubbed_loop["output_dir"],
        config=stubbed_loop["config"],
        client=client,
        broken_start=stubbed_loop["broken_start"],
    )
    assert stop == "passed"
    # Exactly 1 iteration — the un-located write was rejected, only the
    # second (located) submission advanced the iteration counter.
    assert len(records) == 1
    assert records[0]["iteration"] == 1
    assert records[0]["located"]["line"] == 4
    # The rejection was surfaced to the model.
    trace = stubbed_loop["output_dir"] / "tool_trace" / "CUDA_42.jsonl"
    tools = [json.loads(line)["tool"] for line in trace.read_text().strip().splitlines()]
    # write_solution appears 2× in the trace (one rejected, one accepted).
    assert tools.count("write_solution") == 2
    assert tools.count("locate_bug") == 1
