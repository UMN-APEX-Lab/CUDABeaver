"""Compatibility facade for the Cudeubgger agent framework.

The implementation moved to :mod:`harness.cudeubgger` to keep the agent code
maintainable. Existing callers may continue importing ``harness.agent_loop``.
"""

from __future__ import annotations

from .cudeubgger import common as _common
from .cudeubgger import deps as _deps
from .cudeubgger import prompts as _prompts
from .cudeubgger.blackboard import (
    run_task_blackboard_agent as _run_task_blackboard_agent,
    run_task_multi_role_no_blackboard_agent as _run_task_multi_role_no_blackboard_agent,
)
from .cudeubgger.common import *  # re-export legacy private helpers used by tests/tools
from .cudeubgger.react import run_task_agent as _run_task_agent
from .cudeubgger.runtime import (
    _auto_submit_after_apply_edit,
    _build_tool_env,
    _call_json_action_or_record_api_error,
    _call_llm_or_record_api_error,
    _dispatch_tool_calls,
    _handle_locate_bug,
    _handle_no_op_turn,
    _handle_submission,
    _make_iter_record,
    _persist_iter_artifacts,
    _pre_turn_context_mgmt,
    _seed_broken_start,
    _write_tool_trace,
)
from .cudeubgger.prompts import *

# `import *` intentionally skips underscore-prefixed names. The old
# `harness.agent_loop` exposed a number of private helpers that tests and
# notebooks import directly, so copy them into this facade explicitly.
globals().update({
    name: value
    for name, value in vars(_common).items()
    if not name.startswith("__")
})
globals().update({
    name: value
    for name, value in vars(_prompts).items()
    if not name.startswith("__")
})

# Patchable dependency aliases preserved for old tests and ad-hoc harness code.
compile_code = _deps.compile_code
run_test = _deps.run_test
run_sanitizers = _deps.run_sanitizers
classify_result = _deps.classify_result
measure_iter_perf = _deps.measure_iter_perf
evaluate_performance_gate = _deps.evaluate_performance_gate
verify_failed_patch = _deps.verify_failed_patch


def _sync_legacy_dependency_aliases() -> None:
    _deps.compile_code = compile_code
    _deps.run_test = run_test
    _deps.run_sanitizers = run_sanitizers
    _deps.classify_result = classify_result
    _deps.measure_iter_perf = measure_iter_perf
    _deps.evaluate_performance_gate = evaluate_performance_gate
    _deps.verify_failed_patch = verify_failed_patch


def run_task_blackboard_agent(*args, **kwargs):
    _sync_legacy_dependency_aliases()
    return _run_task_blackboard_agent(*args, **kwargs)


def run_task_multi_role_no_blackboard_agent(*args, **kwargs):
    _sync_legacy_dependency_aliases()
    return _run_task_multi_role_no_blackboard_agent(*args, **kwargs)


def run_task_agent(*args, **kwargs):
    _sync_legacy_dependency_aliases()
    return _run_task_agent(*args, **kwargs)
