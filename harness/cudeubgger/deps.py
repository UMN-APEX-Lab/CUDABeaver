"""Patchable external dependencies for Cudeubgger.

Tests and the legacy ``harness.agent_loop`` facade can replace these attributes
without reaching into every implementation module. Runtime code accesses this
module dynamically instead of binding long-lived function aliases.
"""

from ..classifier import classify_result
from ..compiler import compile_code, run_sanitizers, run_test
from ..debug_assist import (
    analyze_patch_scope,
    append_debug_feedback,
    debug_tools_enabled,
    format_precheck_prompt_section,
    run_precheck,
    section_enabled,
    verify_failed_patch,
)
from ..feedback import build_feedback
from ..llm_client import (
    call_llm,
    call_llm_json_action,
    call_llm_with_tools,
    is_context_length_error,
)
from ..performance import evaluate_performance_gate, measure_iter_perf
from ..stagnation import check_stagnation, make_record as make_stagnation_record
from ..trace_io import compact_sanitizer_summary
