"""Context manager coverage for native-free JSON action transcripts."""

from harness.context_manager import (
    ACTION_RESULT_MARKER,
    ContextConfig,
    estimate_tokens,
    get_budget_nudge,
    manage_context,
)


def _action(tool: str, args: dict | None = None) -> dict:
    import json

    return {
        "role": "assistant",
        "content": json.dumps({"tool": tool, "args": args or {}}),
    }


def _result(tool: str, body: str, ok: bool = True) -> dict:
    return {
        "role": "user",
        "content": f"{ACTION_RESULT_MARKER}\ntool={tool}\nok={str(ok).lower()}\n\n{body}",
    }


def test_manage_context_compacts_json_action_transcript():
    large = "CUDA line\n" * 2500
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "initial task"},
        _action("read_code", {"start": 1, "end": 400}),
        _result("read_code", large),
        _action("write_solution", {}),
        _result("write_solution", "[BUILD/TEST RESULT]\nbuild_ok=False | category=compile_error\nfailed"),
        _action("read_code", {"start": 1, "end": 20}),
        _result("read_code", "fresh code"),
        _action("write_solution", {}),
        _result("write_solution", "[BUILD/TEST RESULT]\nbuild_ok=True | category=passed\nok"),
    ]
    cfg = ContextConfig(
        tool_result_cap_chars=1000,
        keep_recent_tool_exchanges=1,
        compact_old_iterations=True,
        chars_per_token=1.0,
        summary_target_chars=500,
    )

    before = estimate_tokens(messages, cfg.chars_per_token)
    stats = manage_context(messages, cfg, summarizer=None)

    assert stats.tokens_after < before
    assert any(layer.startswith("L1:cap") for layer in stats.layers_applied)
    assert any(layer.startswith("L3:old") for layer in stats.layers_applied)
    assert any(
        m["role"] == "user" and m["content"].startswith("[Prev iteration summary]")
        for m in messages
    )


def test_force_budget_nudge_is_locate_phase_aware():
    nudge = get_budget_nudge("force", in_locate_phase=True)
    assert "locate_bug" in nudge
    assert "write_solution` on the next turn" not in nudge
