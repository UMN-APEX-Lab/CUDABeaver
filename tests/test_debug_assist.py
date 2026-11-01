"""Tests for optional debug-assist helpers."""

from harness.debug_assist import analyze_patch_scope


def _cfg():
    return {
        "debug_tools": {
            "enabled": True,
            "patch_scope_guard": {"enabled": True},
        }
    }


def test_patch_scope_accepts_agent_locate_bug_shape():
    baseline = "\n".join(f"line {i}" for i in range(1, 13))
    lines = baseline.splitlines()
    lines[1] = "line 2 fixed"
    candidate = "\n".join(lines)

    result = analyze_patch_scope(
        _cfg(),
        baseline,
        candidate,
        {"file": "solution.cu", "line": 2, "bug_description": "wrong index"},
    )

    assert result["suspicious_lines"] == [2]
    assert result["overlap_lines"] == [2]
    assert result["missed_suspicious_lines"] is False


def test_patch_scope_flags_missed_agent_located_line():
    baseline = "\n".join(f"line {i}" for i in range(1, 13))
    lines = baseline.splitlines()
    lines[9] = "line 10 changed"
    candidate = "\n".join(lines)

    result = analyze_patch_scope(
        _cfg(),
        baseline,
        candidate,
        {"file": "solution.cu", "line": 2, "bug_description": "wrong index"},
    )

    assert result["suspicious_lines"] == [2]
    assert result["overlap_lines"] == []
    assert result["missed_suspicious_lines"] is True
