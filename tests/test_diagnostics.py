import subprocess

from harness import diagnostics


def test_compute_sanitizer_supports_wrapped_python(monkeypatch, tmp_path):
    calls = []

    def fake_run_pg(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout="========= COMPUTE-SANITIZER\n========= ERROR SUMMARY: 0 errors\n",
            stderr="",
        )

    monkeypatch.setattr(diagnostics, "sanitizer_available", lambda: True)
    monkeypatch.setattr(diagnostics, "run_pg", fake_run_pg)

    result = diagnostics.run_compute_sanitizer(
        "python -c \"print('uses torch later')\"",
        tmp_path,
        tool="memcheck",
        timeout=7,
    )

    assert result.ok
    cmd, kwargs = calls[0]
    assert "--target-processes=all" in cmd
    assert "--require-cuda-init=no" in cmd
    assert "python -c" in cmd
    assert kwargs["timeout"] == 7
    assert result.meta["wrapped_command"] is True


def test_compute_sanitizer_renders_tool_specific_options(monkeypatch, tmp_path):
    calls = []

    def fake_run_pg(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(
            cmd,
            99,
            stdout="========= Race reported\n========= ERROR SUMMARY: 1 error\n",
            stderr="",
        )

    monkeypatch.setattr(diagnostics, "sanitizer_available", lambda: True)
    monkeypatch.setattr(diagnostics, "run_pg", fake_run_pg)

    result = diagnostics.run_compute_sanitizer(
        "./test.out",
        tmp_path,
        tool="racecheck",
        options={
            "target_processes_filter": "regex:python.*|test.out",
            "kernel_name": "kernel_substring=my_kernel",
            "kernel_name_exclude": "kernel_substring=warmup",
            "launch_skip": 2,
            "launch_count": 3,
            "force_blocking_launches": True,
            "show_backtrace": "device",
            "racecheck_detect_level": "warn",
            "racecheck_report": "hazard",
            "racecheck_trace_sync": True,
            "racecheck_deadlock_timeout": 1000,
        },
    )

    assert result.ok
    cmd = calls[0]
    assert "--tool=racecheck" in cmd
    assert "--target-processes-filter=regex:python.*|test.out" in cmd
    assert "--kernel-name=kernel_substring=my_kernel" in cmd
    assert "--kernel-name-exclude=kernel_substring=warmup" in cmd
    assert "--launch-skip=2" in cmd
    assert "--launch-count=3" in cmd
    assert "--force-blocking-launches" in cmd
    assert "--show-backtrace=device" in cmd
    assert "--racecheck-detect-level=warn" in cmd
    assert "--racecheck-report=hazard" in cmd
    assert "--racecheck-trace-sync=yes" in cmd
    assert "--racecheck-deadlock-timeout=1000" in cmd
    assert result.meta["options"]["racecheck_report"] == "hazard"


def test_compute_sanitizer_rejects_unknown_option(monkeypatch, tmp_path):
    monkeypatch.setattr(diagnostics, "sanitizer_available", lambda: True)

    result = diagnostics.run_compute_sanitizer(
        "./test.out",
        tmp_path,
        tool="memcheck",
        options={"not_a_real_option": True},
    )

    assert not result.ok
    assert "unsupported compute-sanitizer option" in (result.error or "")

