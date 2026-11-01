import json
from pathlib import Path

from harness import run_experiment


def _make_dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset"
    (dataset / "input").mkdir(parents=True)
    (dataset / "prompts").mkdir()
    (dataset / "testbench").mkdir()
    (dataset / "input" / "cuda_1.json").write_text(
        json.dumps({
            "task_id": "cuda_1",
            "solution_file": "solution.cu",
            "difficulty": "unit",
            "level": "unit",
        }),
        encoding="utf-8",
    )
    (dataset / "prompts" / "cuda_1.txt").write_text("Write a CUDA kernel.", encoding="utf-8")
    return dataset


def _config(tmp_path: Path) -> dict:
    return {
        "api": {},
        "experiment": {
            "method": "iterative",
            "max_iterations": 1,
            "max_history_rounds": 1,
            "measure_perf_on_pass": True,
            "performance_gate": {
                "enabled": True,
                "min_speedup": 1.0,
                "fail_on_skipped": True,
            },
        },
        "build": {
            "workdir": str(tmp_path / "work"),
            "timeout": 1,
            "test_timeout": 1,
        },
        "logging": {"save_api_responses": False},
    }


def test_stdout_perf_unmeasured_uses_reference_perf_before_failing(tmp_path, monkeypatch):
    dataset = _make_dataset(tmp_path)
    output = tmp_path / "out"

    monkeypatch.setattr(
        run_experiment,
        "call_llm",
        lambda *a, **k: "__global__ void kernel() {}\n",
    )
    monkeypatch.setattr(
        run_experiment,
        "compile_code",
        lambda *a, **k: (True, ""),
    )
    # Correctness-only CUDA/CUDABench-style output: no Speedup/runtime fields.
    monkeypatch.setattr(
        run_experiment,
        "run_test",
        lambda *a, **k: (0, "All tests passed"),
    )
    monkeypatch.setattr(
        run_experiment,
        "_measure_iter_perf",
        lambda *a, **k: {
            "speedup": 1.2,
            "candidate_runtime_ms": 1.0,
            "reference_runtime_ms": 1.2,
            "timing_method": "region",
        },
    )

    records, stop_reason = run_experiment.run_task(
        "cuda_1",
        dataset,
        output,
        _config(tmp_path),
        client=object(),
    )

    assert stop_reason == "passed"
    assert len(records) == 1
    assert records[0]["status"] == "passed"
    assert records[0]["category"] == "passed"
    assert records[0]["performance"]["speedup"] == 1.2
    assert records[0]["performance_gate"]["passed"] is True


def test_reference_perf_skip_remains_perf_unmeasured(tmp_path, monkeypatch):
    dataset = _make_dataset(tmp_path)
    output = tmp_path / "out"

    monkeypatch.setattr(run_experiment, "call_llm", lambda *a, **k: "__global__ void kernel() {}\n")
    monkeypatch.setattr(run_experiment, "compile_code", lambda *a, **k: (True, ""))
    monkeypatch.setattr(run_experiment, "run_test", lambda *a, **k: (0, "All tests passed"))
    monkeypatch.setattr(
        run_experiment,
        "_measure_iter_perf",
        lambda *a, **k: {"skipped": "reference_not_cached"},
    )

    records, stop_reason = run_experiment.run_task(
        "cuda_1",
        dataset,
        output,
        _config(tmp_path),
        client=object(),
    )

    assert stop_reason == "max_iterations"
    assert len(records) == 1
    assert records[0]["status"] == "failed"
    assert records[0]["category"] == "perf_unmeasured"
    assert records[0]["performance_gate"]["passed"] is False
    assert records[0]["performance_gate"]["reason"] == "skipped:reference_not_cached"
