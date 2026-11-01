"""Parse direct cudaEvent timing emitted by benchmark executables."""
from __future__ import annotations

import re


def _find_float(text: str, names: tuple[str, ...]) -> float | None:
    for name in names:
        m = re.search(rf"\b{name}\s*=\s*([0-9.eE+-]+)", text)
        if m:
            return float(m.group(1))
    return None


def _find_int(text: str, names: tuple[str, ...]) -> int | None:
    value = _find_float(text, names)
    if value is None:
        return None
    return max(1, int(round(value)))


class CudaEventStdout:
    def wrap_command(self, cmd: str, workdir: str) -> str:
        return cmd

    def parse(self, stdout: str, stderr: str) -> dict:
        text = stdout + "\n" + stderr
        mean_ms = _find_float(
            text,
            (
                "CUDA_EVENT_MEAN_MS",
                "CUDA_EVENT_AVG_MS",
                "CUDA_EVENT_ELAPSED_MS",
                "CUDA_EVENT_TIME_MS",
            ),
        )
        total_ms = _find_float(text, ("CUDA_EVENT_TOTAL_MS",))
        n_instances = _find_int(text, ("CUDA_EVENT_N_ITERS", "CUDA_EVENT_N_TRIALS")) or 1

        if mean_ms is None and total_ms is not None:
            mean_ms = total_ms / n_instances
        if total_ms is None and mean_ms is not None:
            total_ms = mean_ms * n_instances
        if mean_ms is None:
            raise ValueError(
                "cudaEvent timing output not found. Expected CUDA_EVENT_MEAN_MS=... "
                f"in stdout/stderr tail:\n{text[-500:]}"
            )

        return {
            "method": "cuda_event_stdout",
            "mean_ms": mean_ms,
            "p50_ms": mean_ms,
            "p95_ms": mean_ms,
            "stddev_ms": 0.0,
            "raw_samples_ms": [],
            "n_instances": n_instances,
            "total_ms": total_ms,
        }
