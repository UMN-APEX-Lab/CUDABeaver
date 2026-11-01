"""Parse ThunderKittens benchmark stdout."""
from __future__ import annotations

import re
import statistics


def _float_ms(value: str, unit: str) -> float:
    x = float(value)
    unit = unit.lower()
    if unit in {"us", "µs"}:
        return x / 1000.0
    if unit == "s" or unit.startswith("second"):
        return x * 1000.0
    return x


class ThunderKittensStdout:
    def wrap_command(self, cmd: str, workdir: str) -> str:
        return cmd

    def parse(self, stdout: str, stderr: str) -> dict:
        text = (stdout or "") + "\n" + (stderr or "")
        samples: list[float] = []

        for m in re.finditer(
            r"ThunderKittens:\s*([0-9.eE+-]+)\s*(ms|us|µs|seconds?|s)\b",
            text,
            flags=re.IGNORECASE,
        ):
            samples.append(_float_ms(m.group(1), m.group(2)))

        for m in re.finditer(
            r"Method\s+(?P<name>[A-Za-z0-9_./-]+).*?"
            r"Time:\s*(?P<value>[0-9.eE+-]+)\s*(?P<unit>ms|us|µs|seconds?|s)\b",
            text,
            flags=re.IGNORECASE,
        ):
            name = m.group("name").lower()
            if "tk" in name or "thunder" in name:
                samples.append(_float_ms(m.group("value"), m.group("unit")))

        if not samples:
            for m in re.finditer(
                r"Average execution time:\s*([0-9.eE+-]+)\s*(ms|us|µs|seconds?|s)\b",
                text,
                flags=re.IGNORECASE,
            ):
                samples.append(_float_ms(m.group(1), m.group(2)))

        if not samples:
            for m in re.finditer(
                r"Kernel execution time:\s*([0-9.eE+-]+)\s*(ms|us|µs|seconds?|s)\b",
                text,
                flags=re.IGNORECASE,
            ):
                samples.append(_float_ms(m.group(1), m.group(2)))

        samples = [x for x in samples if x > 0]
        if not samples:
            raise ValueError(
                "thunderkittens_stdout: no timing output found. "
                f"Last 800 chars:\n{text[-800:]}"
            )

        mean_ms = statistics.geometric_mean(samples)
        return {
            "method": "thunderkittens_stdout",
            "mean_ms": mean_ms,
            "p50_ms": statistics.median(samples),
            "stddev_ms": statistics.stdev(samples) if len(samples) >= 2 else 0.0,
            "raw_samples_ms": samples,
            "n_instances": len(samples),
        }
