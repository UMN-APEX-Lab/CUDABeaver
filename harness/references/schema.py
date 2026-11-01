"""Reference JSON schema for performance evaluation."""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from typing import Literal, Optional


class ValidationError(ValueError):
    pass


_TIMING_MODES = {
    "region",                     # Legacy alias; executed via cuda_event_stdout
    "kernels",                    # Legacy alias; executed via cuda_event_stdout
    "cuda_event_stdout",          # Direct cudaEvent timing printed by the benchmark
    "process",                    # Process wall clock (curated_cuda_pool, unused in our 166)
    "kernelbench_eval_result",    # kernelbench eval_kernel_against_ref result dict
    "total_elapsed",              # Generic wall-clock fallback (TK)
    "applied_kernels_make_speedup",
    "thunderkittens_stdout",
}
_REF_TYPES = {"file", "kernelbench_problem"}


@dataclass(frozen=True)
class ReferenceSolution:
    type: Literal["file", "kernelbench_problem"]
    files: Optional[list] = None
    problem_path: Optional[str] = None
    problem_content: Optional[str] = None  # embedded snapshot for self-containment

    @classmethod
    def from_dict(cls, d: dict) -> "ReferenceSolution":
        t = d.get("type")
        if t not in _REF_TYPES:
            raise ValidationError(
                f"reference_solution.type must be one of {_REF_TYPES}, got {t!r}"
            )
        if t == "file":
            files = d.get("files")
            if not isinstance(files, list) or not files:
                raise ValidationError(
                    "reference_solution.files must be a non-empty list when type=file"
                )
            for f in files:
                if not isinstance(f, dict) or "path" not in f or "content" not in f:
                    raise ValidationError("each file must have path and content")
            return cls(type="file", files=files)
        pp = d.get("problem_path")
        pc = d.get("problem_content")
        if not isinstance(pp, str) or not pp:
            raise ValidationError(
                "reference_solution.problem_path required when type=kernelbench_problem"
            )
        if pc is not None and not isinstance(pc, str):
            raise ValidationError("reference_solution.problem_content must be str if present")
        return cls(type="kernelbench_problem", problem_path=pp, problem_content=pc)


@dataclass(frozen=True)
class TimingMode:
    type: Literal[
        "region",
        "kernels",
        "cuda_event_stdout",
        "process",
        "kernelbench_eval_result",
        "total_elapsed",
        "applied_kernels_make_speedup",
        "thunderkittens_stdout",
    ]
    include: Optional[list] = None
    time_type: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "TimingMode":
        t = d.get("type")
        if t not in _TIMING_MODES:
            raise ValidationError(
                f"timing_mode.type must be one of {_TIMING_MODES}, got {t!r}"
            )
        return cls(type=t, include=d.get("include"), time_type=d.get("time_type"))


@dataclass(frozen=True)
class Reference:
    task_id: str
    reference_solution: ReferenceSolution
    benchmark_command: str
    timing_mode: TimingMode
    n_trials: int
    warmup_trials: int
    requires_arch: Optional[str]

    @classmethod
    def from_dict(cls, d: dict) -> "Reference":
        for field_name in (
            "task_id",
            "reference_solution",
            "benchmark_command",
            "timing_mode",
        ):
            if field_name not in d:
                raise ValidationError(f"missing required field: {field_name}")
        return cls(
            task_id=str(d["task_id"]),
            reference_solution=ReferenceSolution.from_dict(d["reference_solution"]),
            benchmark_command=str(d["benchmark_command"]),
            timing_mode=TimingMode.from_dict(d["timing_mode"]),
            n_trials=int(d.get("n_trials", 100)),
            warmup_trials=int(d.get("warmup_trials", 3)),
            requires_arch=d.get("requires_arch"),
        )

    def canonical_hash(self) -> str:
        """SHA256 of the reference_solution only — invariant to trial counts.

        For kernelbench_problem: hashes only problem_path, NOT problem_content.
        Rationale: problem_content is a snapshot for self-containment; we don't
        want adding/refreshing it to invalidate the (expensive) reference cache.
        Upstream-content-aware invalidation would need a separate mechanism.
        """
        sol = self.reference_solution
        if sol.type == "file":
            payload = {
                "type": "file",
                "files": sorted(sol.files, key=lambda f: f["path"]),
            }
        else:
            payload = {
                "type": "kernelbench_problem",
                "problem_path": sol.problem_path,
            }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()

    def timing_fingerprint(self) -> str:
        """Hash timing configuration so stale reference timing caches are rejected."""
        payload = {
            "benchmark_command": self.benchmark_command,
            "timing_mode": {
                "type": self.timing_mode.type,
                "include": self.timing_mode.include,
                "time_type": self.timing_mode.time_type,
            },
            "n_trials": self.n_trials,
            "warmup_trials": self.warmup_trials,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()
