"""applied-kernels `make test` stdout parser.

FrontierKernel's `./test` binary embeds BOTH the original CUTLASS reference
and the LLM solution kernels, runs each, and prints:

    Kernel time: X.YYY ms
    Ref time:    A.BBB ms
    Speedup:     Z.ZZ

  - `Kernel time` = the candidate (LLM solution) kernel runtime
  - `Ref time`    = the original CUTLASS / TK / flash-attn reference
  - `Speedup`     = Ref/Kernel ratio (applied-kernels's internal computation)

Our framework's standard speedup is also Ref/Cand, so applied-kernels's internal speedup
should match what compute_speedup(ref_mean, cand_mean) gives later.

We expose:
  - mean_ms = Kernel time  (candidate runtime, used by run_candidate)
  - kh_internal_ref_ms     (applied-kernels's reported ref time, kept for cross-check)
  - kh_internal_speedup    (applied-kernels's reported speedup, kept for cross-check)

When invoked as the REFERENCE run (placeholder solution.h naive baseline),
mean_ms still equals "Kernel time" — i.e. the naive kernel's runtime, which
serves as our reference baseline for speedup calculation against the LLM.
"""
import re


class AppliedKernelsMakeSpeedup:
    def wrap_command(self, cmd: str, workdir: str) -> str:
        # applied-kernels binary handles its own measurement; no external profiler wrap
        return cmd

    def parse(self, stdout: str, stderr: str) -> dict:
        text = (stdout or "") + "\n" + (stderr or "")
        m_kernel = re.search(r"Kernel time\s*[:=]?\s*([\d.]+)\s*ms", text, re.IGNORECASE)
        m_ref = re.search(r"Ref time\s*[:=]?\s*([\d.]+)\s*ms", text, re.IGNORECASE)
        if not m_kernel and not m_ref:
            raise ValueError(
                f"applied_kernels_make_speedup: no 'Kernel time:' or 'Ref time:' in output. "
                f"Last 500 chars:\n{text[-500:]}"
            )
        kernel_ms = float((m_kernel or m_ref).group(1))
        result = {
            "method": "applied_kernels_make_speedup",
            "mean_ms": kernel_ms,
            "n_trials": 1,
        }
        if m_ref:
            result["kh_internal_ref_ms"] = float(m_ref.group(1))
        m_speedup = re.search(r"Speedup\s*[:=]?\s*([\d.]+)", text, re.IGNORECASE)
        if m_speedup:
            result["kh_internal_speedup"] = float(m_speedup.group(1))
        return result
