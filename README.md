# CUDABeaver (NeurIPS 2026)

<p align="center">
  <img src="assets/repair.png" alt="CUDABeaver: LLM-driven CUDA kernel repair pipeline" width="100%">
</p>

A benchmark of **199 broken CUDA kernels** paired with reference
implementations and runnable testbenches, designed to evaluate large
language model (LLM) debugging capability on GPU kernel code.

This single repository ships **everything** end-to-end:
- the 199-instance dataset under `data/` (broken_start + reference + testbench + per-instance metadata),
- the evaluation harness (`harness/`) that drives the gen → build → test → score loop,
- the analysis pipeline (`analysis/`) that produces every paper table,
- 274 evaluation configs (`configs/`) covering 5 axes (iter / repeated / feedback / history / fewshot) plus the tool-calling agent (`agent/`),
- bundled CUTLASS + ThunderKittens headers (`data/_external/`) and a vendored copy of upstream KernelBench (`harness/_vendor/kernelbench/`) so no external clones are needed.

---


## Quick start

```bash
git clone <this-repo> cuda-debugger-bench
cd cuda-debugger-bench

# (A) pip + venv
bash setup_env.sh && source .venv/bin/activate
# OR (B) conda
# conda env create -f environment.yml && conda activate cuda-debugger-bench && bash scripts/post_conda_setup.sh

bash scripts/prepare_benchmark.sh
pytest tests/test_smoke.py -v
```

Then point a cfg's `api:` block at your cloud-API or vLLM endpoint and run

```bash
python -m harness.run_experiment --config configs/iter/<model>.yaml \
    --mode debug --v2-task-list benchmark/task_list.json
```

(see [Reproducing paper tables](#reproducing-paper-tables)).

---

## Repository layout

```
harness/                   evaluation engine
  run_experiment.py          main entrypoint (gen → build → test → score)
  compiler.py                build dispatcher (nvcc / make / inline-cuda / class 2 runner)
  classifier.py              5-category outcome classifier
  feedback.py                Exp 3: feedback-level dispatcher (L0 … L4, error-aware modes)
  few_shot.py                Exp 7: in-context debug-pair retrieval
  benchmark.py               perf measurement orchestrator
  performance.py             speedup gate (candidate vs cached reference baseline)
  timing_modes/              timing backends (cuda_event_stdout, elapsed, applied-kernels, kernelbench)
  task_registry.py           resolves full instance ids via benchmark/task_list.json
  applied_kernels_eval.py    adapter for Class 2 def.py-based tasks
  applied_kernels_runner/    vendored class-1/2 backend (runner, eval, compiler, …)
  _vendor/kernelbench/       upstream KernelBench (MIT, ScalingIntelligence/KernelBench)
  llm_client.py              OpenAI-compatible cloud-API + vLLM client
  parallel_runner.py         per-task parallelism (N workers across N GPUs)
  queue_daemon.py            crash-safe queue daemon for long-batch sweeps
  references/schema.py       task / instance schema definitions
  agent_loop.py              method=agent entry: ReAct / blackboard tool-calling debugger
  agent_tools.py             agent tool registry (read / analyze / sanitize / build / test / edit)
  cudeubgger/                agent workflows (react.py, blackboard.py) + trace records
  context_manager.py         agent token budget, nudges, history compaction
  context_summary.py         LLM-backed summarisation of oversized tool results
  debug_assist.py            precheck, patch-scope analysis, failed-patch verification
  diagnostics.py             compute-sanitizer wrapper (normalized, never raises)

analysis/                  result aggregation
  scanner.py                 walks outputs/ and emits Run records
  pool.py                    per-task pooled metrics + 5-category counts
  axis_normalize.py          cfg basename → (model, axis, experiment) parser
  build_master.py            CLI: produces master.csv across all runs
  pivot_tables.py            CLI: paper tables + per-model summary + summary.md
  render_latex.py            booktabs LaTeX renderer
  p_sweep.py                 CLI: offline gate re-scoring at 9 thresholds

configs/                   274 evaluation configs
  iter/                      single-shot debug
  repeated/                  repeated@k vs iterative@k
  feedback/                  L0 (none) → L4 (oracle); + L3_raw + error-aware modes
  history/                   H1 … H4 multi-turn history rounds
  fewshot/                   in-context debug-pair retrieval
  agent/                     tool-calling agent (ReAct / blackboard workflows)

scripts/                   one-time setup + utilities
  prepare_benchmark.sh       materialize flat dataset layout (one-time, idempotent)
  post_conda_setup.sh        register vendored kernelbench in conda env (one-time)
  migrate_full_id_keying.py  documented migration to full-instance-id keying

data/                      dataset (199 instances)
  <instance_id>/             per-instance broken_start + reference + testbench + metadata
  _external/                 bundled CUTLASS (BSD-3) + ThunderKittens (MIT) headers

benchmark/                 canonical task list + timing baselines
  task_list.json             199 full-instance-id entries (used by --mode debug)
  references/                144 per-instance reference timing specs
  references_cache/          measured baselines: Blackwell sm_120
docs/                      PROTOCOL, PERF_PROTOCOL, REPRODUCE,
                           DATA_SCHEMA (family semantics), CURATION (funnel)
tests/                     smoke + per-family install fixtures (no LLM key)
examples/                  pointers to reproduction walk-throughs (docs/REPRODUCE.md)

manifest.json              top-level index over the 199 instances
croissant.json             Croissant 1.0 metadata + 22 RAI fields (RAI 1.0)
README.md                  this file
LICENSE                    CC BY 4.0
```

---

## Dataset card

### Dataset summary

The benchmark consists of 199 instances across 11 task families spanning the
breadth of GPU programming workloads — from low-level cuBLAS / cuFFT /
cuSOLVER / cuSPARSE library calls to high-performance attention kernels
(FlashAttention) and CUTLASS GEMM tile schedules. Each
instance pairs an LLM-generated broken implementation with the canonical
reference and a runnable testbench that detects whichever of five failure
modes the bug induces (compile_error, logic_error, memory_crash, perf_broken,
timeout).

The intended use is **single-shot debugging evaluation** — given a broken
starting point and ground-truth context, can a model produce a corrected
implementation that passes the testbench? Multi-iteration variants
(repeated@k, iterative@k, history-augmented, feedback-graded, few-shot) are
described in the companion paper and supported by the released harness.

### Supported tasks

- **Code repair / Single-shot debugging**: Given a broken CUDA / inline-CUDA
  Python source plus a natural-language task description, generate a
  corrected implementation that satisfies the bundled testbench.
- **Iterative debugging**: Multi-turn dialogue variants where each
  iteration sees the prior failure log; supported by the harness's
  `feedback` and `history` axes.
- **Difficulty / family stratification**: Each instance carries a
  `task_family` and `error_category` label, enabling stratified analysis
  by kernel domain or failure mode.

The companion paper introduces the metric `pass@k` (with optional
performance gate `p`), defined as the probability that at least one of
`k` independently sampled solutions passes correctness AND achieves
`speedup ≥ p × reference_speed`.

### Languages

Code: CUDA C++ (`.cu`, `.cuh`, `.h`), Python (`.py`).
Natural-language prompts and task descriptions: English.

### Dataset structure

#### Data instances

One subdirectory per instance, named with the scheme:

```
<task_family_n>__<error_category>__<broken_start_hash8>
```

Examples: `cuda_113__compile_error__81a2bfc5`,
`kernelbench_level1_16__logic_error__117b6454`,
`cutlass_00_basic_gemm__perf_broken__b536032b`.

Each subdirectory contains:

```
data/<instance_id>/
  broken_start.cu      LLM-generated buggy code (or .py / .h depending on family)
  reference.cu         Reference implementation that passes the testbench
  prompt.txt           User-facing task description shown to the LLM
  testbench/           Runnable build + correctness + performance harness
  input.json           build_command, test_command, schema metadata
  instance.json        Per-instance metadata (id, family, category, difficulty)
```

Top-level files:

```
data/_external/        Bundled CUTLASS + ThunderKittens headers (BSD-3 / MIT)
manifest.json          Index over all 199 instances (compact entry per instance)
croissant.json         Croissant 1.0 metadata + 22 RAI fields (RAI 1.0)
LICENSE                CC BY 4.0
```

#### Data fields

`instance.json` schema:

| Field | Type | Description |
|---|---|---|
| `id` | string | Globally unique instance ID (`<family_n>__<category>__<hash8>`) |
| `task_family` | string | One of 12 families (cuda, cudabench, cutlass, cublas, …) |
| `error_category` | string | One of 5 (compile_error, logic_error, memory_crash, perf_broken, timeout) |
| `difficulty` | string | "easy", "medium", or "hard" |
| `upstream_library` | string | Source of the reference implementation (e.g. "NVIDIA cuBLAS", "ThunderKittens") |
| `broken_start_provenance` | string | Provenance of the broken starting code |
| `broken_start_lines` | integer | Line count of the broken source |
| `broken_start_hash` | string | First 16 hex chars of `sha256(broken_start)` |
| `files` | object | Per-file relative paths within the instance directory |

`input.json` schema: contains `build_command`, `test_command`, `solution_file`,
`level`, `tags`, plus a `hardware` block when the testbench has
architecture-specific requirements.

#### Data splits

The dataset is released as a **single split** (199 instances). The companion
paper evaluates each instance under multiple cfgs (iter, repeated, feedback
L0–L4, history H1–H4, fewshot K1–K5); these are evaluation axes, not
splits — every cfg sees the same 199 instances.

#### Composition

| Task family | n | Description |
|---|---|---|
| `cuda` | 55 | Curated standalone nvcc + thrust kernels |
| `cudabench` | 5 | CUDABench-derived matrix workloads (sanitizer + output-comparison oracle) |
| `cublas` | 2 | NVIDIA cuBLAS reference workloads |
| `cufft` | 1 | NVIDIA cuFFT reference workloads |
| `cufft_samples` | 9 | NVIDIA cuFFT sample C2C / R2C kernels |
| `cusolver` | 7 | NVIDIA cuSOLVER reference workloads |
| `cusparse` | 5 | NVIDIA cuSPARSE reference workloads |
| `cutlass` | 20 | CUTLASS GEMM / Conv / fusion examples |
| `flashattention` | 10 | FlashAttention-2 forward / backward / split kernels |
| `layernorm` | 16 | Forward / backward LayerNorm + parallel variants |
| `kernelbench` | 69 | KernelBench level-1 + level-2 PyTorch-extension kernels |

Error category distribution: `compile_error` 76, `logic_error` 45,
`perf_broken` 41, `memory_crash` 32, `timeout` 5.

### Dataset creation

#### Curation rationale

LLMs are increasingly deployed for low-level GPU programming, but standard
code-evaluation benchmarks measure code generation from scratch — not the
practical setting where a developer asks an LLM to fix existing broken code.
This benchmark fills that gap by providing 199 *real* failure modes
(generated by frontier LLMs themselves on GPU kernel tasks) paired with
ground-truth references and runnable testbenches.

#### Source data

**Initial Data Collection and Normalization** —
Broken starting code was collected by sampling the iter-1 outputs of a
committee of frontier LLMs on a curated pool of GPU kernel tasks (April
2026 – May 2026). For each (task, model) pair we kept the first iter-1
output exhibiting one of five target failure modes. Source LLM names are
aggregated into the `broken_start_provenance` field rather than reported
per-instance to avoid attributing specific failures to specific vendors.

Reference implementations are imported unmodified from public open-source
kernels (NVIDIA CUDA samples, CUTLASS examples, FlashAttention,
ThunderKittens, etc.). Each instance's `upstream_library` field records
the canonical source.

---

## Setup

### Option A — pip + venv (recommended)

```bash
bash setup_env.sh
source .venv/bin/activate
bash scripts/prepare_benchmark.sh
```

`setup_env.sh` creates a venv from `requirements.txt` and registers the
vendored `kernelbench` package via a `.pth` so `import kernelbench` works.
`prepare_benchmark.sh` materializes the flat dataset layout the harness
expects (idempotent — re-runnable any time).

### Option B — conda

```bash
conda env create -f environment.yml
conda activate cuda-debugger-bench
bash scripts/post_conda_setup.sh
bash scripts/prepare_benchmark.sh
```

`environment.yml` pulls PyTorch + CUDA from the official `pytorch` and
`nvidia` channels for ABI alignment with the system `nvcc`. Adjust
`pytorch-cuda` if your toolkit version (12.x vs 13.x) differs.

### System prerequisites

- CUDA toolkit **`nvcc` ≥ 12.8** (cusolver tasks use `cusolverDnXgeev`,
  several testbenches include `<nvtx3/nvtx3.hpp>`; CUDA 12.9/13.x tested).
  Export both `PATH` and `CUDA_HOME` — `setup_env.sh` checks this.
- A CUDA-capable GPU (SM ≥ 80; corpus validated on Blackwell sm_120 and
  Hopper sm_90 — task commands auto-adapt to the executing GPU, see
  `docs/DATA_SCHEMA.md` § Architecture adaptation)
- A torch wheel matching your **driver's** CUDA version (see requirements.txt)
- Python 3.10–3.12 (3.11 is the tested target)

---

## Smoke test

```bash
pytest tests/test_smoke.py -v                 # <1 s: imports + classifier
pytest tests/test_family_fixtures.py -v      # ~15-30 min: builds + passes one
                                             # task per family on your GPU
```

No LLM key required for either. The family-fixture suite is the full
install check: green means nvcc, torch+ninja JIT, the vendored kernelbench,
and the applied-kernels make flow all work on this machine. Preflight any
experiment config without spending tokens via:

```bash
python -m harness.run_experiment -c configs/iter/<model>.yaml \
    --mode debug --v2-task-list benchmark/task_list.json --dry-run
```

---

## Reproducing paper tables

See **`docs/REPRODUCE.md`** for the full command sequence per paper table.
Quick sketch (Table 1, iter axis):

```bash
for cfg in configs/iter/*.yaml; do
  python -m harness.run_experiment \
    --config "$cfg" \
    --v2-task-list benchmark/task_list.json \
    --mode debug
done

python -m analysis.build_master  --results-root outputs --out-dir analysis_out
python -m analysis.pivot_tables  --in-dir analysis_out --out-dir analysis_out
```

`analysis_out/tables/exp1_iter.tex` matches paper Table 1.

### Agent mode

Setting `experiment.method: agent` replaces the fixed gen → build → test loop
with a tool-calling debugger (configs under `configs/agent/`). It always starts
from the instance's `broken_start`, so it runs in debug mode only:

```bash
python -m harness.run_experiment -c configs/agent/agent_smoke.yaml \
    --mode debug --v2-task-list benchmark/task_list.json \
    --tasks cuda_110__logic_error__957be815
```

- **Workflow** (`agent.workflow`): `react` — a single agent that must call
  `locate_bug` before `apply_edit` / `write_solution` are unlocked; or
  `blackboard` — analyzer → executor → verifier → coder roles, each limited to
  the tool subset in `agent.blackboard.*_tools`.
- **Tools**: `read_code`, `read_test_file`, `read_prompt`,
  `run_static_analysis`, `run_compute_sanitizer`, `build_only`, `test_only`,
  `locate_bug`, `apply_edit`, `write_solution` (build + test = one patch
  submission). `agent.tools` takes an allow-list for ablations.
- **Budgets**: `max_patch_submissions`, `max_tool_calls_per_task`, per-tool
  `quotas`, and a context `token_budget`; oversized tool results are
  summarised (`use_llm_summary`) or truncated past `tool_result_cap_chars`.
- **Performance gate**: `agent.performance_gate.min_speedup` requires a passing
  patch to also reach that speedup over the cached reference baseline.

The model endpoint must support OpenAI-style tool calling (or set
`action_protocol: json`). Each run writes a per-task `tool_trace/<task>.jsonl`
next to the usual `manifest.jsonl`.

### LLM endpoints — required swap

Each cfg's `api:` block ships with placeholder `<CLOUD_API_BASE_URL>` +
`CLOUD_API_KEY`. Users must point these at their own infrastructure
(any OpenAI-compatible cloud API or a self-hosted vLLM server). See
`docs/REPRODUCE.md § LLM endpoints` for details.

---

## Family-specific runtime requirements

All required code/headers are bundled — most instances build with just
`nvcc`. Per-family detail:

| Family                                  | n  | Bundled at                                          |
|-----------------------------------------|----|-----------------------------------------------------|
| cuda, cudabench, cublas, cufft, cusolver, cusparse | 75 | (nvcc only)                                         |
| cutlass                                 | 20 | `data/_external/cutlass`                            |
| kernelbench                             | 69 | `harness/_vendor/kernelbench/` (registered via .pth)|
| flashattention, layernorm, cufft_samples| 35 | `harness/applied_kernels_runner/` + reference_sources |

**Total: 199 instances**, buildable on CUDA 12.8+ or 13 (see the
build-audit table in `docs/REPRODUCE.md`).

---

## License

The benchmark + harness + analysis code is released under **CC BY 4.0**
(see `LICENSE`).

Bundled third-party packages keep their own licenses:

- `harness/_vendor/kernelbench/` — MIT, see
  `harness/_vendor/kernelbench/LICENSE.txt`
- `data/_external/cutlass/` — BSD 3-Clause License (NVIDIA Corporation),
  see `data/_external/cutlass/LICENSE.txt`
- `data/_external/ThunderKittens/` — MIT License (HazyResearch), see
  `data/_external/ThunderKittens/LICENSE.txt`

---

## Citation

```bibtex
@inproceedings{li2026cudabeaver,
  title     = {{CUDABeaver}: Benchmarking {LLM}-Based Automated {CUDA} Debugging},
  author    = {Li, Shiyang and Chen, Haoyang and Fazzini, Mattia and Ding, Caiwen},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Evaluations and Datasets Track},
  year      = {2026}
}
```
