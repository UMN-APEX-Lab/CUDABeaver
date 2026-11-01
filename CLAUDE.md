# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

CUDABeaver is a benchmark of 199 broken CUDA kernels (NeurIPS 2026 D&B submission). It ships the
dataset (`data/`), the evaluation harness (`harness/`) that runs the gen → build → test → classify
→ feedback loop on an LLM, and the analysis pipeline (`analysis/`) that produces the paper tables.
`README.md` is the public dataset card; this file documents how the code actually fits together,
including the newer agent subsystem that the README does not yet cover.

## Setup & common commands

```bash
bash setup_env.sh && source .venv/bin/activate   # venv from requirements.txt + registers vendored kernelbench via .pth
bash scripts/prepare_benchmark.sh                # REQUIRED before any run — see "Dataset layout" below
pytest tests/test_smoke.py -v                    # no LLM key / no GPU needed; <1s
```

- Conda alternative: `conda env create -f environment.yml && conda activate cuda-debugger-bench && bash scripts/post_conda_setup.sh`.
- Run a single test: `pytest tests/test_agent_tools.py::test_read_code_returns_lines -v`.
- Unit tests that don't need a GPU/key: `test_smoke`, `test_agent_tools`, `test_agent_loop`, `test_diagnostics`, `test_llm_client`. Actually building/testing a kernel needs `nvcc` ≥ 12 and a CUDA GPU (SM ≥ 80).

## Running experiments

Entry point is always `python -m harness.run_experiment -c <config.yaml>`. Pin the build GPU with
`CUDA_VISIBLE_DEVICES`; set `OPENAI_API_KEY` (`EMPTY` for a local vLLM endpoint).

```bash
# debug mode: model starts from a known-broken kernel (the benchmark's real use case)
python -m harness.run_experiment -c configs/iter/gemma4.yaml --mode debug --v2-task-list benchmark/task_list.json
# run specific tasks only
python -m harness.run_experiment -c configs/agent/agent_smoke.yaml --tasks CUDA_10
```

Two orthogonal selectors decide behavior:

- **`--mode`** (CLI): `generation` = write from scratch (legacy); `debug` = start from a `broken_start`. Debug mode **requires** `--v2-task-list <json>`, whose `entries[].broken_start` supplies `{code_path, error_category, log_excerpt}` per task.
- **`experiment.method`** (config): `iterative` (default; each iter sees prior failure + feedback, stops early on pass/stagnation), `repeated` (independent attempts, per-attempt seed, runs all n for pass@k), or `agent` (ReAct tool-calling loop — see below).

Parallelism: `--n-workers N --gpus 0,2,5` fans tasks across GPUs (`harness/parallel_runner.py`). Use
`--resume <result_dir>` to skip completed tasks. Output lands in `<output_dir>/<timestamp>/` with a
copy of the config and a `manifest.jsonl` of every iteration.

## Architecture

### The core eval loop (`harness/run_experiment.py`)
`run_task()` is the per-task driver. It loads `input/<stem>.json` + `prompts/<stem>.txt`, then loops:
call LLM → `compiler.compile_code` / `run_test` → `classifier.classify_result` → `feedback.build_feedback`.
`method == "agent"` short-circuits to `agent_loop.run_task_agent` instead.

- **`classifier.py`** maps raw build/test output to one of five categories: `compile_error`, `logic_error`, `memory_crash`, `perf_broken`, `timeout` (plus pass). This labeling is shared by the dataset and the harness.
- **`compiler.py`** is the build dispatcher — routes to nvcc / make / inline-cuda / the Class-2 runner depending on the instance's `build_command`.
- **`feedback.py`** implements the feedback axis L0 (none) → L4 (oracle), plus `L3_raw` and error-aware variants. **`few_shot.py`** implements in-context debug-pair retrieval (with leave-one-out contamination guards).
- **`llm_client.py`** is an OpenAI-compatible client. Backend (`vllm` vs cloud `openai`) is auto-detected from `api.base_url` (localhost → vLLM, key defaults to `EMPTY`). Exposes `call_llm`, `call_llm_with_tools`, `call_llm_json_action`.

### The agent subsystem (debug-only, ReAct)
`method: agent` runs a tool-calling debugger. **It requires a `broken_start`** — generation from
scratch is unsupported and raises. Key files:

- **`agent_loop.py`** — `run_task_agent` seeds the broken candidate into `env.current_code`, then runs a two-phase ReAct loop: a **LOCATE phase** where `write_solution` is hidden until the model calls `locate_bug`, then a **FIX phase** where `apply_edit` + `write_solution` (build+test) are unlocked. Each `write_solution` is one "patch submission"; runs are bounded by `agent.max_patch_submissions`, per-tool `quotas`, and `max_tool_calls_per_task`. Writes a `tool_trace.jsonl` beside the manifest.
- **`agent_tools.py`** — the tool registry (`TOOL_SCHEMAS`) and `call_tool`. Tools: `read_code`, `read_test_file`, `read_prompt`, `run_static_analysis`, `run_compute_sanitizer`, `build_only`, `test_only`, `locate_bug`, `apply_edit`, `write_solution`. `schemas_for_config(allowed)` filters to an allow-list for ablations.
- **Blackboard workflow** (`agent.workflow: blackboard`) splits the loop into analyzer → executor → verifier → coder roles, each restricted to a tool subset (`analyzer_tools`/`executor_tools`/`coder_tools` in the config).
- **Context management** — `context_manager.py` (token-budget tracking, budget nudges, history compaction) + `context_summary.py` (LLM-backed L1/L2/L3 summarization of oversized tool results, falling back to head+tail truncation). Tuned via the `agent.token_budget` / `*_ratio` / `tool_result_cap_chars` / `use_llm_summary` config keys.
- **`debug_assist.py`** (precheck, patch-scope analysis, failed-patch verification) and **`diagnostics.py`** (a safe compute-sanitizer wrapper returning a normalized `DiagResult`, never raising) back the investigation tools. **`trace_io.py`** compacts sanitizer output. These are gated by feature flags and inert unless enabled.

### Performance & timing
`performance.py` + `benchmark.py` gate solutions on speedup (`agent.performance_gate.min_speedup`).
`timing_modes/` holds five timing backends (elapsed, kernels, applied-kernels, region, kernelbench).
Class-1/Class-2 PyTorch-extension tasks (kernelbench, flashattention, layernorm) run through the
vendored backend in `applied_kernels_runner/`; `applied_kernels_eval.py` is the adapter.

### Analysis pipeline (`analysis/`)
`scanner.py` walks `outputs/` into Run records → `build_master.py` emits `master.csv` →
`pivot_tables.py` renders the paper tables (`render_latex.py` for booktabs LaTeX).
`axis_normalize.py` parses a config basename into `(model, axis, experiment)`.

```bash
python -m analysis.build_master --results-root outputs --out-dir analysis_out
python -m analysis.pivot_tables  --in-dir analysis_out --out-dir analysis_out
```

## Dataset layout — important

The on-disk dataset is **per-instance**: `data/<instance_id>/` holding `broken_start.cu`,
`reference.cu`, `prompt.txt`, `input.json`, `instance.json`, `testbench/`. Instance IDs are
`<task_family_n>__<error_category>__<hash8>`.

But the **harness reads a flat layout** — `dataset_dir/input/<stem>.json`, `dataset_dir/prompts/<stem>.txt`,
`dataset_dir/testbench/<stem>/`. `scripts/prepare_benchmark.sh` creates that flat layout as gitignored
symlinks over `data/`, and also links the bundled CUTLASS / ThunderKittens headers into each
testbench's `reference_sources/`. **Nothing builds until this script has been run.** Most configs set
`dataset_dir: data`. Re-run the script any time (idempotent); note it collapses on `stem`, so two
instances sharing a stem collide to one link.

## Configs
`configs/` holds the evaluation axes, one file per (model × axis): `iter/`, `repeated/`, `feedback/`,
`history/`, `fewshot/`, `agent/`. Each file's `api:` block ships placeholder `base_url`/key — point it
at your own OpenAI-compatible cloud API or local vLLM server before running. `max_tokens: service_limit`
means "use remaining context after reserve" rather than a fixed cap.

## Vendored code — do not treat as project source
`harness/_vendor/kernelbench/` is upstream KernelBench (MIT) and `data/_external/{cutlass,ThunderKittens}`
are bundled third-party headers. The benchmark is self-contained so reviewers need no external clones;
avoid editing vendored trees.
