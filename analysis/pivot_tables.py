"""CLI: master.csv → 6 .tex/.csv tables + summary.md.

Usage:
    python -m harness.tools.integrated.pivot_tables \
        --in-dir evaluation/outputs_analysis/integrated \
        --out-dir evaluation/outputs_analysis/integrated
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from .render_latex import render_table

_METRICS = (
    "pass_at_1",
    "pass_at_k",
    "debug_rate_at_k",
    "compilation_rate_at_1",
    "avg_iters_to_pass",
)

_AXIS_ORDER = {
    "feedback": ["L0", "L1", "L2", "L3", "L3_raw", "L4"],
    "history":  ["H1", "H2", "H3", "H4"],
    "fewshot":  ["matched_K1", "matched_K3", "matched_K5",
                 "random_K1",  "random_K3",  "random_K5",
                 "curated_K1", "curated_K3", "curated_K5"],
}

_MODEL_ORDER = ["gpt5_4", "qwen36_27b", "gemma4", "glm_4_7", "kimi", "qwen3_6plus", "minimax_m2_7"]


def _pivot_pass_at_k(master: pd.DataFrame, experiment: str) -> pd.DataFrame:
    sub = master[(master["experiment"] == experiment) & (master["metric"] == "pass_at_k")].copy()
    if sub.empty:
        return pd.DataFrame()
    pivoted = sub.pivot(index="model", columns="axis", values="value").reset_index()
    cols = ["model"] + [a for a in _AXIS_ORDER.get(experiment, sorted(sub["axis"].unique()))
                        if a in pivoted.columns]
    pivoted = pivoted[cols]
    pivoted["__model_order"] = pivoted["model"].apply(
        lambda m: _MODEL_ORDER.index(m) if m in _MODEL_ORDER else 999
    )
    pivoted = pivoted.sort_values("__model_order").drop(columns="__model_order").reset_index(drop=True)
    return pivoted


def _coverage_footnote(master: pd.DataFrame, experiment: str) -> str | None:
    sub = master[(master["experiment"] == experiment) & (master["metric"] == "pass_at_k")]
    non_full = sub[sub["coverage"] != "main+memcrash"]
    if non_full.empty:
        return None
    parts = []
    for _, r in non_full.iterrows():
        parts.append(f"{r['model']}/{r['axis']}={r['coverage']}")
    return "Coverage (non main+memcrash): " + "; ".join(parts)


def _build_exp1_iter(master: pd.DataFrame) -> pd.DataFrame:
    sub = master[master["experiment"] == "iter"].copy()
    if sub.empty:
        return pd.DataFrame()
    rows = []
    for model in _MODEL_ORDER:
        ms = sub[sub["model"] == model]
        if ms.empty:
            continue
        def get(metric):
            r = ms[ms["metric"] == metric]
            return r["value"].iloc[0] if not r.empty else None
        rows.append({
            "model": model,
            "pass@1": get("pass_at_1"),
            "pass@k": get("pass_at_k"),
            "compile@1": get("compilation_rate_at_1"),
            "debug@k": get("debug_rate_at_k"),
            "n_tasks": int(ms["n_tasks"].iloc[0]) if "n_tasks" in ms else None,
        })
    return pd.DataFrame(rows)


def _build_exp2_repeated_vs_iter(master: pd.DataFrame) -> pd.DataFrame:
    iter_pk = master[(master["experiment"] == "iter") & (master["metric"] == "pass_at_k")].set_index("model")["value"]
    rep_pk = master[(master["experiment"] == "repeated") & (master["metric"] == "pass_at_k")].set_index("model")["value"]
    rows = []
    for model in _MODEL_ORDER:
        if model not in iter_pk.index and model not in rep_pk.index:
            continue
        ipk = iter_pk.get(model)
        rpk = rep_pk.get(model)
        delta = (ipk - rpk) if (ipk is not None and rpk is not None and not pd.isna(ipk) and not pd.isna(rpk)) else None
        rows.append({"model": model, "iter pass@k": ipk, "repeated pass@5": rpk, "Delta (iter-rep)": delta})
    return pd.DataFrame(rows)


def _build_overview(master: pd.DataFrame) -> pd.DataFrame:
    sub = master[master["metric"] == "pass_at_k"]
    rows = []
    for model in _MODEL_ORDER:
        ms = sub[sub["model"] == model]
        if ms.empty:
            continue
        row: dict = {"model": model}
        for exp in ["iter", "repeated", "feedback", "history", "fewshot"]:
            es = ms[ms["experiment"] == exp]
            row[f"best_{exp}"] = es["value"].max() if not es.empty else None
        rows.append(row)
    return pd.DataFrame(rows)


def _coverage_matrix(master: pd.DataFrame) -> str:
    sub = master[master["metric"] == "pass_at_k"][["model", "experiment", "axis", "coverage"]].drop_duplicates()
    if sub.empty:
        return "(no data)"
    sub = sub.copy()
    sub["axis_full"] = sub["experiment"] + "/" + sub["axis"]
    pivoted = sub.pivot(index="model", columns="axis_full", values="coverage").fillna("missing")

    def emoji(c: str) -> str:
        if c == "main+memcrash": return "OK"
        if c == "v2_only": return "v2"
        if c == "v3_only": return "v3"
        if c.startswith("main_partial") or c.startswith("memcrash_partial"): return "P"
        if c == "empty": return "X"
        if c == "missing": return "-"
        return "?"

    cols = sorted(pivoted.columns)
    lines = ["| model | " + " | ".join(cols) + " |"]
    lines.append("|" + "---|" * (1 + len(cols)))
    for model in _MODEL_ORDER:
        if model not in pivoted.index:
            continue
        cells = [model] + [emoji(pivoted.loc[model, c]) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _take_aways(master: pd.DataFrame) -> dict[str, str]:
    sub = master[master["metric"] == "pass_at_k"]
    out: dict[str, str] = {}

    def best_worst(exp: str) -> str:
        es = sub[sub["experiment"] == exp]
        if es.empty:
            return "(no data)"
        per_model = es.groupby("model")["value"].max().sort_values(ascending=False)
        if per_model.empty:
            return "(no data)"
        best_model, best_v = per_model.index[0], per_model.iloc[0]
        worst_model, worst_v = per_model.index[-1], per_model.iloc[-1]
        return f"Best: **{best_model}** pass@k={best_v:.3f}; worst: **{worst_model}** pass@k={worst_v:.3f}."

    for exp in ["iter", "repeated", "feedback", "history", "fewshot"]:
        out[exp] = best_worst(exp)
    return out


def _emit_per_model_md(master: pd.DataFrame, out_dir: Path) -> None:
    """Emit one Markdown file per canonical model summarizing all experiments."""
    per_model_dir = out_dir / "per_model"
    per_model_dir.mkdir(parents=True, exist_ok=True)

    for model in _MODEL_ORDER:
        ms = master[master["model"] == model]
        if ms.empty:
            continue

        lines = [f"# {model}", ""]
        lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}. "
                     f"Source: master.csv ({len(ms)} rows for this model).")
        lines.append("")

        # Coverage summary — count axes by coverage label bucket
        cov_axes = ms[ms["metric"] == "pass_at_k"][["axis", "coverage"]].drop_duplicates()
        cov_buckets: dict[str, int] = {"main+memcrash": 0, "v2_only": 0, "v3_only": 0, "partial": 0}
        for _, r in cov_axes.iterrows():
            c = r["coverage"]
            if c.startswith("main_partial") or c.startswith("memcrash_partial"):
                cov_buckets["partial"] += 1
            elif c in cov_buckets:
                cov_buckets[c] += 1
        lines.append("## Coverage Summary")
        lines.append("")
        cov_str = ", ".join(f"{k}={v}" for k, v in cov_buckets.items() if v > 0)
        lines.append(cov_str if cov_str else "(no data)")
        lines.append("")

        # Single-axis experiments: iter, repeated
        for exp_label, exp_name in [("Exp 1: Iter Base", "iter"), ("Exp 2: Repeated@k", "repeated")]:
            es = ms[ms["experiment"] == exp_name]
            lines.append(f"## {exp_label}")
            lines.append("")
            if es.empty:
                lines.append("(no data)")
                lines.append("")
                continue
            lines.append("| metric | value | n_tasks | n_main | n_memcrash | coverage |")
            lines.append("|---|---|---|---|---|---|")
            for metric in _METRICS:
                row = es[es["metric"] == metric]
                if row.empty:
                    continue
                r = row.iloc[0]
                v = r["value"]
                v_str = f"{v:.4f}" if pd.notna(v) else "--"
                lines.append(f"| {metric} | {v_str} | {int(r['n_tasks'])} | {int(r['n_main'])} | {int(r['n_memcrash'])} | {r['coverage']} |")
            lines.append("")

        # Multi-axis experiments: feedback, history, fewshot
        for exp_label, exp_name in [("Exp 3: Feedback Ablation", "feedback"),
                                     ("Exp 4: History Ablation", "history"),
                                     ("Exp 5: Few-shot Ablation", "fewshot")]:
            es = ms[ms["experiment"] == exp_name]
            lines.append(f"## {exp_label}")
            lines.append("")
            if es.empty:
                lines.append("(no data)")
                lines.append("")
                continue

            axes_seen = es["axis"].unique().tolist()
            ordered = [a for a in _AXIS_ORDER.get(exp_name, sorted(axes_seen)) if a in axes_seen]

            lines.append("| axis | pass@1 | pass@k | debug@k | compile@1 | mean_iters | n_tasks | coverage |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for axis in ordered:
                ax_rows = es[es["axis"] == axis]
                if ax_rows.empty:
                    continue
                def gv(metric):
                    r = ax_rows[ax_rows["metric"] == metric]
                    if r.empty: return "--"
                    v = r["value"].iloc[0]
                    return f"{v:.4f}" if pd.notna(v) else "--"
                first = ax_rows.iloc[0]
                lines.append(f"| {axis} | {gv('pass_at_1')} | {gv('pass_at_k')} | {gv('debug_rate_at_k')} "
                             f"| {gv('compilation_rate_at_1')} | {gv('avg_iters_to_pass')} "
                             f"| {int(first['n_tasks'])} | {first['coverage']} |")
            lines.append("")

        out_path = per_model_dir / f"{model}.md"
        out_path.write_text("\n".join(lines))


def _write_summary_md(master: pd.DataFrame, out_path: Path, source_summary: str) -> None:
    take = _take_aways(master)
    cov = _coverage_matrix(master)
    n_runs = master[["model", "experiment", "axis"]].drop_duplicates().shape[0]
    lines = [
        "# CUDA-Debugger Experiments — Integrated Results",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')} from {n_runs} (model x axis) cells, "
        f"7 models, 5 experiments. {source_summary}",
        "",
        "## Coverage Matrix",
        "",
        "Legend: OK=main+memcrash, v2=v2_only, v3=v3_only, P=partial, -=missing",
        "",
        cov,
        "",
        "## Take-aways",
        "",
        "### Exp 1: Single-shot Iterative Debug (iter)",
        "", take["iter"], "",
        "See `tables/exp1_iter.tex` for the full numbers.",
        "",
        "### Exp 2: Repeated@k vs Iterative@k",
        "", take["repeated"], "",
        "See `tables/exp2_repeated_vs_iter.tex`.",
        "",
        "### Exp 3: Feedback-level Ablation (L0->L4, L3_raw)",
        "", take["feedback"], "",
        "See `tables/exp3_feedback.tex`.",
        "",
        "### Exp 4: History-rounds Ablation (H1->H4)",
        "", take["history"], "",
        "See `tables/exp4_history.tex`.",
        "",
        "### Exp 5: Few-shot (3 strategies x K in {1,3,5})",
        "", take["fewshot"], "",
        "See `tables/exp5_fewshot.tex`.",
        "",
        "## Cross-experiment Pass@k",
        "",
        "See `tables/overview_cross_exp.tex`.",
        "",
        "## Reproducibility",
        "",
        "Regenerate with:",
        "```bash",
        "python -m harness.tools.integrated.build_master --results-root evaluation --out-dir evaluation/outputs_analysis/integrated",
        "python -m harness.tools.integrated.pivot_tables --in-dir evaluation/outputs_analysis/integrated --out-dir evaluation/outputs_analysis/integrated",
        "```",
    ]
    out_path.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    master = pd.read_csv(args.in_dir / "master.csv")
    tables_dir = args.out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    def _emit(stem: str, df: pd.DataFrame, *, bold: str, caption: str, footnote: str | None):
        if df.empty:
            return
        df.to_csv(tables_dir / f"{stem}.csv", index=False)
        tex = render_table(df, index_col="model", bold=bold, caption=caption, footnote=footnote)
        (tables_dir / f"{stem}.tex").write_text(tex)

    _emit("exp1_iter", _build_exp1_iter(master),
          bold="none", caption="Exp 1: iter base (multi-metric)",
          footnote=_coverage_footnote(master, "iter"))
    _emit("exp2_repeated_vs_iter", _build_exp2_repeated_vs_iter(master),
          bold="none", caption="Exp 2: pass@5 vs iter@k",
          footnote=_coverage_footnote(master, "repeated"))
    _emit("exp3_feedback", _pivot_pass_at_k(master, "feedback"),
          bold="row_max", caption="Exp 3: feedback ablation (pass@k)",
          footnote=_coverage_footnote(master, "feedback"))
    _emit("exp4_history", _pivot_pass_at_k(master, "history"),
          bold="row_max", caption="Exp 4: history ablation (pass@k)",
          footnote=_coverage_footnote(master, "history"))
    _emit("exp5_fewshot", _pivot_pass_at_k(master, "fewshot"),
          bold="row_max", caption="Exp 5: few-shot ablation (pass@k)",
          footnote=_coverage_footnote(master, "fewshot"))
    _emit("overview_cross_exp", _build_overview(master),
          bold="row_max", caption="Cross-experiment: best pass@k per model",
          footnote=None)

    summary_path = args.out_dir / "summary.md"
    _write_summary_md(master, summary_path,
                      source_summary=f"Source: {args.in_dir}/master.csv ({len(master)} rows).")
    _emit_per_model_md(master, args.out_dir)
    per_model_dir = args.out_dir / "per_model"
    print(f"[pivot_tables] tables -> {tables_dir}; summary -> {summary_path}; per_model -> {per_model_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
