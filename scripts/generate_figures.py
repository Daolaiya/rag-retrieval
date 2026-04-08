"""
Generate report figures from results/metrics/*_summary.json.
Runs with no CLI flags and writes figures to results/metrics/figures.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS = ["MRR", "Recall@10", "NDCG@10"]

def _chunk_sort_key(label: str) -> tuple[int, int]:
    if label == "original":
        return (0, 0)
    if label.startswith("chunk_"):
        try:
            return (1, int(label.replace("chunk_", "")))
        except ValueError:
            return (2, 0)
    return (2, 0)

def load_results() -> pd.DataFrame:
    rows = []
    summary_files = sorted(RESULTS_DIR.glob("*_summary.json"))
    for path in summary_files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            rows.extend(data)
        elif isinstance(data, dict):
            rows.append(data)

    if not rows:
        sample = RESULTS_DIR / "sample_results.json"
        if sample.exists():
            with open(sample, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                rows.extend(data)
            elif isinstance(data, dict):
                rows.append(data)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    required = {"dataset", "method", "chunk_size"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    return df

def _aggregate(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if value_col not in df.columns:
        return pd.DataFrame()
    keep_cols = ["dataset", "method", "chunk_size", value_col]
    work = df[keep_cols].dropna(subset=[value_col]).copy()
    if work.empty:
        return pd.DataFrame()
    return work.groupby(["dataset", "method", "chunk_size"], as_index=False)[value_col].mean()

def plot_method_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    df_orig = df[df["chunk_size"] == "original"].copy()
    if df_orig.empty:
        print("Skipping method-comparison plots: no 'original' rows found.")
        return

    for metric in METRICS:
        grouped = _aggregate(df_orig, metric)
        if grouped.empty:
            continue
        pivot = grouped.pivot(index="dataset", columns="method", values=metric).sort_index()
        if pivot.empty:
            continue
        pivot.plot(kind="bar", rot=0, figsize=(8, 4.8))
        plt.ylabel(metric)
        plt.title(f"{metric} by Method and Dataset (original corpus)")
        plt.legend(title="Method")
        plt.tight_layout()
        plt.savefig(str(output_dir / f"method_comparison_{metric.lower().replace('@', 'at')}.pdf"), bbox_inches="tight")
        plt.close()

def plot_chunk_impact(df: pd.DataFrame, output_dir: Path) -> None:
    chunk_rows = df[df["chunk_size"].notna()].copy()
    if chunk_rows.empty:
        print("Skipping chunk-impact plots: no chunk data found.")
        return

    chunk_levels = sorted(chunk_rows["chunk_size"].unique().tolist(), key=_chunk_sort_key)
    chunk_pos = {name: idx for idx, name in enumerate(chunk_levels)}
    datasets = sorted(chunk_rows["dataset"].unique().tolist())

    for dataset in datasets:
        ddf = chunk_rows[chunk_rows["dataset"] == dataset].copy()
        if ddf.empty:
            continue

        fig, axes = plt.subplots(1, len(METRICS), figsize=(5.4 * len(METRICS), 4.8), sharex=True)
        if len(METRICS) == 1:
            axes = [axes]

        plotted_any = False
        methods = sorted(ddf["method"].unique().tolist())
        for i, metric in enumerate(METRICS):
            grouped = _aggregate(ddf, metric)
            if grouped.empty:
                axes[i].set_visible(False)
                continue
            metric_df = grouped[grouped["dataset"] == dataset].copy()
            if metric_df.empty:
                axes[i].set_visible(False)
                continue
            for method in methods:
                mdf = metric_df[metric_df["method"] == method].copy()
                if mdf.empty:
                    continue
                mdf["chunk_pos"] = mdf["chunk_size"].map(chunk_pos)
                mdf = mdf.sort_values("chunk_pos")
                axes[i].plot(mdf["chunk_pos"], mdf[metric], marker="o", label=method)
                plotted_any = True
            axes[i].set_title(metric)
            axes[i].set_xlabel("Chunk setting")
            axes[i].set_ylabel(metric)
            axes[i].set_xticks(list(range(len(chunk_levels))))
            axes[i].set_xticklabels(chunk_levels, rotation=30, ha="right")
            axes[i].grid(alpha=0.25)

        if plotted_any:
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, title="Method", loc="upper center", ncol=max(1, len(labels)))
            fig.suptitle(f"Chunk-size impact by metric ({dataset})", y=1.05)
            fig.tight_layout()
            safe_dataset = dataset.replace("/", "_")
            plt.savefig(str(output_dir / f"chunk_impact_{safe_dataset}.pdf"), bbox_inches="tight")
        plt.close(fig)

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_results()
    if df.empty:
        print("No valid metrics summary data found. Run experiments first.")
        return
    plot_method_comparison(df, FIGURES_DIR)
    plot_chunk_impact(df, FIGURES_DIR)
    print(f"Figures saved to {FIGURES_DIR}")

if __name__ == "__main__":
    main()
