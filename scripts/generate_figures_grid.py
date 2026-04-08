"""
Generate comprehensive figures from one grid run summary.
Default input: results/grid_runs/20260328_191443/aggregate/summary.json if present,
otherwise the latest run under results/grid_runs.
Outputs: results/grid_runs/<run_id>/figures/*.pdf
"""

from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GRID_ROOT = ROOT / "results" / "grid_runs"
PREFERRED_RUN_ID = "20260328_191443"
# PREFERRED_RUN_ID = "20260406_235803"
METRICS = ["MRR", "Recall@10", "NDCG@10", "Recall@1", "Recall@5", "Recall@100", "NDCG@1", "NDCG@5", "NDCG@100"]

def _chunk_sort_key(label: str) -> tuple[int, int]:
    if label == "original":
        return (0, 0)
    if label.startswith("chunk_"):
        try:
            return (1, int(label.replace("chunk_", "")))
        except ValueError:
            return (2, 0)
    return (2, 0)

def resolve_run_dir() -> Path | None:
    preferred = GRID_ROOT / PREFERRED_RUN_ID
    if (preferred / "aggregate" / "summary.json").exists():
        return preferred
    if not GRID_ROOT.exists():
        return None
    candidates = [p for p in GRID_ROOT.iterdir() if p.is_dir() and (p / "aggregate" / "summary.json").exists()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.name)[-1]

def load_grid_df(run_dir: Path) -> pd.DataFrame:
    summary_path = run_dir / "aggregate" / "summary.json"
    with open(summary_path, encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    if df.empty:
        return df
    required = {"dataset", "method", "chunk_size"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    return df

def _aggregate(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if value_col not in df.columns:
        return pd.DataFrame()
    keys = ["dataset", "method", "chunk_size"]
    work = df[keys + [value_col]].dropna(subset=[value_col]).copy()
    if work.empty:
        return pd.DataFrame()
    return work.groupby(keys, as_index=False)[value_col].mean()

def plot_method_comparison_original(df: pd.DataFrame, out_dir: Path) -> None:
    df_orig = df[df["chunk_size"] == "original"].copy()
    if df_orig.empty:
        print("Skipping original-corpus method comparison: no 'original' rows.")
        return
    for metric in METRICS:
        grouped = _aggregate(df_orig, metric)
        if grouped.empty:
            continue
        pivot = grouped.pivot(index="dataset", columns="method", values=metric).sort_index()
        if pivot.empty:
            continue
        pivot.plot(kind="bar", rot=0, figsize=(9, 5))
        plt.ylabel(metric)
        plt.title(f"{metric} by Method and Dataset (original corpus)")
        plt.legend(title="Method")
        plt.tight_layout()
        plt.savefig(str(out_dir / f"original_method_{metric.lower().replace('@', 'at')}.pdf"), bbox_inches="tight")
        plt.close()

def plot_chunk_impact(df: pd.DataFrame, out_dir: Path) -> None:
    chunk_levels = sorted(df["chunk_size"].dropna().unique().tolist(), key=_chunk_sort_key)
    if not chunk_levels:
        print("Skipping chunk-impact plots: no chunk labels found.")
        return
    chunk_pos = {name: idx for idx, name in enumerate(chunk_levels)}
    datasets = sorted(df["dataset"].dropna().unique().tolist())
    methods = sorted(df["method"].dropna().unique().tolist())

    for dataset in datasets:
        dataset_df = df[df["dataset"] == dataset].copy()
        if dataset_df.empty:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharex=True)
        core_metrics = [m for m in ["MRR", "Recall@10", "NDCG@10"] if m in dataset_df.columns]
        while len(core_metrics) < 3:
            core_metrics.append(None)

        plotted_any = False
        for ax, metric in zip(axes, core_metrics):
            if metric is None:
                ax.set_visible(False)
                continue
            grouped = _aggregate(dataset_df, metric)
            metric_df = grouped[grouped["dataset"] == dataset].copy()
            if metric_df.empty:
                ax.set_visible(False)
                continue
            for method in methods:
                mdf = metric_df[metric_df["method"] == method].copy()
                if mdf.empty:
                    continue
                mdf["chunk_pos"] = mdf["chunk_size"].map(chunk_pos)
                mdf = mdf.sort_values("chunk_pos")
                ax.plot(mdf["chunk_pos"], mdf[metric], marker="o", linewidth=2.4, markersize=7, label=method)
                plotted_any = True
            ax.set_title(metric, fontsize=14)
            ax.set_xlabel("Chunk setting", fontsize=12)
            ax.set_ylabel(metric, fontsize=12)
            ax.set_xticks(list(range(len(chunk_levels))))
            ax.set_xticklabels(chunk_levels, rotation=25, ha="right")
            ax.tick_params(axis="both", labelsize=11)
            ax.grid(alpha=0.25)

        if plotted_any:
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, title="Method", loc="center left", bbox_to_anchor=(0.93, 0.5), frameon=True, fontsize=11, title_fontsize=11)
            fig.suptitle(f"Chunk-size impact by method ({dataset})", y=1.02, fontsize=15)
            fig.tight_layout(rect=[0, 0, 0.91, 1.0])
            safe_dataset = dataset.replace("/", "_")
            plt.savefig(str(out_dir / f"chunk_impact_{safe_dataset}.pdf"), bbox_inches="tight", dpi=300)
        plt.close(fig)

def plot_metric_heatmaps(df: pd.DataFrame, out_dir: Path) -> None:
    keys = ["dataset", "method"]
    for metric in ["MRR", "Recall@10", "NDCG@10"]:
        if metric not in df.columns:
            continue
        grouped = df[keys + [metric]].dropna(subset=[metric]).groupby(keys, as_index=False)[metric].mean()
        if grouped.empty:
            continue
        pivot = grouped.pivot(index="dataset", columns="method", values=metric).sort_index()
        if pivot.empty:
            continue
        fig, ax = plt.subplots(figsize=(7.2, max(3.5, 0.8 * len(pivot.index))))
        im = ax.imshow(pivot.values, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(f"{metric} heatmap (mean across chunk settings)")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                value = pivot.values[i, j]
                if pd.notna(value):
                    ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        plt.savefig(str(out_dir / f"heatmap_{metric.lower().replace('@', 'at')}.pdf"), bbox_inches="tight")
        plt.close(fig)

def plot_runtime_tradeoff(df: pd.DataFrame, out_dir: Path) -> None:
    if "time_seconds" not in df.columns:
        return
    quality_metric = "Recall@10" if "Recall@10" in df.columns else ("MRR" if "MRR" in df.columns else None)
    if quality_metric is None:
        return
    datasets = sorted(df["dataset"].dropna().unique().tolist())
    methods = sorted(df["method"].dropna().unique().tolist())
    for dataset in datasets:
        ddf = df[df["dataset"] == dataset].copy()
        if ddf.empty:
            continue
        fig, ax = plt.subplots(figsize=(7.4, 5))
        plotted_any = False
        for method in methods:
            mdf = ddf[ddf["method"] == method].copy()
            if mdf.empty or quality_metric not in mdf.columns:
                continue
            ax.scatter(mdf["time_seconds"], mdf[quality_metric], label=method, alpha=0.85)
            plotted_any = True
        if plotted_any:
            ax.set_xlabel("time_seconds")
            ax.set_ylabel(quality_metric)
            ax.set_title(f"Quality vs runtime ({dataset})")
            ax.grid(alpha=0.25)
            ax.legend(title="Method")
            fig.tight_layout()
            safe_dataset = dataset.replace("/", "_")
            plt.savefig(str(out_dir / f"runtime_tradeoff_{safe_dataset}.pdf"), bbox_inches="tight")
        plt.close(fig)

def main() -> None:
    run_dir = resolve_run_dir()
    if run_dir is None:
        print("No valid grid run found under results/grid_runs.")
        return
    out_dir = run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_grid_df(run_dir)
    if df.empty:
        print(f"No usable rows in {run_dir / 'aggregate' / 'summary.json'}.")
        return

    print(f"Using grid run: {run_dir.name}")
    print(f"Rows loaded: {len(df)}")
    plot_method_comparison_original(df, out_dir)
    plot_chunk_impact(df, out_dir)
    plot_metric_heatmaps(df, out_dir)
    plot_runtime_tradeoff(df, out_dir)
    print(f"Figures saved to {out_dir}")

if __name__ == "__main__":
    main()
