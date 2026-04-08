"""
Compare two grid-run summaries side-by-side.

Default behavior (no CLI flags):
- Uses RUN_A_ID and RUN_B_ID below.
- If RUN_B is missing (e.g. still running), it creates a deterministic
  stand-in folder: results/grid_runs/<RUN_B_ID>_FAKE_DO_NOT_REPORT/
  by copying RUN_A and adding small noise to metric values.

Outputs:
- results/comparisons/<RUN_A_ID>_vs_<RUN_B_ID>/overall_by_method.csv
- results/comparisons/<RUN_A_ID>_vs_<RUN_B_ID>/by_dataset_method.csv
- results/comparisons/<RUN_A_ID>_vs_<RUN_B_ID>/by_dataset_method_chunk.csv
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GRID_ROOT = ROOT / "results" / "grid_runs"
OUTPUT_ROOT = ROOT / "results" / "comparisons"

# --- Pick runs to compare ---
# Example for future use:
# RUN_A_ID = "20260328_191443"
# RUN_B_ID = "20260406_235803"
RUN_A_ID = "20260328_191443"
RUN_B_ID = "20260406_235803"

FAKE_SUFFIX = "_FAKE_DO_NOT_REPORT"

# Metrics we know are produced by src.evaluation.evaluate_retrieval
METRIC_COLS = ["MRR", "Recall@1", "Recall@5", "Recall@10", "Recall@100", "NDCG@1", "NDCG@5", "NDCG@10", "NDCG@100"]

def _summary_path(run_id: str) -> Path:
    return GRID_ROOT / run_id / "aggregate" / "summary.json"

def _fake_run_id(run_b_id: str) -> str:
    return f"{run_b_id}{FAKE_SUFFIX}"

def _load_summary_df(run_id: str) -> pd.DataFrame:
    path = _summary_path(run_id)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return df

def _clip_unit_interval(col: str, df: pd.DataFrame) -> pd.DataFrame:
    if col not in df.columns:
        return df
    return df.assign(**{col: df[col].clip(lower=0.0, upper=1.0)})

def create_fake_run_b_from_a() -> str:
    fake_id = _fake_run_id(RUN_B_ID)
    fake_summary_path = _summary_path(fake_id)
    if fake_summary_path.exists():
        print(f"[run_compare] Fake run already exists: {fake_summary_path}")
        return fake_id

    a_path = _summary_path(RUN_A_ID)
    if not a_path.exists():
        raise FileNotFoundError(f"RUN_A summary not found: {a_path}")

    print(f"[run_compare] RUN_B missing; creating fake stand-in: {fake_id}")
    print(f"[run_compare] Base for fake run: {a_path}")

    a_df = _load_summary_df(RUN_A_ID)
    if a_df.empty:
        raise ValueError("RUN_A dataframe is empty; cannot create fake data.")

    # Deterministic perturbations so results are stable between runs of this script.
    seed_material = f"{RUN_A_ID}__{RUN_B_ID}__FAKE".encode("utf-8")
    seed = int.from_bytes(seed_material[:8], byteorder="little", signed=False)
    rng = random.Random(seed)

    out_df = a_df.copy()
    if "run_id" in out_df.columns:
        out_df["run_id"] = fake_id
    else:
        out_df.insert(0, "run_id", fake_id)

    # Small multiplicative noise for quality metrics; slight noise for latency.
    # These are NOT real measurements; the purpose is only pipeline validation.
    for metric in METRIC_COLS:
        if metric not in out_df.columns:
            continue
        # Example perturbation: +/- 2.5% with a bit of extra spread for small values.
        eps_min, eps_max = -0.025, 0.025
        out_df[metric] = out_df[metric].apply(lambda v: float(v) * (1.0 + rng.uniform(eps_min, eps_max)))
        out_df = _clip_unit_interval(metric, out_df)

    if "time_seconds" in out_df.columns:
        out_df["time_seconds"] = out_df["time_seconds"].apply(lambda v: float(v) * (1.0 + rng.uniform(-0.05, 0.05)))
        out_df["time_seconds"] = out_df["time_seconds"].clip(lower=0.0)

    fake_summary_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_json(fake_summary_path, orient="records", indent=2)
    print(f"[run_compare] Fake run created at: {fake_summary_path}")
    return fake_id

def resolve_run_ids() -> tuple[str, str]:
    a_id = RUN_A_ID
    b_id = RUN_B_ID
    if _summary_path(b_id).exists():
        return a_id, b_id
    b_id = create_fake_run_b_from_a()
    return a_id, b_id

def compare_runs(run_a_id: str, run_b_id: str, output_dir: Path) -> None:
    df_a = _load_summary_df(run_a_id)
    df_b = _load_summary_df(run_b_id)

    # Join keys for comparable configs.
    join_keys = ["dataset", "method", "chunk_size", "max_queries"]
    missing_a = [k for k in join_keys if k not in df_a.columns]
    missing_b = [k for k in join_keys if k not in df_b.columns]
    if missing_a or missing_b:
        raise KeyError(f"Missing join keys. RUN_A missing={missing_a}, RUN_B missing={missing_b}")

    metric_cols = [m for m in METRIC_COLS if m in df_a.columns and m in df_b.columns]
    if not metric_cols:
        raise ValueError("No metric columns found to compare.")

    # Keep only what we need before merging (helps performance and avoids name collisions).
    keep_a = join_keys + metric_cols + (["time_seconds"] if "time_seconds" in df_a.columns else [])
    keep_b = join_keys + metric_cols + (["time_seconds"] if "time_seconds" in df_b.columns else [])
    df_a = df_a[keep_a].copy()
    df_b = df_b[keep_b].copy()

    # Rename metric columns with suffixes before merging.
    rename_a = {m: f"{m}_A" for m in metric_cols}
    rename_b = {m: f"{m}_B" for m in metric_cols}
    df_a = df_a.rename(columns=rename_a)
    df_b = df_b.rename(columns=rename_b)

    if "time_seconds" in df_a.columns:
        df_a = df_a.rename(columns={"time_seconds": "time_seconds_A"})
    if "time_seconds" in df_b.columns:
        df_b = df_b.rename(columns={"time_seconds": "time_seconds_B"})

    merged = df_a.merge(df_b, on=join_keys, how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("Merged comparison dataframe is empty. Check that runs share dataset/method/chunk_size/max_queries.")

    # Compute deltas.
    for m in metric_cols:
        merged[f"delta_{m}"] = merged[f"{m}_A"] - merged[f"{m}_B"]

    output_dir.mkdir(parents=True, exist_ok=True)

    overall_rows = []
    for method in sorted(merged["method"].unique().tolist()):
        mdf = merged[merged["method"] == method].copy()
        row = {"method": method, "n_configs": len(mdf)}
        for met in metric_cols:
            row[f"embed1_{met}"] = float(mdf[f"{met}_A"].mean())
            row[f"embed2_{met}"] = float(mdf[f"{met}_B"].mean())
            row[f"delta_{met}"] = float((mdf[f"delta_{met}"]).mean())
        overall_rows.append(row)
    overall_df = pd.DataFrame(overall_rows).sort_values("method")
    overall_df.to_csv(output_dir / "overall_by_method.csv", index=False)

    by_dm = merged.groupby(["dataset", "method"], as_index=False)
    by_dm_rows = []
    for _, g in by_dm:
        row = {"dataset": g["dataset"].iloc[0], "method": g["method"].iloc[0], "n_configs": len(g)}
        for met in metric_cols:
            row[f"embed1_{met}"] = float(g[f"{met}_A"].mean())
            row[f"embed2_{met}"] = float(g[f"{met}_B"].mean())
            row[f"delta_{met}"] = float(g[f"delta_{met}"].mean())
        by_dm_rows.append(row)
    by_dm_df = pd.DataFrame(by_dm_rows).sort_values(["dataset", "method"])
    by_dm_df.to_csv(output_dir / "by_dataset_method.csv", index=False)

    cols_for_chunk = join_keys + [f"{m}_A" for m in metric_cols] + [f"{m}_B" for m in metric_cols] + [f"delta_{m}" for m in metric_cols]
    if "time_seconds_A" in merged.columns and "time_seconds_B" in merged.columns:
        cols_for_chunk = cols_for_chunk + ["time_seconds_A", "time_seconds_B"]
    chunk_df = merged[cols_for_chunk].sort_values(join_keys)
    chunk_df.to_csv(output_dir / "by_dataset_method_chunk.csv", index=False)

    print(f"[run_compare] Comparison complete. Wrote to: {output_dir}")
    print(f"[run_compare] overall_by_method.csv: {output_dir / 'overall_by_method.csv'}")

    # Console preview: overall table only (keeps output readable).
    preview_cols = ["method", "n_configs"] + [c for c in overall_df.columns if c.startswith("embed1_") or c.startswith("embed2_") or c.startswith("delta_")]
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(overall_df[preview_cols].to_string(index=False))

def main() -> None:
    a_id, b_id = resolve_run_ids()
    out_dir = OUTPUT_ROOT / f"{a_id}_vs_{b_id}"
    print(f"[run_compare] RUN_A={a_id}")
    print(f"[run_compare] RUN_B={b_id}")
    compare_runs(a_id, b_id, out_dir)

if __name__ == "__main__":
    main()
