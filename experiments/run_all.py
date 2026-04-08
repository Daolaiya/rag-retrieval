"""
Run full experiment suite: multiple datasets, methods, chunk sizes.
CPU-only. May take 30-60 minutes depending on hardware.
"""

import json
import subprocess
import sys
from pathlib import Path

# Project root (the `project/` folder)
ROOT = Path(__file__).resolve().parent.parent

def main():
    datasets = ["nfcorpus", "scifact"]  # Small datasets for CPU
    chunk_sizes = [128, 256, 512]
    max_queries = 150  # Limit for faster run

    all_results = []

    for dataset in datasets:
        print("\n" + "=" * 60)
        cmd = [sys.executable, "-m", "src.run_experiments", "--dataset", dataset, "--methods", "dense", "sparse", "hybrid",
               "--chunk_sizes", *map(str, chunk_sizes), "--max_queries", str(max_queries), "--output_dir", str(ROOT / "results")]
        print(f"Running: {' '.join(cmd)}")
        print("=" * 60)
        result = subprocess.run(cmd, cwd=str(ROOT))
        if result.returncode != 0:
            print(f"Error: {dataset} failed")
            sys.exit(1)

    # Aggregate results
    results_dir = ROOT / "results" / "metrics"
    for f in results_dir.glob("*_summary.json"):
        with open(f, encoding="utf-8") as fp:
            data = json.load(fp)
            all_results.extend(data)

    summary_path = ROOT / "results" / "full_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll experiments complete. Summary: {summary_path}")

if __name__ == "__main__":
    main()
