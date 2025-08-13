#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# batch-size-rep.json  ->  e.g., 4-64-2.json
FILENAME_RE = re.compile(r"^(\d+)-(\d+)-(\d+)\.json$", re.IGNORECASE)

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Plot mean(kernel) vs size N for a given batch, grouped by replicate."
    )
    ap.add_argument("dir", type=str,
                    help="Directory containing JSON files like 'BATCH-N-REP.json'.")
    ap.add_argument("batch", type=int,
                    help="Batch size to filter (first number in filename).")
    ap.add_argument("--out", type=str, default=None,
                    help="Optional: path to save the figure (PNG).")
    return ap.parse_args()

def read_mean_kernel(json_path: Path) -> float:
    """Read the JSON list and return the mean of 'kernel' values."""
    data = json.loads(json_path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"{json_path.name}: JSON root must be a list.")
    vals: List[float] = []
    for item in data:
        vals.append(float(item["kernel"]))  # guaranteed by problem statement
    if not vals:
        raise ValueError(f"{json_path.name}: no kernel values found.")
    return float(np.mean(vals))

def collect_points(dirpath: Path, target_batch: int) -> Dict[int, List[Tuple[int, float]]]:
    """
    Return a dict: {replicate: [(N, mean_kernel), ...], ...}
    Only includes files matching the given batch (first number in filename).
    """
    by_rep: Dict[int, List[Tuple[int, float]]] = {}
    for p in dirpath.glob("*.json"):
        m = FILENAME_RE.match(p.name)
        if not m:
            continue
        batch_str, size_str, rep_str = m.groups()
        if int(batch_str) != target_batch:
            continue

        N = int(size_str)
        rep = int(rep_str)

        try:
            mean_kernel = read_mean_kernel(p)
        except Exception as e:
            print(f"Skipping {p.name}: {e}")
            continue

        by_rep.setdefault(rep, []).append((N, mean_kernel))

    # Sort each replicate's points by N
    for rep in by_rep:
        by_rep[rep].sort(key=lambda t: t[0])

    if not by_rep:
        raise SystemExit(
            f"No matching JSON files found for batch={target_batch}. "
            "Expected names like 'BATCH-N-REP.json'."
        )
    return by_rep

def plot(by_rep: Dict[int, List[Tuple[int, float]]], batch: int, save_path: str = None) -> None:
    plt.figure(figsize=(7, 4.5))
    for rep, points in sorted(by_rep.items()):
        Ns = [n for n, _ in points]
        means = [v for _, v in points]
        plt.plot(Ns, means, marker='o', label=f"kernel-{rep}")

    plt.yscale('log')
    plt.xlabel("image size (N)")
    plt.ylabel("Mean kernel time")
    plt.title(f"Kernel average time (batch={batch})")
    plt.grid(True, which='both', axis='both')
    plt.legend(title="Replicate")

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
        print(f"Saved figure to {save_path}")
    else:
        plt.show()

def main():
    args = parse_args()
    dirpath = Path(args.dir)
    if not dirpath.is_dir():
        raise SystemExit(f"Not a directory: {dirpath}")
    by_rep = collect_points(dirpath, args.batch)
    plot(by_rep, batch=args.batch, save_path=args.out)

if __name__ == "__main__":
    main()
