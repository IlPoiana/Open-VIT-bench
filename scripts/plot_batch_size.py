#!/usr/bin/env python3
import os
import re
import json
import argparse
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

PATTERN = re.compile(r"^(\d+)-(\d+)-(\d+)\.json$")  # batch-size-k_type.json

def load_kernel_means(directory: str, k_type: str, size_filter: str | None):
    """
    Returns:
      dict[size_str] -> (list of batch_sizes, list of kernel_means)
    """
    grouped = defaultdict(lambda: {"batches": [], "means": []})

    for fname in os.listdir(directory):
        m = PATTERN.match(fname)
        if not m:
            continue
        batch_str, size_str, k_str = m.groups()
        if k_str != k_type:
            continue
        if size_filter is not None and size_str != size_filter:
            continue

        fpath = os.path.join(directory, fname)
        with open(fpath, "r") as f:
            data = json.load(f)
        kernels = [row["kernel"] for row in data if "kernel" in row]
        if not kernels:
            continue

        grouped[size_str]["batches"].append(int(batch_str))
        grouped[size_str]["means"].append(float(np.mean(kernels)))

    # sort by batch size within each size group
    for size_str, d in grouped.items():
        pairs = sorted(zip(d["batches"], d["means"]), key=lambda x: x[0])
        if pairs:
            grouped[size_str]["batches"], grouped[size_str]["means"] = map(list, zip(*pairs))
        else:
            grouped[size_str]["batches"], grouped[size_str]["means"] = [], []

    return grouped

def plot_and_save(grouped, k_type: str, out_path: str):
    if not grouped:
        raise SystemExit("No matching files found to plot.")

    plt.figure(figsize=(9, 6))
    # one line per size
    for size_str, d in sorted(grouped.items(), key=lambda kv: int(kv[0])):
        if not d["batches"]:
            continue
        label = f"size={size_str}"
        plt.plot(d["batches"], d["means"], marker="o", label=label)

    plt.xlabel("Batch Size")
    plt.ylabel("Average Kernel")
    plt.yscale("log")
    sizes_in_plot = ",".join(sorted(grouped.keys(), key=lambda s: int(s)))
    plt.title(f"Average Kernel vs Batch Size (k_type={k_type}; sizes={sizes_in_plot})")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot to: {out_path}")

def main():
    p = argparse.ArgumentParser(description="Plot average kernel vs batch size.")
    p.add_argument("directory", help="Directory containing batch-size-k_type.json files")
    p.add_argument("k_type", help="k_type to filter (e.g., 2)")
    p.add_argument("--size", help="Optional size filter (middle number, e.g., 64)")
    p.add_argument("--out", help="Output PNG path (default auto-generated)")
    args = p.parse_args()

    grouped = load_kernel_means(args.directory, args.k_type, args.size)

    if args.out:
        out_path = args.out
    else:
        tag = f"all-sizes" if args.size is None else f"size-{args.size}"
        out_path = f"kernel_vs_batch_k{args.k_type}_{tag}.png"

    plot_and_save(grouped, args.k_type, out_path)

if __name__ == "__main__":
    main()
