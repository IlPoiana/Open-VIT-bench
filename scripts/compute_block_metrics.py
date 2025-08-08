#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from collections import defaultdict
from statistics import mean, pstdev

def load_data_by_file(dir_path):
    """
    Walk through all .json files in dir_path and collect values per key, per file.
    Returns a dict:
      { filename1: { metric: [v1, v2, …], … },
        filename2: { … }, … }
    """
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"{dir_path!r} is not a directory")

    data_by_file = {}
    for json_file in sorted(dir_path.glob("*.json")):
        try:
            records = json.loads(json_file.read_text())
        except Exception as e:
            print(f"[warning] could not read {json_file.name}: {e}")
            continue

        if not isinstance(records, list):
            print(f"[warning] {json_file.name} does not contain a JSON array; skipping")
            continue

        metrics = defaultdict(list)
        for rec in records:
            if not isinstance(rec, dict):
                continue
            for key, val in rec.items():
                try:
                    metrics[key].append(float(val))
                except (ValueError, TypeError):
                    # skip anything non‐numeric
                    pass

        if metrics:
            data_by_file[json_file.name] = metrics

    return data_by_file

def compute_stats(data):
    """
    Given a dict { key: [values…] }, compute mean & population stddev for each key.
    Returns { key: {"mean":…, "stddev":…}, … }.
    """
    stats = {}
    for key, vals in data.items():
        if not vals:
            continue
        stats[key] = {
            "mean": mean(vals),
            "stddev": pstdev(vals)
        }
    return stats

def main():
    p = argparse.ArgumentParser(
        description="Compute per-file mean & stddev for JSON metrics"
    )
    p.add_argument(
        "directory",
        help="Path to directory containing .json files"
    )
    args = p.parse_args()

    data_by_file = load_data_by_file(args.directory)
    if not data_by_file:
        print("No valid JSON data found.")
        return

    # Header
    print(f"{'File':<20} {'Metric':<20} {'Mean':>12} {'StdDev':>12}")
    print("-" * 68)

    # Rows
    for filename, metrics in data_by_file.items():
        stats = compute_stats(metrics)
        for metric in sorted(stats):
            m = stats[metric]["mean"]
            s = stats[metric]["stddev"]
            print(f"{filename:<20} {metric:<20} {m:12.6f} {s:12.6f}")

if __name__ == "__main__":
    main()
