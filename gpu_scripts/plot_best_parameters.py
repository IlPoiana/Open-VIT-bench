import json
import argparse
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import os

# ---- Parse command-line arguments ----
parser = argparse.ArgumentParser(
    description="Compare batch timing statistics across multiple JSON files"
)
parser.add_argument(
    "json_paths",
    nargs="+",
    help="Paths to JSON files containing benchmark results"
)
args = parser.parse_args()

# ---- Plot setup ----
plt.figure()

# ---- Process each JSON file ----
for json_path in args.json_paths:
    with open(json_path, "r") as f:
        data = json.load(f)

    batch_sizes = []
    best_times = []

    param_counters = defaultdict(Counter)

    for entry in data:
        batch = entry["batch"]
        top3 = entry["top3"]

        best_time = min(item["time"] for item in top3)

        batch_sizes.append(batch)
        best_times.append(best_time)

        for item in top3:
            for param_name, param_value in item["params"].items():
                param_counters[param_name][param_value] += 1

    # Sort for plotting
    batch_sizes, best_times = zip(*sorted(zip(batch_sizes, best_times)))

    label = os.path.basename(json_path)
    plt.plot(batch_sizes, best_times, marker="o", label=label)

    # ---- Print parameter stats per file ----
    print(f"\n=== Parameter frequencies for {label} ===\n")
    for param_name, counter in param_counters.items():
        print(f"{param_name}:")
        for value, freq in counter.most_common():
            print(f"  {value}: {freq}")
        print()

# ---- Final plot formatting ----
plt.xlabel("Batch size")
plt.ylabel("Best time")
plt.title("Best time vs batch size (comparison)")
plt.grid(True)
plt.xscale("log", base=2)
plt.legend()
plt.tight_layout()
plt.show()
