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

    # Group entries by dataset_size (batch_n * batch_size)
    dataset_groups = defaultdict(list)
    
    for entry in data:
        batch_n = entry["batch_n"]
        batch_size = entry ["batch_size"] #entry["minibatch_size"] 
        dataset_size = batch_n * batch_size
        total_time = entry["time"]["total_time"]
        
        dataset_groups[dataset_size].append({
            "total_time": total_time,
            "params": entry["params"],
            "entry": entry
        })
    
    # For each dataset size, find top 3 best times
    dataset_sizes = []
    best_times = []
    param_counters = defaultdict(Counter)
    top3_results = []  # Store top 3 for JSON export
    
    for dataset_size in sorted(dataset_groups.keys()):
        entries = dataset_groups[dataset_size]
        
        # Sort by total_time (ascending) and get top 3
        entries_sorted = sorted(entries, key=lambda x: x["total_time"])
        top3 = entries_sorted[:3]
        
        # Get the best time (minimum)
        best_time = top3[0]["total_time"]
        
        dataset_sizes.append(dataset_size)
        best_times.append(best_time)
        
        # Store top 3 results for this dataset size
        top3_results.append({
            "dataset_size": dataset_size,
            "top3": [item["entry"] for item in top3]
        })
        
        # Count parameter frequencies in top 3
        for item in top3:
            for param_name, param_value in item["params"].items():
                param_counters[param_name][param_value] += 1
    
    # ---- Save top 3 results to JSON ----
    base_name = os.path.splitext(json_path)[0]
    output_path = f"{base_name}_top3.json"
    with open(output_path, "w") as f:
        json.dump(top3_results, f, indent=4)
    print(f"Saved top 3 results to: {output_path}")
    
    # Plot
    label = os.path.basename(json_path)
    plt.plot(dataset_sizes, best_times, marker="o", label=label)
    
    # ---- Print parameter stats per file ----
    print(f"\n=== Parameter frequencies for {label} ===\n")
    for param_name, counter in param_counters.items():
        print(f"{param_name}:")
        for value, freq in counter.most_common():
            print(f"  {value}: {freq}")
        print()

# ---- Final plot formatting ----
plt.xlabel("Dataset size (batch_n × batch_size)")
plt.ylabel("Best time (total_time)")
plt.title("Best time vs dataset size (comparison)")
plt.grid(True)
plt.xscale("log", base=2)
plt.legend()
plt.tight_layout()
plt.show()