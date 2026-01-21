import json
import sys
from collections import defaultdict

filter_ln   = "time"
filter_mlp  = "kernel"
filter_pe   = "total"
filter_block= "total_time"
filter_ph   = "kernel_time"
filter_vit  = "total_time"

def compute_top3(batch_results):
    output_data = []

    # Process each batch to find the best times
    for batch, entries in batch_results.items():
        # Sort entries by time (ascending)
        sorted_entries = sorted(entries, key=lambda x: x['time'])
        
        # Get the top three entries
        top3 = sorted_entries[:3]        
        
        # Append to output_data
        output_data.append({
            'batch': batch,
            'top3': top3,
        })
    return output_data

def process_mlp_batches(input_file, output_file):
    # Read the input JSON file
    with open(input_file, 'r') as file:
        data = json.load(file)

    # Dictionary to hold the batch results
    batch_results = defaultdict(list)

    # Organize the data by batch
    for entry in data:
        batch = entry['batch']
        params = entry['params']
        time = entry['time']
        total_time = entry['time']["kernel"] + entry['time']["transpose"] 
        batch_results[batch].append({'params': params, 'time': total_time, "time_composition": time})
    
    # Prepare the output structure
    output_data = compute_top3(batch_results)

    # Write the results to the output JSON file
    with open(output_file, 'w') as file:
        json.dump(output_data, file, indent=4)

def process_ln_batches(input_file, output_file):
    # Read the input JSON file
    with open(input_file, 'r') as file:
        data = json.load(file)

    # Dictionary to hold the batch results
    batch_results = defaultdict(list)

    # Organize the data by batch
    for entry in data:
        batch = entry['batch']
        params = entry['params']
        time = entry['time']["time"]
        batch_results[batch].append({'params': params, 'time': time})
        
    # Prepare the output structure
    output_data = compute_top3(batch_results)

    # Write the results to the output JSON file
    with open(output_file, 'w') as file:
        json.dump(output_data, file, indent=4)

def process_pe_batches(input_file, output_file):
    # Read the input JSON file
    with open(input_file, 'r') as file:
        data = json.load(file)

    # Dictionary to hold the batch results
    batch_results = defaultdict(list)

    # Organize the data by batch
    for entry in data:
        batch = entry['batch']
        params = entry['params']
        total_time = entry['time']["total"]
        time = entry['time']
        batch_results[batch].append({'params': params, 'time': total_time, "time_composition": time})
    
    # Prepare the output structure
    output_data = compute_top3(batch_results)

    # Write the results to the output JSON file
    with open(output_file, 'w') as file:
        json.dump(output_data, file, indent=4)

def process_block_batches(input_file, output_file):
    # Read the input JSON file
    with open(input_file, 'r') as file:
        data = json.load(file)

    # Dictionary to hold the batch results
    batch_results = defaultdict(list)

    # Organize the data by batch
    for entry in data:
        batch = entry['batch']
        params = entry['params']
        time = entry['time']
        total_time = entry['time']["total_time"]
        batch_results[batch].append({'params': params, 'time': total_time, "time_composition": time})
    
    # Prepare the output structure
    output_data = compute_top3(batch_results)

    # Write the results to the output JSON file
    with open(output_file, 'w') as file:
        json.dump(output_data, file, indent=4)
    
def process_ph_batches(input_file, output_file):
    # Read the input JSON file
    with open(input_file, 'r') as file:
        data = json.load(file)

    # Dictionary to hold the batch results
    batch_results = defaultdict(list)

    # Organize the data by batch
    for entry in data:
        batch = entry['batch']
        params = entry['params']
        time = entry['time']
        total_time = entry['time']["kernel_time"]
        batch_results[batch].append({'params': params, 'time': total_time, "time_composition": time})
    
    # Prepare the output structure
    output_data = compute_top3(batch_results)

    # Write the results to the output JSON file
    with open(output_file, 'w') as file:
        json.dump(output_data, file, indent=4)

def process_vit_batches(input_file, output_file):
     # Read the input JSON file
    with open(input_file, 'r') as file:
        data = json.load(file)

    # Dictionary to hold the batch results
    batch_results = defaultdict(list)

    # Organize the data by batch
    for entry in data:
        batch = entry['batch'] * entry['batch_n']
        params = entry['params']
        time = entry['time']
        total_time = entry['time']["total_time"]
        batch_results[batch].append({'params': params, 'time': total_time, "time_composition": time})
    
    # Prepare the output structure
    output_data = compute_top3(batch_results)

    # Write the results to the output JSON file
    with open(output_file, 'w') as file:
        json.dump(output_data, file, indent=4)




if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <component> <input_json_file> <output_json_file>")
        sys.exit(1)

    component = sys.argv[1]
    input_json_file = sys.argv[2]
    output_json_file = sys.argv[3]

    if(component == "ln"):
        process_ln_batches(input_json_file, output_json_file)
    if(component == "mlp"):
        process_mlp_batches(input_json_file, output_json_file)
    if(component == "pe"):
        process_pe_batches(input_json_file, output_json_file)
    if(component == "block"):
        process_block_batches(input_json_file, output_json_file)
    if(component == "ph"):
        process_ph_batches(input_json_file, output_json_file)
    if(component == "vit"):
        process_vit_batches(input_json_file, output_json_file)

    
