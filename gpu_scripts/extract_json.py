import os
import json
import sys

def extract_json_from_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
        
    # Find the start and end of the JSON object based on the first '{' and the last '}'
    start_index = content.find('{')
    end_index = content.rfind('}') + 1

    if start_index == -1 or end_index == -1 or start_index >= end_index:
        print(f"No valid JSON found in {file_path}")
        return None

    json_str = content[start_index:end_index]
    try:
        json_obj = json.loads(json_str)
        return json_obj
    except json.JSONDecodeError:
        print(f"Invalid JSON format in {file_path}")
        return None

def main(input_directory, output_file):
    json_objects = []

    for filename in os.listdir(input_directory):
        file_path = os.path.join(input_directory, filename)
        if os.path.isfile(file_path):
            json_obj = extract_json_from_file(file_path)
            if json_obj is not None:
                json_objects.append(json_obj)

    # Write the array of JSON objects to the output file
    with open(output_file, 'w') as out_file:
        json.dump(json_objects, out_file, indent=4)

    print(f"Successfully extracted JSON objects to {output_file}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_directory> <output_file>")
        sys.exit(1)
    
    input_directory = sys.argv[1]
    output_file = sys.argv[2]
    
    main(input_directory, output_file)
