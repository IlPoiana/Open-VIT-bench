import os
import json
import re
import sys

def extract_json_block(text: str) -> str:
    """
    Extracts the JSON object from the file (from first { to last }).
    """
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No valid JSON block found")
    return text[start:end + 1]

def remove_trailing_commas(json_text: str) -> str:
    """
    Removes trailing commas before } or ]
    """
    # Remove commas followed by optional whitespace and then } or ]
    return re.sub(r',\s*(\}|\])', r'\1', json_text)

def extract_json_from_directory(input_dir: str, output_file: str):
    extracted_objects = []
    errors = []

    for filename in os.listdir(input_dir):
        if not filename.endswith(".out"):
            continue

        filepath = os.path.join(input_dir, filename)

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            json_block = extract_json_block(content)
            cleaned_json = remove_trailing_commas(json_block)

            # Validate JSON
            obj = json.loads(cleaned_json)
            extracted_objects.append(obj)

        except Exception as e:
            errors.append((filename, str(e)))

    # Write aggregated JSON array
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(extracted_objects, f, indent=2)

    print(f"✅ Extracted {len(extracted_objects)} JSON objects")
    if errors:
        print(f"⚠️  Errors in {len(errors)} files:")
        for fname, err in errors:
            print(f"   - {fname}: {err}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_json.py <input_directory> <output_file.json>")
        sys.exit(1)

    extract_json_from_directory(sys.argv[1], sys.argv[2])
