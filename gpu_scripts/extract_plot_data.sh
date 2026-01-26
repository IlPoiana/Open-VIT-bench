#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_directory> <component_name>"
    echo "Example: $0 ./benchmark_results mlp"
    exit 1
fi

INPUT_DIR="$1"
COMPONENT_NAME="$2"

# Create the output directory structure
OUTPUT_BASE_DIR="json/${COMPONENT_NAME}"
mkdir -p "${OUTPUT_BASE_DIR}"

# Define output file paths
PARAMS_SEARCH_JSON="${OUTPUT_BASE_DIR}/params_search.json"
BEST_PARAMS_JSON="${OUTPUT_BASE_DIR}/best_params.json"

echo "========================================="
echo "Benchmark Analysis Pipeline"
echo "========================================="
echo "Input directory: ${INPUT_DIR}"
echo "Component: ${COMPONENT_NAME}"
echo "Output directory: ${OUTPUT_BASE_DIR}"
echo "========================================="

# Step 1: Extract JSON from files
echo ""
echo "[Step 1/3] Extracting JSON from files..."
python gpu_scripts/extract_json.py "${INPUT_DIR}" "${PARAMS_SEARCH_JSON}"

if [ $? -ne 0 ]; then
    echo "Error: extract_json.py failed"
    exit 1
fi

echo "✓ JSON extraction complete: ${PARAMS_SEARCH_JSON}"

# Step 2: Find best parameters
echo ""
echo "[Step 2/3] Finding best parameters..."
python gpu_scripts/find_best_params.py "${COMPONENT_NAME}" "${PARAMS_SEARCH_JSON}" "${BEST_PARAMS_JSON}"

if [ $? -ne 0 ]; then
    echo "Error: find_best_params.py failed"
    exit 1
fi

echo "✓ Best parameters found: ${BEST_PARAMS_JSON}"

# Step 3: Plot best parameters
echo ""
echo "[Step 3/3] Plotting best parameters..."
python gpu_scripts/plot_best_parameters.py "${BEST_PARAMS_JSON}"

if [ $? -ne 0 ]; then
    echo "Error: plot_best_parameters.py failed"
    exit 1
fi

echo "✓ Plotting complete"

echo ""
echo "========================================="
echo "Data extracted successfully!"
echo "========================================="
echo "Results saved in: ${OUTPUT_BASE_DIR}"
echo "  - params_search.json"
echo "  - best_params.json"
echo "========================================="