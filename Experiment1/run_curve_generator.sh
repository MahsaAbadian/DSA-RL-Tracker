#!/bin/bash
# Bash script to run curve generator separately
# Generates curves for all 3 curriculum stages

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Generating Synthetic Curves for All Stages"
echo "Script directory: $SCRIPT_DIR"
echo "=========================================="
echo ""

# Default values
NUM_CURVES=${1:-1000}
OUTPUT_DIR=${2:-generated_curves}

echo "Generating $NUM_CURVES curves per stage..."
echo "Output base directory: $OUTPUT_DIR"
echo ""

# Run curve generator for all stages
python3 curve_generator.py \
    --output_dir "$OUTPUT_DIR" \
    --num_curves "$NUM_CURVES" \
    --h 128 \
    --w 128 \
    --all_stages

echo ""
echo "=========================================="
echo "Curve generation completed!"
echo "Curves saved to:"
echo "  - $OUTPUT_DIR/stage1/ (Simple curves)"
echo "  - $OUTPUT_DIR/stage2/ (Medium curves)"
echo "  - $OUTPUT_DIR/stage3/ (Complex curves)"
echo "=========================================="

