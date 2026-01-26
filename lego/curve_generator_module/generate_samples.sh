#!/bin/bash
# Script to generate visual samples of curve configurations

# 1. Ask for config path
read -p "Enter the path to the curve configuration file (e.g., Experiment4_separate_stop_v2/config/curve_config.json): " CONFIG_PATH

# 2. Check if file exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: File '$CONFIG_PATH' not found."
    exit 1
fi

# 3. Use the Python sampler
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use conda environment 'rl' if available, otherwise standard python3
if [ -d "/Users/mahsaabadian/miniconda3/envs/rl" ]; then
    PYTHON_CMD="/Users/mahsaabadian/miniconda3/envs/rl/bin/python3"
else
    PYTHON_CMD="python3"
fi

echo "🚀 Generating visual samples using $CONFIG_PATH..."
$PYTHON_CMD "$SCRIPT_DIR/visual_sampler.py" "$CONFIG_PATH"

echo "✅ Generation complete. Check the 'CurveGeneratorModule/sample_curve/' directory."
