#!/bin/bash
# Generate example curves per training stage.
# Prompts for number of examples per stage and saves to runs/stage_examples/.

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

read -p "Number of examples per stage [default 5]: " NUM
NUM=${NUM:-5}

echo "Generating $NUM examples per stage..."
python -u src/curve_generator.py --stage_examples "$NUM" --render_stage_examples --render_cols 5

echo "Done. Grids saved to runs/stage_examples/<stage>/grid.png"

