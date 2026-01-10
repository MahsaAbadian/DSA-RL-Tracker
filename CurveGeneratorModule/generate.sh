#!/bin/bash
# Generate 7x7 grids for curve configs
# Usage: ./generate.sh [config_name] [--num-segments N] [--segment-length-factor F]
#   - If config_name is provided: generates grids for that config only
#   - If no argument: generates grids for all configs
#   - Generates BOTH single-segment and multi-segment curves
#   - Results go to separate folders: outputs/<config>/single_segment/ and outputs/<config>/multi_segment/
#   - --num-segments N: Number of segments (for multi-segment, default: random)
#   - --segment-length-factor F: Segment length factor (default: 1.0)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use virtual environment Python if available, otherwise use system Python
if [ -f "$SCRIPT_DIR/venv/bin/python3" ]; then
    PYTHON_CMD="$SCRIPT_DIR/venv/bin/python3"
elif [ -f "$SCRIPT_DIR/venv/bin/python" ]; then
    PYTHON_CMD="$SCRIPT_DIR/venv/bin/python"
else
    PYTHON_CMD="python3"
fi

# Config files are in ExampleConfigs or curveexperiments folder
if [ -d "$SCRIPT_DIR/ExampleConfigs" ]; then
    CONFIG_DIR="$SCRIPT_DIR/ExampleConfigs"
elif [ -d "$SCRIPT_DIR/curveexperiments" ]; then
    CONFIG_DIR="$SCRIPT_DIR/curveexperiments"
else
    CONFIG_DIR="$SCRIPT_DIR"
fi
OUTPUT_BASE="$CONFIG_DIR/outputs"
MODULE_DIR="$(dirname "$SCRIPT_DIR")"
if [ -d "$CONFIG_DIR/ExampleConfigs" ]; then
    CONFIG_DIR="$CONFIG_DIR/ExampleConfigs"
elif [ -d "$CONFIG_DIR/curveexperiments" ]; then
    CONFIG_DIR="$CONFIG_DIR/curveexperiments"
else
    CONFIG_DIR="$CONFIG_DIR"
fi

# Find all config JSON files
CONFIG_FILES=("$CONFIG_DIR"/*_config.json)

if [ ${#CONFIG_FILES[@]} -eq 0 ] || [ ! -f "${CONFIG_FILES[0]}" ]; then
    echo "⚠️  No config files found in $CONFIG_DIR"
    echo "   Looking for files matching: *_config.json"
    exit 1
fi

# Parse arguments
CONFIG_NAME=""
NUM_SEGMENTS_FLAG="--num-segments 2"  # Default to 2 segments
SEGMENT_LENGTH_FACTOR_FLAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --num-segments)
            NUM_SEGMENTS_FLAG="--num-segments $2"
            shift 2
            ;;
        --segment-length-factor)
            SEGMENT_LENGTH_FACTOR_FLAG="--segment-length-factor $2"
            shift 2
            ;;
        *)
            if [ -z "$CONFIG_NAME" ]; then
                CONFIG_NAME="$1"
            fi
            shift
            ;;
    esac
done

# If config name is provided, generate only for that config
if [ -n "$CONFIG_NAME" ]; then
    
    # Handle both "experiment1" and "experiment1_config" formats
    if [[ "$CONFIG_NAME" == *_config ]]; then
        # Already has _config suffix
        CONFIG_FILE="$CONFIG_DIR/${CONFIG_NAME}.json"
    else
        # Add _config suffix
        CONFIG_FILE="$CONFIG_DIR/${CONFIG_NAME}_config.json"
    fi
    
    # Check if config file exists
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "❌ Config file not found: $CONFIG_FILE"
        echo ""
        echo "Available configs:"
        ls -1 "$CONFIG_DIR"/*_config.json 2>/dev/null | sed 's|.*/||' | sed 's|_config.json||' | sed 's|^|  - |'
        exit 1
    fi
    
    # Use base name for output directory (remove _config suffix if present)
    OUTPUT_BASE_NAME="${CONFIG_NAME%_config}"
    OUTPUT_DIR_SINGLE="$OUTPUT_BASE/$OUTPUT_BASE_NAME/single_segment"
    OUTPUT_DIR_MULTI="$OUTPUT_BASE/$OUTPUT_BASE_NAME/multi_segment"
    OUTPUT_DIR_SIXPOINT="$OUTPUT_BASE/$OUTPUT_BASE_NAME/six_point"
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 Generating grids for: $OUTPUT_BASE_NAME"
    echo "   Config: $CONFIG_FILE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Generate single-segment curves
    echo ""
    echo "📐 Generating SINGLE-SEGMENT curves..."
    echo "   Output: $OUTPUT_DIR_SINGLE"
    $PYTHON_CMD "$SCRIPT_DIR/generate_stage_grids.py" \
        --config "$CONFIG_FILE" \
        --output "$OUTPUT_DIR_SINGLE" \
        --grid-size 7 \
        --seed 42
    
    SINGLE_EXIT_CODE=$?
    
    # Generate multi-segment curves
    echo ""
    echo "📐 Generating MULTI-SEGMENT curves (2 segments)..."
    echo "   Output: $OUTPUT_DIR_MULTI"
    $PYTHON_CMD "$SCRIPT_DIR/generate_stage_grids.py" \
        --config "$CONFIG_FILE" \
        --output "$OUTPUT_DIR_MULTI" \
        --grid-size 7 \
        --seed 42 \
        --multi-segment \
        $NUM_SEGMENTS_FLAG \
        $SEGMENT_LENGTH_FACTOR_FLAG
    
    MULTI_EXIT_CODE=$?
    
    # Generate six-point curves
    echo ""
    echo "📐 Generating SIX-POINT curves..."
    echo "   Output: $OUTPUT_DIR_SIXPOINT"
    $PYTHON_CMD "$SCRIPT_DIR/generate_stage_grids.py" \
        --config "$CONFIG_FILE" \
        --output "$OUTPUT_DIR_SIXPOINT" \
        --grid-size 7 \
        --seed 42 \
        --six-point
    
    SIXPOINT_EXIT_CODE=$?
    
    if [ $SINGLE_EXIT_CODE -eq 0 ] && [ $MULTI_EXIT_CODE -eq 0 ] && [ $SIXPOINT_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✅ Successfully generated grids for $OUTPUT_BASE_NAME"
        echo "   Single-segment: $OUTPUT_DIR_SINGLE"
        echo "   Multi-segment:  $OUTPUT_DIR_MULTI"
        echo "   Six-point:      $OUTPUT_DIR_SIXPOINT"
    else
        echo ""
        if [ $SINGLE_EXIT_CODE -ne 0 ]; then
            echo "❌ Failed to generate single-segment grids"
        fi
        if [ $MULTI_EXIT_CODE -ne 0 ]; then
            echo "❌ Failed to generate multi-segment grids"
        fi
        if [ $SIXPOINT_EXIT_CODE -ne 0 ]; then
            echo "❌ Failed to generate six-point grids"
        fi
        exit 1
    fi
else
    # No argument provided - generate for all configs
    echo "Found ${#CONFIG_FILES[@]} config file(s)"
    echo ""
    
    # Generate grids for each config
    for config_file in "${CONFIG_FILES[@]}"; do
        config_name=$(basename "$config_file" .json)
        # Remove _config suffix if present
        base_name="${config_name%_config}"
        output_dir_single="$OUTPUT_BASE/$base_name/single_segment"
        output_dir_multi="$OUTPUT_BASE/$base_name/multi_segment"
        output_dir_sixpoint="$OUTPUT_BASE/$base_name/six_point"
        
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "📊 Processing: $base_name"
        echo "   Config: $config_file"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        # Generate single-segment curves
        echo ""
        echo "📐 Generating SINGLE-SEGMENT curves..."
        $PYTHON_CMD "$SCRIPT_DIR/generate_stage_grids.py" \
            --config "$config_file" \
            --output "$output_dir_single" \
            --grid-size 7 \
            --seed 42
        
        SINGLE_EXIT=$?
        
        # Generate multi-segment curves
        echo ""
        echo "📐 Generating MULTI-SEGMENT curves (2 segments)..."
        $PYTHON_CMD "$SCRIPT_DIR/generate_stage_grids.py" \
            --config "$config_file" \
            --output "$output_dir_multi" \
            --grid-size 7 \
            --seed 42 \
            --multi-segment \
            $NUM_SEGMENTS_FLAG \
            $SEGMENT_LENGTH_FACTOR_FLAG
        
        MULTI_EXIT=$?
        
        # Generate six-point curves
        echo ""
        echo "📐 Generating SIX-POINT curves..."
        $PYTHON_CMD "$SCRIPT_DIR/generate_stage_grids.py" \
            --config "$config_file" \
            --output "$output_dir_sixpoint" \
            --grid-size 7 \
            --seed 42 \
            --six-point
        
        SIXPOINT_EXIT=$?
        
        if [ $SINGLE_EXIT -eq 0 ] && [ $MULTI_EXIT -eq 0 ] && [ $SIXPOINT_EXIT -eq 0 ]; then
            echo "✅ Successfully generated grids for $base_name"
            echo "   Single-segment: $output_dir_single"
            echo "   Multi-segment:  $output_dir_multi"
            echo "   Six-point:      $output_dir_sixpoint"
        else
            if [ $SINGLE_EXIT -ne 0 ]; then
                echo "❌ Failed to generate single-segment grids for $base_name"
            fi
            if [ $MULTI_EXIT -ne 0 ]; then
                echo "❌ Failed to generate multi-segment grids for $base_name"
            fi
            if [ $SIXPOINT_EXIT -ne 0 ]; then
                echo "❌ Failed to generate six-point grids for $base_name"
            fi
        fi
        echo ""
    done
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🎉 All done! Check outputs in: $OUTPUT_BASE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
fi
