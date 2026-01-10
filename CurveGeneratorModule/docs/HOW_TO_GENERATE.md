# How to Generate Curves

## Overview

You have two curve generators available:

1. **`CurveMakerFlexible`** - Single-segment Bezier curves (shorter, simpler)
2. **`CurveMakerMultiSegment`** - Multi-segment Bezier curves (longer, more complex)

## Quick Start

### Option 1: Generate Stage Grids (Recommended for Experiments)

This generates 7x7 grids (49 curves) for each stage in your config JSON.

**The script automatically generates BOTH single-segment and multi-segment curves:**

```bash
cd CurveGeneratorModule/ExampleConfigs
./generate.sh strong_foundation
```

This will create:
- `outputs/strong_foundation/single_segment/` - Single-segment curves
- `outputs/strong_foundation/multi_segment/` - Multi-segment curves

**With custom multi-segment settings:**
```bash
./generate.sh strong_foundation --num-segments 3 --segment-length-factor 1.2
```

### Option 2: Run Example Scripts

**Single-Segment Examples:**
```bash
cd CurveGeneratorModule/examples/single_segment
source ../../venv/bin/activate
python3 example_single_segment.py
```

**Multi-Segment Examples:**
```bash
cd CurveGeneratorModule/examples/multi_segment
source ../../venv/bin/activate
python3 example_multi_segment.py
```

## Which Generator Should I Use?

### Use Single-Segment (`CurveMakerFlexible`) when:
- ✅ You want shorter, simpler curves
- ✅ Training early stages (foundation)
- ✅ Need faster generation
- ✅ Curves don't need to cover the whole image

### Use Multi-Segment (`CurveMakerMultiSegment`) when:
- ✅ You want longer, more complex curves
- ✅ Curves should cover more of the image
- ✅ Training advanced stages
- ✅ Need winding, complex paths
- ✅ Want curves that traverse the entire image

## Current Setup

### For Your Experiments (`ExampleConfigs/`)

The `generate.sh` script **automatically generates both** single-segment and multi-segment curves:

```bash
cd ExampleConfigs
./generate.sh strong_foundation
```

This creates:
- `outputs/strong_foundation/single_segment/` - Single-segment Bezier curves
- `outputs/strong_foundation/multi_segment/` - Multi-segment Bezier curves (longer curves)

Both generators use the same config file, so you can easily compare the results side-by-side.

## Output Locations

- **Single-segment stage grids**: `ExampleConfigs/outputs/<config_name>/single_segment/stage<N>_<StageName>/grid_7x7.png`
- **Multi-segment stage grids**: `ExampleConfigs/outputs/<config_name>/multi_segment/stage<N>_<StageName>/grid_7x7.png`
- **Single-segment examples**: `examples/single_segment/outputs/`
- **Multi-segment examples**: `examples/multi_segment/outputs/`

## Configuration

Both generators use the same config JSON format. The multi-segment generator can optionally use:

```json
{
  "multi_segment": {
    "num_segments_range": [2, 4]
  }
}
```

This controls the default number of segments when `num_segments=None`.

## Tips

1. **Start with single-segment** for early training stages
2. **Switch to multi-segment** for later stages that need longer paths
3. **Use `num_segments`** parameter to control curve length:
   - `num_segments=2` → Short multi-segment curves
   - `num_segments=4` → Long multi-segment curves
   - `num_segments=None` → Random from config range
4. **Use `segment_length_factor`** to make segments longer:
   - `segment_length_factor=1.0` → Normal length
   - `segment_length_factor=1.5` → 50% longer segments

