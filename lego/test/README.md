# Test Rollout

This folder contains a lightweight rollout script to test curve tracking on a single image.

## Prerequisites

- Activate your `dsa_rl` (or equivalent) environment.
- Ensure you have valid weights in:
  - `lego/test/final_weights/tracker_weights/experiment4_20260121.pth`
  - `lego/test/final_weights/stop_weights/stop_detector_finetune.pth`

## Run the test

From the repository root:

```bash
python3 lego/test/rollout.py --image_path /path/to/your/image.png
```

Optional overrides:

```bash
python3 lego/test/rollout.py \
  --image_path /path/to/your/image.png \
  --movement_weights /path/to/movement.pth \
  --stop_weights /path/to/stop.pth \
  --max_steps 1000 \
  --stop_threshold 0.7
```

## Usage notes

- The script opens a window. Click a **start** point and a **direction** point.
- The tracker follows the curve; the stop module decides when to stop.
