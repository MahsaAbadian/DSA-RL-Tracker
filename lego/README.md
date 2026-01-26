## Lego

This folder contains small test utilities that "patch" different model
components together for evaluation or quick experiments. The goal is to
mix-and-match trained weights (for example, movement from one model and
stop detection from another) without changing training code.

### Files

- `rollout_hybrid_stop.py`: Rollout-style inference that uses 8-action
  movement weights and a standalone stop module to decide when to stop.

### How to run

```bash
python3 Lego/rollout_hybrid_stop.py \
  --image_path /path/to/image.png \
  --movement_weights /path/to/movement_weights.pth \
  --stop_weights /path/to/stop_weights.pth
```

Optional flags:
- `--stop_threshold 0.7` (default)
- `--max_steps 1000` (default)
