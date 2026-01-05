# Experiment 3 — Refine

How this differs from the base (Experiment 1):

- Same model/stop setup as the base.
- Rewards: same magnitudes as the base; changes are in how path continuity is enforced.
- Curve generation: unchanged from Experiment 1.

Run (same scripts as base):

- Training: `./run_train.sh --experiment_name exp3_refine`
- Inference: `./run_rollout.sh --image_path <img> --actor_weights <weights>`

Additional differences (train.py):

- Windowed nearest-point search around expected progress to prevent path jumping.
- Path-jump penalties: penalize large index+spatial jumps; terminate with strong penalty for jumps to a different path.
- Large backward jumps are penalized; small backward jumps on the same path get only small penalty.
- Checkpoint auto-cleanup at end removed (keeps checkpoints).

Curve generator:

- No changes vs Experiment 1.
