# Experiment 2 — Shared Backbone + Separate Stop Head

This experiment is identical to **Experiment 1** except for the policy head design:
- Movement head: 8 movement actions (no stop action in the movement logits).
- Separate stop head: binary logit for STOP, fed by the **same shared CNN+LSTM backbone**.

Quick run (same scripts as base):
- Training: `./run_train.sh --experiment_name exp2_shared`
- Inference: `./run_rollout.sh --image_path <img> --actor_weights <weights>`
