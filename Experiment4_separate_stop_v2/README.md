# Experiment 4 — Decoupled Vision + Separate Stop Head

This README only notes how Experiment 4 differs from the base (Experiment 1) and Experiment 2.

Differences vs Experiment 1/2

- Stop head has its **own CNN backbone**; it only sees the current crop (channel 0).
- Movement head remains on the actor backbone; stop head is independent (helps with sparse stop labels).

Run (same scripts as base)

- Training: `./run_train.sh --experiment_name exp4_decoupled --curve_config config/curve_config.json`
- Inference: `./run_rollout.sh --image_path <img> --actor_weights <weights> --max_steps 1000`

More details: `../docs/DECOUPLED_STOP_EXPLAINED.md`
