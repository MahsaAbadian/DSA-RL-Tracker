# Experiment 5 — Reward Config Extracted

This README only notes how Experiment 5 differs from the base (Experiment 1).

Differences vs Experiment 1
- Same model/architecture as the base.
- All reward terms are externalized to `config/reward_config.json` and snapshotted per run (also copied to `runs/.../reward_config.json`).
- Optional quick curriculum for smoke tests: `config/curve_config_quick.json` (10 episodes per stage) to validate setup.

Run (same scripts as base)
- Training (full): `./run_train.sh --experiment_name exp5_reward --curve_config config/curve_config.json --reward_config config/reward_config.json`
- Training (quick): `./run_train.sh --experiment_name exp5_quick --curve_config config/curve_config_quick.json --reward_config config/reward_config.json`
- Inference: `./run_rollout.sh --image_path <img> --actor_weights <weights>`
