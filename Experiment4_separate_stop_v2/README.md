# Experiment 4 (v2) — Improved Decoupled Stop Architecture

This experiment builds on Experiment 2 by giving the **Stop Head** its own dedicated vision backbone and specialized context to solve the "never-stopping" problem.

## Recent Improvements (v2)

### 1. Enhanced Stop Backbone
- **Dual-Channel Input**: The Stop Head now sees both the **Current Image Crop** (Channel 0) and the **Path Mask** (Channel 3). Knowing where the agent has already been is critical for identifying true endpoints.
- **Deeper Vision**: The stop backbone was deepened with **dilated convolutions** (3 layers, dilations up to 3). This significantly increases the receptive field, allowing the model to see more vessel context within the 33x33 patch.

### 2. Multi-Task Context Integration
- **LSTM Context**: The Stop Head now receives the output of the **Actor LSTM**. It can now use its recent movement history (direction and speed) to help decide if it has run out of "track."

### 3. Training Optimization
- **Weighted Stop Loss**: The `lambda_stop` (weight for the stop classification loss) was increased from **1.0 to 5.0**. This compensates for the massive label imbalance (rare stops) and forces the network to prioritize learning the stopping visual signature.

## Differences vs Experiment 1/2

- **Dedicated Backbone**: Stop head has its own CNN backbone, separate from the movement head.
- **Context Isolation**: The stop logic is decoupled from navigation, preventing confusion in noisy or blurry sections of the image.

## Run (same scripts as base)

- Training: `./run_train.sh --experiment_name exp4_v2 --curve_config config/curve_config.json`
- Inference: `./run_rollout.sh --image_path <img> --actor_weights <weights> --max_steps 1000`

For a deep dive into the theory, see: `docs/DECOUPLED_STOP_EXPLAINED.md`
