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

## Stop Model Details (How It Works)

The stop model is **part of the RL policy** and is trained jointly with PPO. It is not a separate post-processor.

### Architecture Summary

- **Movement Head**: Categorical policy over 8 movement actions.
- **Stop Head**: Bernoulli policy (stop vs continue) with its own CNN backbone.
- **Shared History**: The stop head consumes the **actor LSTM output** to use recent motion context.
- **Input Channels for Stop**: Image crop (channel 0) + path mask (channel 3), so the model can detect endpoints and avoid false stops on already-visited paths.

### Action Sampling Logic

At each step, the policy samples a **stop decision first**:

- This is sampled from the stop head’s Bernoulli distribution (`stop_prob = sigmoid(stop_logit)`, then `torch.bernoulli(stop_prob)`), in `src/train.py` and produced by `DecoupledStopBackboneActorCritic` in `src/models.py`.
- If `stop` is sampled, the environment receives the **stop action index**.
- If `continue` is sampled, the policy then samples **one of the 8 movement actions** from a categorical distribution over movement logits.

This makes stop a **separate decision** from movement direction, which stabilizes training when stop events are rare.

### PPO Objective (Combined Stop + Move)

The PPO log-prob is constructed as:

- **Stop action**: `log p(stop)`
- **Move action**: `log p(continue) + log p(move | continue)`

This lets the stop head participate directly in PPO optimization.

### Supervised Stop Signal (Auxiliary Loss)

In addition to PPO, the stop head receives a **binary supervision label**:

- `stop_label = 1` when the agent is **near the end** of the curve (even if it does not stop)
- `stop_label = 0` otherwise

This adds a BCE loss term (weighted by `lambda_stop`) to reduce class imbalance and improve stopping reliability.

## Setup

Experiment 4 uses the central CurveGeneratorModule, which requires additional dependencies. Set up the environment:

**Option 1: Using conda environment (Recommended)**

```bash
# From repository root
conda env create -f environment.yml
conda activate dsa_rl
```

**Option 2: Manual installation**

```bash
# Activate your Python environment, then:
pip install numpy scipy opencv-python torch torchvision
```

The scripts will automatically check for dependencies and provide helpful error messages if packages are missing.

## Curve Generation

**Experiment 4 uses the central CurveGeneratorModule** for all curve generation:

- **Generator**: `CenterlineMask5PointsGenerator` - Creates smooth 5-point Bezier curves with a mandatory 1-pixel centerline mask.
- **Config**: `strong_foundation_config.json` from `CurveGeneratorModule/ExampleConfigs/`
- **Benefits**:
  - Centralized, maintainable curve generation code
  - Consistent curve generation across experiments
  - **1-pixel Centerline Masks**: Forces the agent to learn high-precision tracking by using 1px masks regardless of visual vessel width.

The curve configuration is loaded from the central module and merged with experiment-specific settings. Stage-specific parameters (width, intensity, curvature, etc.) are still controlled via the training config, but the underlying curve generation uses the shared module.

## Run

- Training: `./run_train.sh --experiment_name exp4_v2 --curve_config config/curve_config.json`
- Inference: `./run_rollout.sh --image_path <img> --actor_weights <weights> --max_steps 1000`

For a deep dive into the theory, see: `docs/DECOUPLED_STOP_EXPLAINED.md`
