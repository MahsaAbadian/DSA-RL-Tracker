# Stop Module

A standalone binary classifier that detects when a curve-tracking agent should stop. This module is trained separately from the tracker and provides a stop signal during inference.

## Overview

The stop module is a **supervised learning** model that takes a 2-channel 33×33 crop and outputs a single logit indicating whether the agent should stop at the current location.

### Architecture

- **Input**: 2-channel 33×33 crop
  - Channel 0: Image crop (grayscale DSA image)
  - Channel 1: Path mask (binary mask showing where agent has been)
- **Backbone**: 3-layer CNN with GroupNorm and PReLU
  - Conv2d(2→32) → Conv2d(32→64) → Conv2d(64→128) → AdaptiveAvgPool2d
- **Head**: 2-layer MLP
  - Linear(128→64) → Linear(64→1)
- **Output**: Single logit → sigmoid → stop probability (0.0 to 1.0)

## Training

### Quick Start

```bash
# Activate your environment
conda activate dsa_rl  # or your environment

# Train with defaults
python src/train_standalone.py

# Or use the bash script
./run_train.sh
```

### Training Process

The training script (`train_standalone.py`) performs supervised learning:

1. **Data Generation** (on-the-fly, no pre-saved dataset):
   - Generates synthetic curves using `curve_generator_module`
   - For each curve, creates 2 samples:
     - **Positive (label=1.0)**: 33×33 crop at the **endpoint** → "STOP here"
     - **Negative (label=0.0)**: 33×33 crop at a **midpoint** → "DON'T stop here"
   - Default: 5,000 curves → 10,000 total samples (5k positive + 5k negative)

2. **Training Setup**:
   - Loss: `BCEWithLogitsLoss` (binary cross-entropy)
   - Optimizer: Adam with learning rate `1e-4`
   - Batch size: `64`
   - Epochs: `15` (default)
   - Train/Val split: 80%/20%

3. **Training Loop**:
   - For each epoch:
     - Train on batches, compute loss, backprop
     - Validate on held-out set
     - Save best model based on validation accuracy
   - Reports per-class metrics (stop accuracy, go accuracy)

### Training Logs Explained

During training, you'll see logs like:
```
Epoch 01/15 | T-Loss: 0.6234 | T-Acc: 65.23% | V-Acc: 68.45% (Stop: 72.1%, Go: 64.8%)
⭐ New Best Val Accuracy! Saved to stop_module/weights/stop_detector_v1.pth
```

Here's how each metric is calculated:

#### **T-Loss (Training Loss)**
- **Formula**: `train_loss / len(train_loader)`
- **Calculation**:
  - For each batch: `loss = BCEWithLogitsLoss(model(x), y)`
  - Accumulate: `train_loss += loss.item()` (line 224)
  - Average: Divide by number of batches (`len(train_loader)`)
- **What it means**: Average binary cross-entropy loss per batch during training
- **Lower is better**: Should decrease as training progresses

#### **T-Acc (Training Accuracy)**
- **Formula**: `train_correct / train_size`
- **Calculation**:
  - For each batch: `preds = (torch.sigmoid(logits) > 0.5).float()` (line 225)
  - Count correct: `train_correct += (preds == y).sum().item()` (line 226)
  - Accuracy: `train_correct / train_size` where `train_size = 0.8 * total_samples` (line 253)
- **What it means**: Percentage of training samples predicted correctly
- **Higher is better**: Should increase toward 100%

#### **V-Acc (Validation Accuracy)**
- **Formula**: `val_correct / val_size`
- **Calculation**:
  - Same as T-Acc but on validation set (held-out 20% of data)
  - `val_size = 0.2 * total_samples` (line 254)
- **What it means**: Percentage of validation samples predicted correctly
- **Higher is better**: Indicates generalization (not just memorization)

#### **Stop Accuracy (Per-Class: Positive Samples)**
- **Formula**: `pos_correct / pos_total`
- **Calculation**:
  - Filter validation samples where `y == 1.0` (positive/endpoint samples) (line 246)
  - Count correct predictions: `pos_correct += (preds[pos_mask] == y[pos_mask]).sum().item()` (line 248)
  - Total positive samples: `pos_total += pos_mask.sum().item()` (line 249)
  - Accuracy: `pos_correct / pos_total` (line 255)
- **What it means**: How well the model detects endpoints (should stop)
- **Higher is better**: Should be > 70% for good performance

#### **Go Accuracy (Per-Class: Negative Samples)**
- **Formula**: `neg_correct / neg_total`
- **Calculation**:
  - Filter validation samples where `y == 0.0` (negative/midpoint samples) (line 247)
  - Count correct predictions: `neg_correct += (preds[neg_mask] == y[neg_mask]).sum().item()` (line 250)
  - Total negative samples: `neg_total += neg_mask.sum().item()` (line 251)
  - Accuracy: `neg_correct / neg_total` (line 256)
- **What it means**: How well the model detects midpoints (should continue)
- **Higher is better**: Should be > 70% for good performance

#### **Best Model Saving**
- **Trigger**: `if val_acc > best_val_acc` (line 263)
- **Action**: Saves model weights to disk (line 265)
- **What it means**: Only saves when validation accuracy improves (prevents overfitting)

**Example Interpretation**:
- `T-Acc: 65.23%` → Model correctly classifies 65% of training samples
- `V-Acc: 68.45%` → Model correctly classifies 68% of validation samples
- `Stop: 72.1%` → Model correctly identifies 72% of endpoints
- `Go: 64.8%` → Model correctly identifies 65% of midpoints

**Good Training Signs**:
- T-Loss decreases over epochs
- T-Acc and V-Acc both increase
- V-Acc close to T-Acc (not much overfitting)
- Both Stop and Go accuracies > 70%

### Training Parameters

| Parameter | Default | Description | Where it comes from |
|-----------|---------|-------------|---------------------|
| `samples_per_class` | 5000 | Positive/negative samples per class | Default in `StopDataset.__init__()` |
| `epochs` | 15 | Training epochs | Default in argparse |
| `batch_size` | 64 | Batch size for training | Default in argparse |
| `learning_rate` | 1e-4 | Adam optimizer learning rate | Default in argparse |
| `CROP` | 33 | Crop size (33×33) | Constant in `train_standalone.py` |
| Train/Val split | 80/20 | Dataset split ratio | Hardcoded in `train_stop_detector()` |

### Vessel Realism Mode

By default, training uses **vessel realism** features:
- **Tapering**: Curves are wide at start (3-8 pixels) → narrow at end (1-2 pixels)
- **Fading**: Curves are bright at start (0.7-1.0) → dim at end (0.2-0.5)

This mimics real vessel appearance where vessels taper and fade as they terminate.

To disable: `--no_vessel_realism`

### Output

Trained weights are saved to:
- Default: `stop_module/weights/stop_detector_v1.pth`
- Custom: Use `--output /path/to/weights.pth`

## Usage in Inference

The stop module is used in `Lego/test/rollout.py`:

```python
from stop_module.src.models import StandaloneStopDetector

# Load model
stop_model = StandaloneStopDetector().to(device)
stop_model.load_state_dict(torch.load(weights_path))
stop_model.eval()

# During tracking
stop_input = torch.cat([obs_t[:, 0:1, :, :], obs_t[:, 3:4, :, :]], dim=1)  # [image, path_mask]
stop_logit = stop_model(stop_input) + stop_logit_bias
stop_prob = torch.sigmoid(stop_logit)

if stop_prob > threshold:
    # Stop the agent
```

### Inference Parameters

- `stop_threshold`: Probability threshold (default: 0.5)
- `stop_logit_bias`: Additive bias to logit (default: 1.0, higher = more likely to stop)
- `min_steps_before_stop`: Prevent stopping before N steps (default: 10)

## Testing

Run the test script to verify the model works:

```bash
python test.py
```

This will:
- Load the trained model
- Test with dummy inputs
- Show example predictions

## Files

- `src/models.py`: Model architecture (`StandaloneStopDetector`)
- `src/train_standalone.py`: Training script
- `test.py`: Test script for verification
- `weights/`: Directory for saved model weights
- `run_train.sh`: Bash wrapper for training

## Notes

- The stop module is **independent** of the tracker module
- It uses the same `curve_generator_module` for data generation (ensures consistency)
- Training is fast (~5-10 minutes on CPU, ~1-2 minutes on GPU)
- The model is small (~50KB weights file)
