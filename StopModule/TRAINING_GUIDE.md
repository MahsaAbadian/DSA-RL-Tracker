# 🚀 Stop Module Training Guide

## Quick Start Commands

### Basic Training (Default Settings)
```bash
cd StopModule
./run_train.sh
```

This will:
- Generate 10,000 samples (5,000 endpoints + 5,000 midpoints)
- **Use vessel-realistic features**: Tapering (wide→narrow) and fading (bright→dim) at endpoints
- Train for 15 epochs
- Use batch size 64
- Learning rate: 1e-4
- Save to `StopModule/weights/stop_detector_v1.pth`

### Custom Training Examples

#### Longer Training with More Data
```bash
./run_train.sh --epochs 30 --samples 10000
```

#### Larger Batch Size (Faster Training)
```bash
./run_train.sh --batch_size 128 --epochs 20
```

#### Custom Learning Rate
```bash
./run_train.sh --lr 5e-5 --epochs 25
```

#### Save to Custom Location
```bash
./run_train.sh --output weights/my_custom_stop_detector.pth
```

#### Use Specific Curve Config
```bash
./run_train.sh --config ../Experiment5_reward_config/config/curve_config.json
```

#### Full Custom Training
```bash
./run_train.sh \
    --epochs 25 \
    --batch_size 128 \
    --samples 10000 \
    --lr 5e-5 \
    --output weights/stop_detector_v2.pth
```

#### Disable Vessel Realism (Uniform Width/Intensity)
```bash
# Use uniform curves without tapering/fading
./run_train.sh --no_vessel_realism
```

## Using Python Directly

If you prefer to call Python directly:

```bash
# Basic
python StopModule/src/train_standalone.py

# With options
python StopModule/src/train_standalone.py \
    --epochs 20 \
    --batch_size 128 \
    --samples 10000 \
    --lr 5e-5 \
    --output weights/custom.pth \
    --config path/to/config.json
```

## Training Strategy Recommendations

### For Quick Testing
```bash
./run_train.sh --epochs 10 --samples 2000
```

### For Production Quality
```bash
./run_train.sh --epochs 30 --samples 10000 --batch_size 128
```

### For Fine-Tuning Existing Model
```bash
# Start with lower learning rate
./run_train.sh --epochs 20 --lr 1e-5 --samples 15000
```

## What Gets Trained?

The model learns to classify:
- **Positive (Label=1)**: Agent is at the exact endpoint of the curve
- **Negative (Label=0)**: Agent is at a midpoint (not at the end)

### Vessel Realism Features (Default)

By default, training uses **vessel-realistic features** to better match real DSA vessels:

- **Tapering**: Vessels start wide (3-8 pixels) and taper to 1-2 pixels at the end
- **Fading**: Vessel intensity fades from bright (0.7-1.0) to dim (0.2-0.5) at endpoints
- **Realistic Termination**: Endpoints are very narrow and faded, mimicking real vessel behavior

This helps the model learn the characteristic "fade-out" pattern where vessels become narrow and lose intensity before terminating, which is crucial for detecting real vessel endpoints.

To disable and use uniform curves: `--no_vessel_realism`

The training automatically:
1. Generates random curves with varied parameters
2. Creates balanced positive/negative samples
3. Splits into 80% train / 20% validation
4. Tracks per-class accuracy (Stop vs Go)
5. Saves the best model based on validation accuracy

## Monitoring Training

During training, you'll see output like:
```
Epoch 01/15 | T-Loss: 0.6234 | T-Acc: 65.23% | V-Acc: 68.45% (Stop: 70.1%, Go: 66.8%)
⭐ New Best Val Accuracy! Saved to StopModule/weights/stop_detector_v1.pth
```

Look for:
- **T-Acc**: Training accuracy
- **V-Acc**: Validation accuracy (most important)
- **Stop/Go**: Per-class accuracy breakdown

## After Training

Once training completes, the weights are saved and can be:
1. Used directly for inference
2. Loaded into the FineTune module
3. Integrated into RL training pipelines

