# 🛑 Standalone Stop Detector Module

This module focuses exclusively on the "Terminal Task": detecting if the agent is at the exact endpoint of a DSA curve.

## 🧠 Why Standalone?
- **Supervised Precision**: Instead of hoping the RL agent finds the reward at the end, we train this model with thousands of labeled "End vs. Not-End" images.
- **Independence**: This model does not care about movement history or PPO math. It only cares about the current visual crop.

## 🛠️ Usage

### 1. Training on Google Colab (Recommended)
Use the provided notebook for free GPU training:
- `StopModule/colab_stop_training.ipynb`

### 2. Local Training

#### Quick Start (Default Settings)
```bash
cd StopModule
./run_train.sh
```

Or using Python directly:
```bash
python StopModule/src/train_standalone.py
```

**Default behavior:**
- Generates a balanced dataset of 10,000 crops (5,000 endpoints, 5,000 midpoints)
- Trains for 15 epochs with batch size 64
- Learning rate: 1e-4
- Saves weights to `StopModule/weights/stop_detector_v1.pth`

#### Custom Training Options
```bash
# More epochs and larger batch size
./run_train.sh --epochs 20 --batch_size 128

# Larger dataset for better generalization
./run_train.sh --samples 10000 --epochs 20

# Custom learning rate
./run_train.sh --lr 5e-5

# Custom output path
./run_train.sh --output weights/my_stop_detector.pth

# Use specific curve config
./run_train.sh --config ../Experiment5_reward_config/config/curve_config.json

# See all options
./run_train.sh --help
```

#### Python Script Options
```bash
python StopModule/src/train_standalone.py \
    --epochs 20 \
    --batch_size 128 \
    --samples 10000 \
    --lr 5e-5 \
    --output weights/custom_stop_detector.pth \
    --config path/to/config.json
```

**Command-line Arguments:**
- `--epochs, -e`: Number of training epochs (default: 15)
- `--batch_size, -b`: Batch size (default: 64)
- `--samples, -s`: Samples per class (default: 5000)
- `--lr, -l`: Learning rate (default: 1e-4)
- `--output, -o`: Output path for weights
- `--config, -c`: Path to curve config JSON (optional)

### 3. Integration with Fine-Tuning
Once you have a high-accuracy `.pth` file, you can load these weights into the `FineTune/` hub to replace the weak, RL-trained stop heads.

---

## 🏗️ Architecture
- **Input**: 2-channel crop (Image + Path Mask).
- **Backbone**: 3-layer Dilated CNN (GroupNorm + PReLU).
- **Output**: Single logit (Terminal Probability).

## 🩸 Vessel Realism Features

By default, the stop detector is trained with **vessel-realistic features** to better match real DSA vessel characteristics:

- **Tapering (Wide → Narrow)**: Vessels start wide and taper to 1-2 pixels at the end, mimicking real vessel narrowing
- **Fading (Bright → Dim)**: Vessel intensity fades from bright (0.7-1.0) to dim (0.2-0.5) at the endpoint, simulating contrast fading

These features help the model learn to recognize the characteristic "fade-out" pattern of real vessel endpoints, where vessels become very narrow and lose intensity before terminating.

To disable vessel realism (use uniform width/intensity):
```bash
./run_train.sh --no_vessel_realism
```
