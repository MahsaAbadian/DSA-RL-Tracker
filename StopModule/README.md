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
To train a fresh stop detector locally:
```bash
python StopModule/src/train_standalone.py
```
This will:
1. Generate a balanced dataset of 10,000 crops (5,000 endpoints, 5,000 midpoints).
2. Train a dedicated CNN-classifier for 10 epochs.
3. Save the weights to `StopModule/weights/stop_detector_v1.pth`.

### 2. Integration with Fine-Tuning
Once you have a high-accuracy `.pth` file, you can load these weights into the `FineTune/` hub to replace the weak, RL-trained stop heads.

---

## 🏗️ Architecture
- **Input**: 2-channel crop (Image + Path Mask).
- **Backbone**: 3-layer Dilated CNN (GroupNorm + PReLU).
- **Output**: Single logit (Terminal Probability).
