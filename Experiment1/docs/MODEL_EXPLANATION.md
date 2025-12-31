# Model Architecture Explanation

This document explains how the CNN and LSTM are combined, what data flows through each component, and why the architecture is designed this way.

## Overview: CNN + LSTM Hybrid Architecture

The model uses a **hybrid architecture** that combines:
- **CNN**: Processes spatial image patches (what the agent "sees")
- **LSTM**: Processes temporal action history (what the agent "did recently")
- **Fusion**: Concatenates CNN features + LSTM features → Final decision

```
┌─────────────┐
│ Image Patch │──┐
│  (4×33×33)  │  │
└─────────────┘  │
                 ├──► CNN ──► [64-dim features]
┌─────────────┐  │
│ Action Hist │  │
│  (K×9)      │──┘
└─────────────┘
                 │
                 ├──► LSTM ──► [64-dim features]
                 │
                 ▼
         ┌───────────────┐
         │ Concatenate   │
         │ [64 + 64]     │
         └───────┬───────┘
                 │
                 ▼
         ┌───────────────┐
         │ Actor Head    │
         │ (MLP)         │
         └───────┬───────┘
                 │
                 ▼
         ┌───────────────┐
         │ Action Logits │
         │ (9 actions)   │
         └───────────────┘
```

---

## 1. How CNN and LSTM are Combined

### The Forward Pass Flow:

```python
# Step 1: CNN processes image observation
feat_a = self.actor_cnn(actor_obs).flatten(1)  # (B, 64)

# Step 2: LSTM processes action history
lstm_a, hc_actor = self.actor_lstm(ahist_onehot, hc_actor)  # (B, K, 64)
lstm_output = lstm_a[:, -1, :]  # Take last timestep: (B, 64)

# Step 3: Concatenate CNN features + LSTM features
joint_a = torch.cat([feat_a, lstm_output], dim=1)  # (B, 128)

# Step 4: Final decision through actor head
logits = self.actor_head(joint_a)  # (B, 9)
```

**Key Point**: The CNN and LSTM process **different types of information** in parallel, then their outputs are concatenated before making the final decision.

---

## 2. What Goes Into the LSTM?

The LSTM receives **action history** as one-hot encoded vectors:

```python
# Action history shape: (B, K, n_actions)
# K = 8 (last 8 actions)
# n_actions = 9 (8 movement directions + 1 stop)

# Example: If agent took actions [2, 5, 0, 8, 1, ...]
# Then ahist_onehot[0] = [0, 0, 1, 0, 0, 0, 0, 0, 0]  # action 2
#      ahist_onehot[1] = [0, 0, 0, 0, 0, 1, 0, 0, 0]  # action 5
#      ahist_onehot[2] = [1, 0, 0, 0, 0, 0, 0, 0, 0]  # action 0
#      ...
```

**Why one-hot?** 
- Each action is represented as a 9-dimensional vector
- Only one position is "1" (the action taken), rest are "0"
- This makes it easy for the LSTM to learn patterns in action sequences

**What the LSTM learns:**
- Movement patterns (e.g., "agent tends to go right after going up")
- Momentum (e.g., "agent has been moving diagonally")
- Action sequences (e.g., "agent alternates between two directions")

---

## 3. How Much Memory Does the LSTM Have?

The LSTM has **two types of memory**:

### a) **Hidden State Size**: 64 dimensions
```python
self.actor_lstm = nn.LSTM(input_size=n_actions, hidden_size=64, batch_first=True)
```
- The hidden state is a 64-dimensional vector
- This is the "working memory" that gets updated each timestep
- It stores compressed information about the action sequence

### b) **Action History Window**: K = 8 timesteps
```python
def __init__(self, n_actions=9, K=8):
```
- The model explicitly tracks the last **8 actions**
- This is a **fixed window** (not infinite memory)
- Older actions beyond K=8 are forgotten

**Total Memory Capacity:**
- **Short-term (explicit)**: Last 8 actions (via `ahist_onehot`)
- **Long-term (implicit)**: Compressed in 64-dim hidden state (can remember patterns longer than 8 steps)

**Why K=8?**
- Balances between:
  - **Too small (K=2-4)**: Not enough context to learn movement patterns
  - **Too large (K=16+)**: More parameters, slower training, diminishing returns
- 8 actions ≈ 16 pixels of movement (each action moves ~2 pixels), which captures local curve direction

---

## 4. What is the Actor Head?

The **actor head** is a **Multi-Layer Perceptron (MLP)** that makes the final decision:

```python
self.actor_head = nn.Sequential(
    nn.Linear(128, 128),    # First layer: 128 → 128
    nn.PReLU(),             # Activation function
    nn.Linear(128, 9)       # Output layer: 128 → 9 (one logit per action)
)
```

**What it does:**
1. Takes the **concatenated features** (64 from CNN + 64 from LSTM = 128 total)
2. Applies two linear transformations with activation
3. Outputs **9 logits** (one for each possible action)
4. Higher logit = higher probability of selecting that action

**Why "Actor"?**
- In reinforcement learning, the **actor** is the policy network (decides what action to take)
- The **critic** is the value network (estimates how good the current state is)
- This model has both: `AsymmetricActorCritic` = Actor (policy) + Critic (value)

---

## 5. What is `nn.Sequential`?

`nn.Sequential` is a PyTorch container that **chains layers together**:

```python
self.actor_cnn = nn.Sequential(
    nn.Conv2d(4, 32, 3, padding=1),      # Layer 1
    gn(32),                               # Layer 2 (GroupNorm)
    nn.PReLU(),                           # Layer 3 (activation)
    nn.Conv2d(32, 32, 3, padding=2, ...), # Layer 4
    gn(32),                               # Layer 5
    nn.PReLU(),                           # Layer 6
    # ... more layers
)
```

**What it does:**
- Applies layers **in sequence** (left to right)
- Input flows: `input → Layer1 → Layer2 → Layer3 → ... → output`
- Equivalent to manually calling each layer, but cleaner code

**Example:**
```python
# Instead of writing:
x = conv1(input)
x = norm1(x)
x = activation1(x)
x = conv2(x)
# ...

# You write:
x = sequential_model(input)  # Does all steps automatically
```

---

## 6. Why Stack Previous 32×32 Patches (Image History)?

The model stacks **4 channels** of 33×33 patches (not 32×32, but close):

```python
# From train.py, obs() method:
ch0 = crop32(self.ep.img, int(curr[0]), int(curr[1]))      # Current position
ch1 = crop32(self.ep.img, int(p1[0]), int(p1[1]))          # Previous position (-1)
ch2 = crop32(self.ep.img, int(p2[0]), int(p2[1]))          # Previous position (-2)
ch3 = crop32(self.path_mask, int(curr[0]), int(curr[1]))   # Path mask (where agent has been)

actor_obs = np.stack([ch0, ch1, ch2, ch3], axis=0)  # Shape: (4, 33, 33)
```

**Why 4 channels?**

1. **Channel 0 (Current)**: What the agent sees NOW
   - Shows the curve at current position
   - Most important for immediate decisions

2. **Channel 1 (Previous -1)**: What the agent saw 1 step ago
   - Provides **temporal context** (how the curve looked before)
   - Helps detect curve direction/curvature

3. **Channel 2 (Previous -2)**: What the agent saw 2 steps ago
   - Provides **longer temporal context**
   - Helps detect curve trends (e.g., "curve is curving left")

4. **Channel 3 (Path Mask)**: Where the agent has been
   - Shows visited pixels (prevents backtracking)
   - Helps avoid loops

**Why not use LSTM for image history?**

- **CNN processes spatial patterns** (what's in the image)
- **LSTM processes action patterns** (what actions were taken)
- Using CNN for image history is more efficient:
  - CNN can learn spatial-temporal patterns directly (e.g., "curve goes from left to right")
  - LSTM would need to process full images, which is computationally expensive
  - Stacking patches as channels is a common technique (like RGB images have 3 channels)

**Analogy:**
- Think of it like a **video**: Each channel is a "frame" showing the agent's view at different times
- The CNN learns to recognize patterns across these frames (e.g., "curve is moving upward")

---

## Summary: Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ INPUTS                                                       │
├─────────────────────────────────────────────────────────────┤
│ actor_obs: (B, 4, 33, 33)                                   │
│   - ch0: Current image patch                                │
│   - ch1: Previous image patch (-1 step)                     │
│   - ch2: Previous image patch (-2 steps)                     │
│   - ch3: Path mask (visited pixels)                         │
│                                                              │
│ ahist_onehot: (B, K=8, n_actions=9)                        │
│   - Last 8 actions as one-hot vectors                       │
└─────────────────────────────────────────────────────────────┘
                    │
                    ├──────────────────┬──────────────────┐
                    ▼                  ▼                  ▼
            ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
            │   CNN        │  │   LSTM       │  │   (Critic)   │
            │              │  │              │  │              │
            │ Processes    │  │ Processes    │  │ Uses GT map  │
            │ image        │  │ action       │  │ for training │
            │ patches      │  │ history      │  │ only         │
            │              │  │              │  │              │
            │ Output: 64   │  │ Output: 64   │  │ Output: 64   │
            └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
                   │                 │                 │
                   └────────┬────────┘                 │
                            ▼                          │
                    ┌───────────────┐                  │
                    │ Concatenate   │                  │
                    │ [64 + 64]    │                  │
                    │ = 128 dims   │                  │
                    └───────┬───────┘                  │
                            │                          │
                            ▼                          │
                    ┌───────────────┐                  │
                    │ Actor Head    │                  │
                    │ (MLP)         │                  │
                    └───────┬───────┘                  │
                            │                          │
                            ▼                          │
                    ┌───────────────┐                  │
                    │ Action Logits │                  │
                    │ (9 actions)   │                  │
                    └───────────────┘                  │
```

---

## Key Design Decisions

1. **Asymmetric Architecture**: Actor and Critic use different inputs
   - Actor: Only sees what agent can see (4-channel patches)
   - Critic: Also sees ground truth map (for better value estimation during training)

2. **Hybrid CNN+LSTM**: 
   - CNN: Spatial understanding (what's in the image)
   - LSTM: Temporal understanding (action patterns)

3. **Fixed Action History Window (K=8)**:
   - Explicit short-term memory
   - LSTM hidden state provides longer-term memory

4. **Image History via Channel Stacking**:
   - Efficient way to provide temporal context
   - CNN learns spatial-temporal patterns directly

