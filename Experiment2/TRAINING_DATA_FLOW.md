# Training Data Flow - What Labels Are Used?

## Step-by-Step Data Collection & Label Generation

### 1. **During Episode (Data Collection)**

For each timestep in an episode:

```python
# Model outputs (from current state)
movement_logits = model(...)  # 8 numbers (probabilities for actions 0-7)
stop_logit = model(...)       # 1 number (logit for stop probability)
value = model(...)            # 1 number (predicted future reward)

# Agent samples an action
action = sample(movement_logits, stop_logit)  # Either 0-7 (move) or 8 (stop)

# Environment step
next_obs, reward, done, info = env.step(action)

# Store data for this timestep:
ep_traj["obs"].append(obs)           # Observation (what agent saw)
ep_traj["ahist"].append(ahist)       # Action history (last 8 actions)
ep_traj["act"].append(action)        # Action taken (0-8)
ep_traj["logp"].append(logp)         # Log probability of that action
ep_traj["val"].append(value)         # Value prediction
ep_traj["rew"].append(reward)        # Reward received
ep_traj["stop_label"].append(stop_label)  # ← LABEL FOR STOP HEAD
```

### 2. **Stop Label Generation (Line 1384)**

```python
stop_label = 1 if (info.get('stopped_correctly') or info.get('reached_end')) else 0
```

**What this means:**
- `stop_label = 1` (should stop): If agent reached the end OR stopped correctly at this step
- `stop_label = 0` (should continue): Otherwise (agent should keep moving)

**Important:** This label is created **at each timestep**, not just at the end!

### 3. **After Episode Ends (Compute Returns & Advantages)**

```python
# Compute advantages from rewards (PPO needs this)
rews = [r1, r2, r3, ..., rN]  # Rewards from episode
vals = [v1, v2, v3, ..., vN, 0]  # Value predictions + terminal value

# Compute advantages (GAE - Generalized Advantage Estimation)
adv = compute_advantages(rews, vals)  # How much better/worse than expected

# Compute returns (target values for critic)
ret = adv + vals[:-1]  # Returns = advantages + value predictions
```

### 4. **Store in Buffer**

```python
final_ep_data = {
    "obs": [...],          # All observations
    "ahist": [...],        # All action histories
    "act": [...],          # All actions taken
    "logp": [...],         # All log probabilities (old policy)
    "adv": [...],          # All advantages (for PPO)
    "ret": [...],          # All returns (for critic)
    "stop_label": [...]    # All stop labels (for stop head)
}
```

## Loss Calculation (What Labels/Targets Are Used?)

### **Loss 1: PPO Policy Loss (Movement Head)**

**Inputs:**
- `obs`, `ahist`: Observations and action history
- `act`: Actions that were taken (0-8)
- `adv`: Advantages (how good/bad each action was)
- `logp`: Old log probabilities (from when action was sampled)

**Process:**
1. Model recomputes: `movement_logits, stop_logit = model(obs, ahist)`
2. Compute new log probabilities: `new_logp = log_prob(action)`
3. Compare old vs new: `ratio = exp(new_logp - old_logp)`
4. PPO loss: `p_loss = -min(ratio * adv, clip(ratio) * adv)`

**Label/Target:** Uses **advantages** (not explicit labels) - tells model if action was good or bad

### **Loss 2: Value Loss (Critic Head)**

**Inputs:**
- `obs`, `ahist`: Observations and action history  
- `ret`: Returns (computed from rewards + advantages)

**Process:**
1. Model recomputes: `value = model(obs, ahist)` (critic head)
2. Value loss: `v_loss = MSE(value, ret)`

**Label/Target:** `ret` (returns) - what the actual total reward was from that state

### **Loss 3: Stop Loss (Stop Head)**

**Inputs:**
- `obs`, `ahist`: Observations and action history
- `stop_labels`: Binary labels (1 = should stop, 0 = should continue)

**Process:**
1. Model recomputes: `stop_logit = model(obs, ahist)` (stop head)
2. Stop loss: `stop_loss = BCE(stop_logit, stop_labels)`

**Label/Target:** `stop_labels` - explicit binary labels indicating when agent should stop

## Combined Loss

```python
total_loss = p_loss + 0.5 * v_loss + lambda_stop * stop_loss - 0.01 * entropy
```

## Summary Table

| Component | Loss Type | Label/Target | Source |
|-----------|-----------|--------------|--------|
| **Movement Head** | PPO (policy gradient) | Advantages (`adv`) | Computed from rewards |
| **Critic Head** | MSE (regression) | Returns (`ret`) | Computed from rewards + advantages |
| **Stop Head** | BCE (binary classification) | Stop labels (`stop_label`) | From `info` dict (reached_end/stopped_correctly) |

## Key Insight

- **Movement & Critic**: Use **reward-based signals** (advantages/returns) - no explicit "correct" actions
- **Stop Head**: Uses **explicit binary labels** - we know when agent should have stopped (from environment info)

This is **semi-supervised learning**: Movement uses RL (reward signals), Stop uses supervised learning (explicit labels).

