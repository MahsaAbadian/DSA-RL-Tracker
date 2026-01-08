# 🧠 Experiment 4 (v2): Decoupled Vision Stop Architecture

In this version, we solve the **"Confused Backbone"** problem.

## 1. The Problem in v1

In Experiment 2 (v1), the "eyes" (CNN) of the model were shared between the Actor and the Stop Head.

- **The Conflict**: The Actor is looking for "direction" (where the line goes). The Stop Head is looking for "termination" (where the line ends).
- **The Result**: If the line gets blurry or noisy, the Actor might get confused, which automatically makes the Stop Head fail as well because they use the same features.

## 2. The Solution: Decoupled Backbones

In Experiment 4, we gave the Stop Head its own specialized set of "eyes."

```python
# From models.py
self.actor_cnn = nn.Sequential(...) # Takes 4 channels (History + Path)
self.stop_cnn = nn.Sequential(...)  # Takes 1 channel (Only Current Crop)
```

### Why this is better:

1.  **Specialization**: The `stop_cnn` only has one job: recognize the visual signature of an endpoint.
2.  **Robustness**: By seeing the **Path Mask**, the Stop Head can confirm it has reached a dead end (vessel ahead is background, but there is a trail behind it).
3.  **Context**: Integrating the **Actor LSTM** output means the Stop Head knows its recent movement (velocity/direction), which helps it realize it has exhausted the vessel track.

## 3. How the Data Flows Now

When the agent takes a step:

1.  The **Observation** (4 channels) is fed into the `actor_cnn`.
2.  **Channel 0** (Current Crop) and **Channel 3** (Path Mask) are isolated and fed into the `stop_cnn`.
3.  The **Actor LSTM** output is concatenated with the stop vision features.
4.  The results are combined at the very end to decide whether to move or stop.

## 4. Addressing Rare Labels

Correct stops are rare (label imbalance). In v2, we mitigate this in two ways:
1.  **Feature Detection**: The dedicated backbone makes it easier to learn "Endpoint vs. Not Endpoint".
2.  **Loss Weighting**: We increase the `lambda_stop` loss weight to **5.0**, forcing the model to pay attention to the few stop signals it receives.

---

### 🎓 Beginner Lesson: Multi-Task vs. Specialized Models

In AI, we often start with "Multi-Task" models (one backbone for everything) because they are efficient. But when tasks are very different—like **Navigation** (moving) and **Classification** (is this an endpoint?)—giving each task its own "brain branch" usually leads to much higher accuracy.



<!-- Git push test at 2026-01-08 09:09:55 -->

<!-- Git push test at 2026-01-08 09:10:26 -->

<!-- Git push test at 2026-01-08 09:11:33 -->

<!-- Git push test at 2026-01-08 09:56:40 -->

<!-- Git push test at 2026-01-08 10:58:05 -->