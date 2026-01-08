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

1.  **Specialization**: The `stop_cnn` only has one job: recognize the visual signature of an endpoint. It doesn't care about movement history or where the agent has already been.
2.  **Robustness**: If the Actor is spinning in circles because it's lost, the Stop Head can still look at the current patch and say: _"I don't see an endpoint here, keep trying."_
3.  **Efficiency**: The `stop_cnn` is smaller and faster. It focuses on high-level spatial patterns rather than temporal movement.

## 3. How the Data Flows Now

When the agent takes a step:

1.  The **Observation** (4 channels) is fed into the `actor_cnn`.
2.  **Channel 0** (the current visual crop) is isolated and fed into the `stop_cnn`.
3.  The results are combined at the very end to decide whether to move or stop.

## 4. Addressing Rare Labels

Even with a better backbone, correct stops are still rare. In v2, the `stop_cnn` helps because it turns the problem into a pure **Feature Detection** task. It is much easier for a dedicated CNN to learn "Endpoint vs. Not Endpoint" than it is for a shared CNN to learn "Move Left AND Stop if you see this."

---

### 🎓 Beginner Lesson: Multi-Task vs. Specialized Models

In AI, we often start with "Multi-Task" models (one backbone for everything) because they are efficient. But when tasks are very different—like **Navigation** (moving) and **Classification** (is this an endpoint?)—giving each task its own "brain branch" usually leads to much higher accuracy.



<!-- Git push test at 2026-01-08 09:09:55 -->

<!-- Git push test at 2026-01-08 09:10:26 -->