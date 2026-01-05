# Experiment 4 — Decoupled Vision + Separate Stop Head

This experiment improves the reliability of the stopping mechanism by giving the **Stop Head** its own dedicated vision backbone (CNN).

## 🚀 Run Training
```bash
./run_train.sh --experiment_name exp4_decoupled --curve_config config/curve_config.json
```

## 🔍 Run Inference
```bash
./run_rollout.sh --image_path <path_to_image> --actor_weights <path_to_weights> --max_steps 1000
```

## 🧠 Key Differences from v1 (Experiment 2)
- **Decoupled Backbones**: Instead of sharing a backbone, the Stop Head uses a specialized CNN that only looks at the **current visual crop** (Channel 0).
- **Task Specialization**: The Actor focuses on "Where to go" (Path following), while the Stop Head focuses purely on "Is this the end?" (Feature detection).
- **Robustness**: If the Actor gets lost or starts looping, the Stop Head can still function independently to detect the curve endpoint.

## 🎓 Learning More
For a detailed explanation of why this architecture works better for rare labels and endpoint detection, see the tutor guide:
👉 **`../docs/DECOUPLED_STOP_EXPLAINED.md`**
