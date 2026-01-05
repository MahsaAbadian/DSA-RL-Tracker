# Experiment 2 — Shared Backbone + Separate Stop Head

Run training:
```
./run_train.sh
# or with options:
# ./run_train.sh --experiment_name exp2_shared --base_seed 42 --curve_config config/curve_config.json
```

Run inference:
```
./run_rollout.sh --image_path <path_to_image> --actor_weights <path_to_weights> --max_steps 1000
```

Key differences from Experiment1:
- Actor head outputs 8 movement actions (no stop action).
- Separate stop head (binary) decides when to stop.
- Shared backbone (CNN+LSTM) feeds both heads.

