# 🎯 DSA RL Fine-Tuning Hub

This directory contains the tools to take your best-performing models ("Champions") and refine them for specific edge cases, difficult curriculum stages, or improved stopping behavior.

---

## 🔄 The Fine-Tuning Workflow

The workflow consists of two main steps: **Registering** a successful model and **Fine-Tuning** it with new parameters.

### 1. Registering a Champion
When an experiment (like Experiment 4) finishes with high success rates, you "bank" it into the fine-tuning hub.

**What it does:**
- Scans the `metrics.json` of your run.
- Ensures the final success rate is **$\ge 90\%$** (you can override this).
- Copies the final `.pth` weights and the `curve_config.json` used.
- Creates a `metadata.json` so the fine-tuner knows if the model uses the Decoupled Stop architecture (8 actions) or the Standard architecture (9 actions).

**How to run:**
```bash
python FineTune/register_champion.py Experiment4_separate_stop_v2/runs/your_run_folder --name Best_Stop_Model
```
*Your model is now safely stored in `FineTune/base_weights/Best_Stop_Model/`.*

---

### 2. Running a Fine-Tuning Session
Once a model is registered, you can run targeted experiments to improve it.

**What it does:**
- Loads the base weights from your champion.
- Loads a specific **Fine-Tune Config** (JSON) containing hyperparameters.
- Applies **Stage Overrides** (e.g., training a model on "Stage 11" conditions but with extra noise).
- Saves results in a new folder: `runs/FT_experiment_name_TIMESTAMP/`.

**How to run:**
```bash
# Option A: Use a dedicated experiment config (Recommended)
python FineTune/src/finetune.py --config FineTune/configs/refine_experiment_v1.json

# Option B: Quick run with command line overrides
python FineTune/src/finetune.py --champion Best_Stop_Model --stage 11 --lr 1e-6
```

---

## ⚙️ Configuration Guide

### Fine-Tune JSON (`FineTune/configs/`)
This is where you "play" with the refinement settings.

```json
{
  "experiment_name": "fix_early_stopping",
  "base_champion": "Best_Stop_Model",
  "target_stage_id": 11,
  "episodes": 5000,
  "hyperparameters": {
    "learning_rate": 0.00001,
    "lambda_stop": 15.0,    // Increase weight of Stop Head training
    "ppo_clip": 0.1,        // Smaller clip for more stable refinement
    "entropy_coef": 0.005   // Lower entropy to stick to learned pathing
  },
  "stage_overrides": {
    "noise": 0.6,           // Override Stage 11 noise levels
    "min_intensity": 0.05   // Make paths even fainter than standard
  }
}
```

---

## 📈 Recording Results
Every fine-tuning run records its own `metrics.json` and a copy of the `finetune_config.json` used. You can find these in the root `runs/` directory, prefixed with **`FT_`**.

### Why use this flow?
1. **Safety**: You never "break" your original champion; you only create refined versions of it.
2. **Speed**: Fine-tuning usually only takes 2k–5k episodes, compared to 100k+ for a full run.
3. **Traceability**: You can look back at `FT_.../finetune_config.json` to see exactly what `lambda_stop` value finally fixed your stopping problem.
