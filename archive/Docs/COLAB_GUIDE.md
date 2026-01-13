# 📓 Colab Notebook Development Guide

This guide outlines the standard structure and conventions for Colab notebooks in the `DSA-RL-Tracker` repository. These conventions ensure that training sessions are reproducible and that results are never lost.

## 1. Standard Notebook Structure

Every experiment should have a `colab_training.ipynb` file structured as follows:

### Cell 1: Environment Setup
- **Action**: Clone the repository, navigate to the root, and install dependencies.
- **Convention**: Always use `!git clone` and `!pip install -r ExperimentX/requirements.txt`.

### Cell 2: Unified Training & Auto-Push
- **Action**: Run the `train.py` script followed immediately by Git commands to commit and push results.
- **Why**: Training in Colab can take hours and the runtime might disconnect. Putting the `git push` in the same cell as `!python src/train.py` ensures that if the training finishes, the results are saved to GitHub automatically.
- **Git Configuration**: Include `!git config --global user.email` and `!git config --global user.name` inside this cell if not already configured globally in the environment.
- **Files to Push**: Always target the `runs/` directory for the specific experiment.

### Cell 3: Inference (Optional)
- **Action**: Provide a template for running `rollout.py` to test the newly trained weights.

## 2. Key Path Conventions

- **Root Path**: `/content/DSA-RL-Tracker/`
- **Experiment Paths**: Navigate to the specific experiment folder before running commands (e.g., `%cd /content/DSA-RL-Tracker/Experiment4_separate_stop_v2`).

## 3. Git Persistence (Critical)

Colab runtimes are temporary. To avoid losing hours of training:
1.  Ensure the training command and the git push command are in the **same code block**.
2.  If the user is using a private repo or needs to push, ensure the remote URL includes a token or the environment has a `GH_TOKEN` set up.
3.  **Commit Message Format**: Use `Results: Automated push after [Experiment Name] training session`.

## 4. Tips for AI Assistants

When editing `.ipynb` files:
- Use the `write` tool with the full JSON structure of the notebook.
- Ensure `\n` is used for line breaks inside the `source` array of each cell.
- Maintain the `nbformat` and `metadata` fields to keep the file valid.

---
*Created on 2026-01-08 for the DSA-RL-Tracker team.*

