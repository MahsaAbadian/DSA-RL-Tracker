#!/usr/bin/env python3
"""
Helper script to register high-performing models as base weights for fine-tuning.
Checks if the final success rate is >= threshold and copies the weights.
"""
import os
import sys
import json
import shutil
import argparse
from datetime import datetime

def register_champion(run_dir, name=None, threshold=0.9):
    run_dir = os.path.abspath(run_dir)
    if not os.path.exists(run_dir):
        print(f"❌ Run directory not found: {run_dir}")
        return

    metrics_path = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        print(f"❌ Metrics file not found: {metrics_path}")
        return

    config_path = os.path.join(run_dir, "curve_config.json")
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return

    # Load metrics
    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    except Exception as e:
        print(f"❌ Error loading metrics: {e}")
        return

    # Load config to get n_actions
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return

    # Determine final success rate
    if not metrics.get("stages"):
        print("❌ No stages found in metrics.")
        return

    last_stage = metrics["stages"][-1]
    
    # Check for final_success_rate or the last entry in success_rates
    success_rate = last_stage.get("final_success_rate")
    if success_rate is None and last_stage.get("success_rates"):
        success_rate = last_stage["success_rates"][-1]
    
    if success_rate is None:
        print("❌ Could not determine success rate from metrics.")
        return

    print(f"📊 Final Stage ({last_stage['name']}) Success Rate: {success_rate:.2%}")

    if success_rate < threshold:
        print(f"⚠️  Success rate {success_rate:.2%} is below threshold {threshold:.2%}. Skipping.")
        return

    # Setup champion directory
    if name is None:
        name = f"champion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    base_weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "base_weights", name)
    os.makedirs(base_weights_dir, exist_ok=True)

    # Find final weights
    weights_dir = os.path.join(run_dir, "weights")
    if not os.path.exists(weights_dir):
        print(f"❌ Weights directory not found: {weights_dir}")
        return

    # Look for the last FINAL model
    final_models = [f for f in os.listdir(weights_dir) if f.endswith("_FINAL.pth") and f.startswith("model_")]
    if not final_models:
        print(f"❌ No FINAL models found in {weights_dir}")
        return
    
    # Sort by name or just take the last one if there are multiple stages
    final_models.sort()
    best_model = final_models[-1]
    
    # Copy files
    shutil.copy2(os.path.join(weights_dir, best_model), os.path.join(base_weights_dir, "weights.pth"))
    shutil.copy2(config_path, os.path.join(base_weights_dir, "curve_config.json"))
    
    # Create metadata
    metadata = {
        "original_run_dir": run_dir,
        "registration_date": datetime.now().isoformat(),
        "success_rate": success_rate,
        "n_actions": config.get("n_actions", 8 if "Experiment4" in run_dir else 9), 
        "stage_name": last_stage['name'],
        "experiment_type": "decoupled_stop" if "Experiment4" in run_dir else "standard"
    }
    
    with open(os.path.join(base_weights_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Champion '{name}' registered successfully in {base_weights_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register a high-performing run as base weights.")
    parser.add_argument("run_dir", type=str, help="Path to the experiment run directory")
    parser.add_argument("--name", type=str, default=None, help="Name for the registered champion")
    parser.add_argument("--threshold", type=float, default=0.9, help="Success rate threshold (default: 0.9)")
    
    args = parser.parse_args()
    register_champion(args.run_dir, args.name, args.threshold)
