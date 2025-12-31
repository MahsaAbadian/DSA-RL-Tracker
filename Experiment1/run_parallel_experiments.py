#!/usr/bin/env python3
"""
Parallel training script for multiple curve generation configurations.

Runs multiple training experiments in parallel, each with a different curve configuration.
Each run saves results to runs/ folder with a copy of the configuration used.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def create_config_file(config_dict, output_path):
    """Save configuration dictionary to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    return output_path

def load_base_config(config_path=None):
    """Load base configuration file."""
    if config_path is None:
        # Look in config/ directory first, then Experiment1 root for backward compatibility
        config_path = os.path.join(SCRIPT_DIR, "config", "curve_config.json")
        if not os.path.exists(config_path):
            config_path = os.path.join(SCRIPT_DIR, "curve_config.json")
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        # Return default minimal config
        return {
            "image": {"height": 128, "width": 128},
            "training_stages": [
                {
                    "stage_id": 1,
                    "name": "Stage1_Bootstrap",
                    "episodes": 8000,
                    "learning_rate": 1e-4,
                    "curve_generation": {
                        "width_range": [2, 4],
                        "noise_prob": 0.0,
                        "invert_prob": 0.5,
                        "min_intensity": 0.6,
                        "branches": False,
                        "curvature_factor": 0.5
                    },
                    "training": {
                        "noise": 0.0,
                        "tissue": False,
                        "strict_stop": False,
                        "mixed_start": False
                    }
                }
            ]
        }

def run_training_experiment(config_dict, experiment_name, base_seed, config_id):
    """
    Run a single training experiment with given configuration.
    
    Args:
        config_dict: Configuration dictionary
        experiment_name: Base experiment name
        base_seed: Base seed for reproducibility
        config_id: Unique identifier for this configuration
    
    Returns:
        tuple: (config_id, success, run_dir, error_message)
    """
    try:
        # Create unique experiment name
        unique_name = f"{experiment_name}_config{config_id}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_experiment_name = f"{unique_name}_{timestamp}"
        
        # Create temporary config file
        temp_config_dir = os.path.join(SCRIPT_DIR, "temp_configs")
        os.makedirs(temp_config_dir, exist_ok=True)
        temp_config_path = os.path.join(temp_config_dir, f"config_{config_id}.json")
        create_config_file(config_dict, temp_config_path)
        
        # Build training command
        train_script = os.path.join(SCRIPT_DIR, "src", "train.py")
        cmd = [
            sys.executable,
            train_script,
            "--experiment_name", full_experiment_name,
            "--base_seed", str(base_seed + config_id),  # Different seed per config
            "--curve_config", temp_config_path
        ]
        
        print(f"[Config {config_id}] Starting training: {full_experiment_name}")
        print(f"[Config {config_id}] Command: {' '.join(cmd)}")
        
        # Run training
        result = subprocess.run(
            cmd,
            cwd=SCRIPT_DIR,
            capture_output=True,
            text=True,
            timeout=None  # No timeout, let it run
        )
        
        # Find the run directory
        runs_dir = os.path.join(SCRIPT_DIR, "runs")
        run_dirs = []
        if os.path.exists(runs_dir):
            run_dirs = [d for d in os.listdir(runs_dir) 
                       if d.startswith(full_experiment_name)]
        
        run_dir = None
        if run_dirs:
            # Get most recent
            run_dir = os.path.join(runs_dir, sorted(run_dirs)[-1])
            # Copy config to run directory
            config_copy_path = os.path.join(run_dir, "curve_config.json")
            create_config_file(config_dict, config_copy_path)
            print(f"[Config {config_id}] Config saved to: {config_copy_path}")
        
        # Clean up temp config
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
        
        if result.returncode == 0:
            print(f"[Config {config_id}] ✅ Training completed successfully")
            return (config_id, True, run_dir, None)
        else:
            error_msg = result.stderr[-500:] if result.stderr else "Unknown error"
            print(f"[Config {config_id}] ❌ Training failed: {error_msg}")
            return (config_id, False, run_dir, error_msg)
            
    except Exception as e:
        error_msg = str(e)
        print(f"[Config {config_id}] ❌ Exception: {error_msg}")
        return (config_id, False, None, error_msg)

def generate_config_variations(base_config, variations):
    """
    Generate multiple configuration variations from base config.
    
    Args:
        base_config: Base configuration dictionary
        variations: List of variation dictionaries, each specifying what to change
    
    Returns:
        List of configuration dictionaries
    """
    configs = []
    
    for i, variation in enumerate(variations):
        # Deep copy base config
        import copy
        config = copy.deepcopy(base_config)
        
        # Apply variation
        for key_path, value in variation.items():
            # Skip comment fields
            if key_path == "comment":
                continue
                
            # Handle nested keys like "training_stages.0.curve_generation.width_range"
            keys = key_path.split('.')
            target = config
            for key in keys[:-1]:
                # Handle list indices
                if key.isdigit():
                    target = target[int(key)]
                else:
                    if key not in target:
                        target[key] = {}
                    target = target[key]
            
            # Set final value
            final_key = keys[-1]
            if final_key.isdigit():
                target[int(final_key)] = value
            else:
                target[final_key] = value
        
        configs.append(config)
    
    return configs

def main():
    parser = argparse.ArgumentParser(
        description="Run parallel training experiments with different curve configurations"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="parallel_exp",
        help="Base name for experiments (default: parallel_exp)"
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="Base seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--base_config",
        type=str,
        default=None,
        help="Path to base curve_config.json (default: curve_config.json)"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Path to JSON file containing list of config variations (alternative to --variations)"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Maximum number of parallel workers (default: number of CPUs)"
    )
    parser.add_argument(
        "--variations",
        type=str,
        default=None,
        help="JSON string with list of variations (alternative to --config_file)"
    )
    
    args = parser.parse_args()
    
    # Load base configuration
    base_config = load_base_config(args.base_config)
    
    # Determine number of workers
    max_workers = args.max_workers or min(multiprocessing.cpu_count(), 4)  # Limit to 4 by default
    print(f"Using {max_workers} parallel workers")
    
    # Load variations
    if args.config_file:
        # Load from file
        with open(args.config_file, 'r') as f:
            config_data = json.load(f)
            # Handle both formats: list of variations or list of objects with "variations" key
            if isinstance(config_data, list):
                if len(config_data) > 0 and "variations" in config_data[0]:
                    # Format: [{"name": "...", "variations": {...}}, ...]
                    variations = [item["variations"] for item in config_data]
                else:
                    # Format: [{...}, {...}] - direct variations
                    variations = config_data
            else:
                raise ValueError("Config file must contain a list of variations")
    elif args.variations:
        # Parse from string
        variations = json.loads(args.variations)
    else:
        # Default: create some example variations
        print("No variations specified, using default example variations...")
        variations = [
            {
                "training_stages.0.curve_generation.width_range": [2, 4],
                "training_stages.0.curve_generation.curvature_factor": 0.5
            },
            {
                "training_stages.0.curve_generation.width_range": [1, 3],
                "training_stages.0.curve_generation.curvature_factor": 1.0
            },
            {
                "training_stages.0.curve_generation.width_range": [1, 5],
                "training_stages.0.curve_generation.curvature_factor": 1.5
            }
        ]
    
    # Generate configurations
    configs = generate_config_variations(base_config, variations)
    
    print(f"\n{'='*60}")
    print(f"Starting {len(configs)} parallel training experiments")
    print(f"Base experiment name: {args.experiment_name}")
    print(f"Base seed: {args.base_seed}")
    print(f"Max workers: {max_workers}")
    print(f"{'='*60}\n")
    
    # Create temp configs directory
    temp_config_dir = os.path.join(SCRIPT_DIR, "temp_configs")
    os.makedirs(temp_config_dir, exist_ok=True)
    
    # Run experiments in parallel
    start_time = time.time()
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        futures = {
            executor.submit(
                run_training_experiment,
                config,
                args.experiment_name,
                args.base_seed,
                i
            ): i
            for i, config in enumerate(configs)
        }
        
        # Collect results as they complete
        for future in as_completed(futures):
            config_id = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"[Config {config_id}] ❌ Exception in executor: {e}")
                results.append((config_id, False, None, str(e)))
    
    # Clean up temp configs directory
    if os.path.exists(temp_config_dir):
        import shutil
        shutil.rmtree(temp_config_dir)
    
    # Print summary
    elapsed_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total experiments: {len(configs)}")
    print(f"Successful: {sum(1 for r in results if r[1])}")
    print(f"Failed: {sum(1 for r in results if not r[1])}")
    print(f"Total time: {elapsed_time/60:.1f} minutes")
    print(f"{'='*60}\n")
    
    # Print detailed results
    print("Detailed Results:")
    for config_id, success, run_dir, error in sorted(results):
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  Config {config_id}: {status}")
        if run_dir:
            print(f"    Run directory: {run_dir}")
        if error:
            print(f"    Error: {error[:100]}...")
    
    # Save summary to file
    summary_path = os.path.join(SCRIPT_DIR, "runs", f"{args.experiment_name}_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    
    summary = {
        "experiment_name": args.experiment_name,
        "base_seed": args.base_seed,
        "total_experiments": len(configs),
        "successful": sum(1 for r in results if r[1]),
        "failed": sum(1 for r in results if not r[1]),
        "total_time_minutes": elapsed_time / 60,
        "results": [
            {
                "config_id": r[0],
                "success": r[1],
                "run_dir": r[2],
                "error": r[3]
            }
            for r in results
        ]
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_path}")

if __name__ == "__main__":
    main()

