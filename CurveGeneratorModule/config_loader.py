#!/usr/bin/env python3
"""
Configuration loader for curve generation.
Reads curve configuration from JSON files.
"""
import json
import os
from typing import Dict, Optional, Tuple


def load_curve_config(config_path: Optional[str] = None) -> Tuple[Dict, Optional[str]]:
    """Load curve generation configuration from JSON file.
    
    Args:
        config_path: Path to the curve config JSON file. If None, will search
                    for default locations. Can be absolute or relative path.
    
    Returns:
        tuple: (config_dict, actual_config_path)
        - config_dict: The loaded configuration dictionary (or empty dict if not found)
        - actual_config_path: The absolute path that was actually used (or None if not found)
    
    The config_dict structure typically includes:
        - image: {height, width}
        - bezier: {n_samples, margin, min_distance, control_point_spread, control_point_factor}
        - branches: {num_branches_range, start_range, thickness_factor}
        - noise: {num_blobs_range, blob_sigma_range, blob_intensity_range, ...}
        - training_stages: [{stage_id, name, curve_generation: {...}, ...}]
    
    Search order when config_path is None:
        1. Current working directory: config/curve_config.json
        2. Current working directory: curve_config.json
        3. Parent directory: config/curve_config.json
    """
    if config_path is None:
        # Search in common locations relative to current working directory
        cwd = os.getcwd()
        default_paths = [
            os.path.join(cwd, "config", "curve_config.json"),
            os.path.join(cwd, "curve_config.json"),
            os.path.join(os.path.dirname(cwd), "config", "curve_config.json"),
        ]
        for path in default_paths:
            if os.path.exists(path):
                config_path = os.path.abspath(path)
                break
        else:
            # No config found, return empty dict
            print(f"⚠️  No curve config found in default locations")
            print(f"   Searched: {default_paths}")
            print("   Using default configuration")
            return {}, None
    
    # Convert to absolute path if relative
    if not os.path.isabs(config_path):
        # Try relative to current working directory first
        abs_path = os.path.abspath(config_path)
        if os.path.exists(abs_path):
            config_path = abs_path
        else:
            # Try relative to common config locations
            cwd = os.getcwd()
            test_paths = [
                os.path.join(cwd, config_path),
                os.path.join(cwd, "config", config_path),
            ]
            for test_path in test_paths:
                if os.path.exists(test_path):
                    config_path = os.path.abspath(test_path)
                    break
            else:
                # Use absolute path anyway (will show error if not found)
                config_path = abs_path
    
    # Load the config
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"✓ Loaded curve configuration from: {config_path}")
            return config, os.path.abspath(config_path)
        except json.JSONDecodeError as e:
            print(f"⚠️  Error parsing JSON config file: {config_path}")
            print(f"   Error: {e}")
            print("   Using default configuration")
            return {}, None
        except Exception as e:
            print(f"⚠️  Error reading config file: {config_path}")
            print(f"   Error: {e}")
            print("   Using default configuration")
            return {}, None
    else:
        print(f"⚠️  Config file not found: {config_path}")
        print("   Using default configuration")
        return {}, None


def save_config_snapshot(config_dict: Dict, output_path: str) -> str:
    """Save a snapshot of the configuration to a file.
    
    This is useful for saving the exact config used in a training run
    for reproducibility.
    
    Args:
        config_dict: The configuration dictionary to save
        output_path: Path where to save the config JSON file
    
    Returns:
        str: The absolute path where the config was saved
    """
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"✓ Saved curve configuration snapshot to: {output_path}")
    return output_path

