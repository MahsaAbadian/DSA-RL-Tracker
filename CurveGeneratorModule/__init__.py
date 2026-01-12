#!/usr/bin/env python3
"""
Curve Generator Module

A centralized module for curve generation used across experiments.
Provides a unified, config-driven CurveMaker class.
"""

from .config_loader import load_curve_config, save_config_snapshot
from .generator import CurveMaker, CurveMakerFlexible

__all__ = [
    'load_curve_config',
    'save_config_snapshot',
    'CurveMaker',
    'CurveMakerFlexible',
]
