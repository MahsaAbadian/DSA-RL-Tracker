#!/usr/bin/env python3
"""
Curve Generator Module

A centralized module for curve generation used across experiments.
Provides:
- Configuration loading from JSON files
- Curve generation with flexible parameters
- Config snapshot saving for reproducibility
"""

from .config_loader import load_curve_config, save_config_snapshot
from .generator import CurveMakerFlexible
from .generator_multisegment import CurveMakerMultiSegment
from .generator_sixpoint import CurveMakerSixPoint

__all__ = [
    'load_curve_config',
    'save_config_snapshot',
    'CurveMakerFlexible',
    'CurveMakerMultiSegment',
    'CurveMakerSixPoint',
]

