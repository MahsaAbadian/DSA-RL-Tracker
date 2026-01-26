# Curve Generator Module

Config-driven generator for synthetic vessel curves and masks.

## What it provides
- `generator.py`: `CurveMaker` class that draws curves, masks, noise, and backgrounds.
- `config_loader.py`: load/save JSON configs for curve generation.
- `visual_sampler.py`: helper for visual sampling (see file for usage).
- `config/stop_curve_config.json`: default config for the stop module.

## Quick start
Use the loader to get a config and then sample a curve:

```python
from curve_generator_module import CurveMaker, load_curve_config

cfg, _ = load_curve_config("config/stop_curve_config.json")
maker = CurveMaker(h=128, w=128, config=cfg)
img, mask, pts_all = maker.sample_curve(
    width_range=(2, 5),
    curvature_factor=1.0,
    noise_prob=0.1,
    invert_prob=0.0,
    background_intensity=0.0,
)
```

## Config loading behavior
`load_curve_config()` searches these paths when no path is provided:
- `./config/curve_config.json`
- `./curve_config.json`
- `../config/curve_config.json`

You can always pass an explicit path:
```python
cfg, path = load_curve_config("config/stop_curve_config.json")
```

## Config fields (high level)
Common sections in a config JSON:
- `image`: output size (e.g., `height`, `width`)
- `bezier`: curve geometry parameters
- `branches`: branch count and thickness
- `noise`: blob noise and blur
- `tissue_noise`: low-frequency background texture

See `curve_config_tutorial.ipynb` for examples and visuals.

## Stop module default config
The stop module uses this config by default:
`Lego/curve_generator_module/config/stop_curve_config.json`

You can override it in training with:
```bash
python lego/stop_module/src/train_standalone.py --config /path/to/config.json
```
