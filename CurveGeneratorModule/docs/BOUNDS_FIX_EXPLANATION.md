# Bounds Fix Explanation

## Problem Identified

Curves were going outside the 128x128 pixel image bounds due to several issues:

### 1. **Control Points Exceeding Bounds**
- **Issue**: Control points (p1, p2) were calculated without bounds checking
- **Root Cause**: `spread = [h, w] * bezier_spread * curvature_factor` could create very large spreads
  - Example: With `curvature_factor=2.0` and `bezier_spread=0.5`, spread = `[128, 128] * 0.5 * 2.0 = [128, 128]`
  - Control points could be `center ± 128 * 0.8 = center ± 102.4`, easily exceeding bounds
- **Location**: Lines 106, 113-117 in original code

### 2. **Bezier Curves Extending Beyond Control Points**
- **Issue**: Even if control points are within bounds, the bezier curve itself can extend beyond them
- **Root Cause**: Cubic Bezier curves are not guaranteed to stay within the convex hull of their control points
- **Location**: Line 120 - no clipping after generating bezier points

### 3. **No Margin Accounting for Thickness**
- **Issue**: Margin didn't account for line thickness, so thick lines could extend beyond bounds
- **Root Cause**: Only `bezier_margin` was used, not accounting for `max_thickness`

### 4. **Fallback Point Generation**
- **Issue**: Fallback `p3 = [h - p0[0], w - p0[1]]` could create points outside bounds
- **Location**: Line 102 in original code

## Solutions Implemented

### 1. **Added Helper Methods**
```python
def _effective_margin(self, max_thickness=10):
    """Calculate effective margin accounting for line thickness."""
    return self.bezier_margin + max_thickness

def _clip_point(self, p, margin=None):
    """Clip a point to stay within image bounds with margin."""
    ...

def _clip_points(self, pts, margin=None):
    """Clip all points to stay within image bounds with margin."""
    ...
```

### 2. **Limited Spread Based on Available Space**
```python
# Calculate maximum available spread from center to bounds
max_spread_y = min(center[0] - m, self.h - m - center[0])
max_spread_x = min(center[1] - m, self.w - m - center[1])
max_spread = np.array([max_spread_y, max_spread_x], dtype=np.float32)

# Use the smaller of desired spread and available space (70% of max for safety)
spread = np.minimum(desired_spread, max_spread * 0.7)
```

### 3. **Clip Control Points**
```python
# CRITICAL: Clip control points to bounds
p1 = self._clip_point(p1, margin=m)
p2 = self._clip_point(p2, margin=m)
```

### 4. **Clip Generated Bezier Points**
```python
# Generate bezier curve points
ts = np.linspace(0, 1, n_samples, dtype=np.float32)
pts = np.stack([_cubic_bezier(p0, p1, p2, p3, t) for t in ts], axis=0)

# CRITICAL: Clip all generated points to bounds
# Bezier curves can extend beyond their control points!
pts = self._clip_points(pts, margin=m)
```

### 5. **Fixed Fallback Point Generation**
```python
if p3 is None:
    # Fallback: place p3 opposite to p0
    p3 = np.array([self.h - p0[0], self.w - p0[1]], dtype=np.float32)
    p3 = self._clip_point(p3, margin=m)  # Now clipped!
    # Ensure minimum distance
    if np.linalg.norm(p0 - p3) < self.bezier_min_distance:
        # Create a point at minimum distance
        direction = p3 - p0
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            direction = direction / norm
            p3 = p0 + direction * self.bezier_min_distance
            p3 = self._clip_point(p3, margin=m)
```

## Result

All curves are now guaranteed to stay within bounds:
- Control points are clipped to `[margin, h-margin-1]` and `[margin, w-margin-1]`
- Generated bezier points are clipped after generation
- Effective margin accounts for line thickness
- Spread is limited based on available space

## Testing

You can verify the fix by checking that all generated points are within bounds:
```python
pts = generator._generate_bezier_points(curvature_factor=2.0, allow_self_cross=True, self_cross_prob=0.4)
assert np.all(pts >= margin) and np.all(pts < [h-margin, w-margin])
```

