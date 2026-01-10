# How Curves Are Generated

## Cubic Bezier Curve Basics

A cubic Bezier curve uses **4 control points**:

- **p0** = Start point (on the curve)
- **p1** = First control point (pulls the curve)
- **p2** = Second control point (pulls the curve)
- **p3** = End point (on the curve)

The curve **does NOT pass through p1 and p2**, but they control how much the curve bends.

## Current Generation Process

### Step 1: Place Start and End Points (p0, p3)

```python
p0 = random_point(margin=m)  # Random point within margin
p3 = random_point(margin=m)  # Random point within margin, far enough from p0
```

### Step 2: Calculate Center

```python
center = (p0 + p3) / 2.0  # Midpoint between start and end
```

### Step 3: Calculate Maximum Available Spread

```python
# How far can control points go before hitting bounds?
max_spread_y = min(center[0] - m, self.h - m - center[0])
max_spread_x = min(center[1] - m, self.w - m - center[1])
max_spread = [max_spread_y, max_spread_x]
```

**Problem**: If p0/p3 are near edges, `max_spread` becomes very small!

### Step 4: Calculate Desired Spread

```python
desired_spread = [h, w] * bezier_spread * curvature_factor
# Example: [128, 128] * 0.5 * 2.0 = [128, 128]
```

### Step 5: Limit Spread (THE PROBLEM!)

```python
spread = min(desired_spread, max_spread * 0.7)  # 70% safety factor
```

**Problem**: This is too conservative! When p0/p3 are near edges, spread becomes tiny.

### Step 6: Generate Control Points

```python
p1 = center + random_direction * spread * bezier_factor
p2 = center + random_direction * spread * bezier_factor
```

### Step 7: Clip Control Points

```python
p1 = clip_to_bounds(p1)  # This can reduce curvature!
p2 = clip_to_bounds(p2)  # This can reduce curvature!
```

**Problem**: Clipping control points after generation reduces curvature.

### Step 8: Generate Bezier Points

```python
for t in [0, 0.001, 0.002, ..., 1.0]:
    point = cubic_bezier(p0, p1, p2, p3, t)
```

### Step 9: Clip All Bezier Points

```python
all_points = clip_to_bounds(all_points)
```

**Problem**: Clipping points flattens curves that extend beyond bounds.

## Why Curves Aren't Curvy Enough

1. **Conservative Safety Factor**: The `0.7` multiplier (70% of max) is too restrictive
2. **Edge Cases**: When p0/p3 are near edges, `max_spread` is tiny, limiting curvature
3. **Double Clipping**: Control points are clipped, then bezier points are clipped again
4. **No Adaptive Strategy**: Doesn't adjust strategy based on where p0/p3 are located

## Solutions to Consider

### Option 1: Increase Safety Factor

- Change `0.7` to `0.85` or `0.9` (more aggressive)
- Risk: More curves might need clipping

### Option 2: Adaptive Spread Calculation ✅ IMPLEMENTED

- If p0/p3 are near center → use larger spread (90% of max)
- If p0/p3 are near edges → use smaller spread but still reasonable (70% of max)
- Linearly interpolates between center and edge based on distance from image center
- This allows curves near center to be more curvy while edge curves stay safe

### Option 3: Rejection Sampling

- Generate control points
- Check if bezier curve stays within bounds
- If not, regenerate (up to N attempts)
- Only clip as last resort

### Option 4: Smarter Control Point Placement

- Instead of clipping after generation, calculate bounds-aware control points
- Place p1/p2 such that the resulting bezier curve stays within bounds
- Use mathematical bounds of bezier curves (convex hull property)

### Option 5: Use Bezier Curve Bounds Property

- Cubic Bezier curves stay within the convex hull of their control points
- If we ensure control points are within bounds, the curve should be too
- But clipping control points reduces curvature

## Recommended Approach

**Combine Option 2 + Option 4**:

1. Calculate spread more intelligently based on p0/p3 position
2. Place control points with bounds-awareness from the start
3. Use rejection sampling if needed for extreme cases
4. Only clip as absolute last resort

This way we get curvy curves that stay in bounds!
