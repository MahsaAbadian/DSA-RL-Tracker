# Strong Foundation Configuration Notes

## Stage 1: Diverse Shapes (Foundation)

This stage is designed to expose the RL agent to **all kinds of curve shapes** including:
- **Ribbon/cancer ribbon shapes** (self-crossing loops with tapering)
- **S-curves** (high curvature)
- **Straight segments** (low curvature)
- **Complex angles** (high curvature_factor)
- **Variable width paths** (wide_to_narrow tapering)

### Key Parameters for Diversity:

1. **`allow_self_cross: true`** + **`self_cross_prob: 0.4`**
   - Enables loops and ribbon shapes (like cancer ribbons)
   - 40% of curves will have self-crossing behavior

2. **`curvature_factor: 2.0`**
   - High curvature allows for complex shapes, loops, and sharp angles
   - Combined with self-crossing, creates ribbon-like paths

3. **`width_variation: "none"`**
   - Constant width for Stage 1 (foundation)
   - Focus on shape diversity, not width/intensity variations

4. **`intensity_variation: "none"`**
   - Constant intensity for Stage 1 (foundation)
   - Variations introduced in later stages

5. **Increased `control_point_spread: 0.5`** and **`control_point_factor: 0.8`**
   - Allows more extreme control point positions
   - Creates more diverse curve shapes

## Bezier Curves vs Other Curve Types

**Current approach: Cubic Bezier curves**

Cubic Bezier curves are actually quite powerful and can represent:
- ✅ Straight lines (low curvature_factor)
- ✅ Smooth curves (medium curvature_factor)
- ✅ S-curves (high curvature_factor)
- ✅ Loops/ribbons (high curvature_factor + allow_self_cross)
- ✅ Sharp turns (very high curvature_factor)

**For a path finder (roads, vessels, etc.), Bezier curves should be sufficient** because:
1. They're smooth and continuous (good for paths)
2. They can represent most real-world paths
3. They're computationally efficient
4. They're flexible enough with the right parameters

**Potential enhancements** (if needed later):
- **Multi-segment Bezier paths**: Chain multiple Bezier curves for paths with sharp corners
- **B-splines**: For even smoother, more controlled curves
- **Catmull-Rom splines**: For paths that pass through specific points

However, for now, **cubic Bezier curves with diverse parameters should work well** for training a general path finder.

## Stage Progression

- **Stage 1**: Diverse shapes (foundation) - all curve types, ribbon shapes, high visibility
- **Stage 2**: Complex curves - builds on stage 1 with more challenges
- **Stages 3-11**: Progressive difficulty (thin paths, low intensity, noise, etc.)

The key is that **Stage 1 exposes the agent to the full diversity of shapes** it will encounter, so it learns robust path-following from the start.

