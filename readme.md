## Explainability Integrity

This project enforces **physical and anatomical constraints** on explanations:

- Lung-aware gating (CAM ∩ lung mask)
- Score-CAM for visually stable maps
- Energy-in-lungs metric (quantitative sanity check)
- Optional augmentation stability testing
- High-resolution overlays on original X-rays

> Heatmaps are **not treated as segmentations**. They are evidence visualizations,
> explicitly constrained and labeled with uncertainty.

### Metrics to Watch
- `energy_inside_lungs` ↑
- `energy_outside_lungs` ↓
- `entropy` moderate (too high = diffuse; too low = brittle)
