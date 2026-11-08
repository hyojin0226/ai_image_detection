"""Grad-CAM heatmaps for the final 64-frame model; no surrogate model."""
import numpy as np
from PIL import Image


def positive_gradcam(activation, gradient):
    activation, gradient = np.asarray(activation), np.asarray(gradient)
    if activation.shape != gradient.shape or activation.ndim != 3:
        raise ValueError('Invalid activation/gradient dimensions')
    if not np.isfinite(activation).all() or not np.isfinite(gradient).all():
        raise ValueError('Nonfinite Grad-CAM values')
    heat = np.maximum(np.sum(activation * gradient.mean(axis=(0, 1)), axis=-1), 0)
    maximum = float(heat.max())
    # Do not manufacture a colored explanation for zero or saturated gradients.
    return (heat / maximum).astype(np.float32) if maximum > 1e-20 else None


def overlay_heatmap(frame, heat):
    base = Image.fromarray(frame)
    h = np.asarray(Image.fromarray(heat).resize(base.size, Image.Resampling.BILINEAR)).clip(0, 1)
    h = np.where(np.any(frame > 0, axis=-1), h, 0)
    color = np.stack([np.ones_like(h), 1 - h, np.zeros_like(h)], axis=-1) * 255
    alpha = (.6 * h)[..., None]
    overlay = np.rint(frame * (1 - alpha) + color * alpha).clip(0, 255).astype(np.uint8)
    return Image.fromarray(overlay), h


def strongest_zone(heat, frame):
    # Black padding contains no picture; exclude it from descriptive localization.
    mask = np.any(frame > 0, axis=-1)
    weights = np.where(mask, heat, 0)
    if not weights.any():
        return None
    yy, xx = np.nonzero(mask)
    y0, y1, x0, x1 = yy.min(), yy.max() + 1, xx.min(), xx.max() + 1
    zones = []
    for row in range(3):
        for col in range(3):
            a, b = y0 + (y1-y0)*row//3, y0 + (y1-y0)*(row+1)//3
            c, d = x0 + (x1-x0)*col//3, x0 + (x1-x0)*(col+1)//3
            zones.append((float(weights[a:b,c:d].sum()), row, col))
    mass, row, col = max(zones)
    if mass <= 0:
        return None
    return [['왼쪽 위', '위쪽 중앙', '오른쪽 위'],
            ['왼쪽 중앙', '중앙', '오른쪽 중앙'],
            ['왼쪽 아래', '아래쪽 중앙', '오른쪽 아래']][row][col]
