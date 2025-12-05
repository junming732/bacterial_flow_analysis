#!/usr/bin/env python3
"""
Debug script to check optical flow direction values
"""

import numpy as np
import cv2
from pathlib import Path

base_path = Path('/home/junming/nobackup_junming')

# Check one RIF10 position
data_path = base_path / 'RIF10_raw_data201_210' / 'Pos201' / 'aphase'
img_files = sorted(data_path.glob('img_*.tiff'))

# Load frame 80 and 81
img1 = cv2.imread(str(img_files[80]), cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(str(img_files[81]), cv2.IMREAD_GRAYSCALE)

# Compute optical flow
flow = cv2.calcOpticalFlowFarneback(
    img1, img2, None,
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
)

flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

# Get significant flow regions
threshold = np.percentile(flow_magnitude[flow_magnitude > 0], 60)
mask = flow_magnitude > threshold

flow_x = flow[..., 0][mask]
flow_y = flow[..., 1][mask]

print("="*70)
print("OPTICAL FLOW DEBUG - RIF10 Pos201 Frame 80")
print("="*70)
print(f"\nFlow X (horizontal):")
print(f"  Min: {flow_x.min():.4f}")
print(f"  Max: {flow_x.max():.4f}")
print(f"  Mean: {np.mean(flow_x):.4f}")
print(f"  Median: {np.median(flow_x):.4f}")
print(f"  --> Median {'POSITIVE = RIGHTWARD' if np.median(flow_x) > 0 else 'NEGATIVE = LEFTWARD'}")

print(f"\nFlow Y (vertical):")
print(f"  Min: {flow_y.min():.4f}")
print(f"  Max: {flow_y.max():.4f}")
print(f"  Mean: {np.mean(flow_y):.4f}")
print(f"  Median: {np.median(flow_y):.4f}")
print(f"  --> Median {'POSITIVE = DOWNWARD' if np.median(flow_y) > 0 else 'NEGATIVE = UPWARD'}")

# Check what direction this gives
mean_u = np.mean(flow_x)
mean_v = np.mean(flow_y)

abs_u = abs(mean_u)
abs_v = abs(mean_v)

print(f"\nDirection determination (using MEAN):")
print(f"  Mean U = {mean_u:.4f}")
print(f"  Mean V = {mean_v:.4f}")
print(f"  |U| = {abs_u:.4f}")
print(f"  |V| = {abs_v:.4f}")
print(f"  Ratio |U|/|V| = {abs_u/(abs_v+1e-10):.2f}x")

if abs_u > abs_v * 1.5:
    result = "→ RIGHTWARD" if mean_u > 0 else "← LEFTWARD"
elif abs_v > abs_u * 1.5:
    result = "↓ DOWNWARD" if mean_v > 0 else "↑ UPWARD"
else:
    if mean_u > 0 and mean_v < 0:
        result = "↗ right-up"
    elif mean_u < 0 and mean_v < 0:
        result = "↖ left-up"
    elif mean_u > 0 and mean_v > 0:
        result = "↘ right-down"
    else:
        result = "↙ left-down"

print(f"\nFINAL RESULT: {result}")
print("="*70)