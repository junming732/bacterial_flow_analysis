#!/usr/bin/env python3
"""
Script 2: Qualitative Visualization
Generates 4-panel vector field visualizations with LARGE visible arrows.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# --- 1. Original Visualization Logic ---

def create_custom_colormap():
    colors = ['#08306b', '#2171b5', '#6baed6', '#c6dbef',
              '#ffffcc', '#ffeda0', '#fed976', '#feb24c',
              '#fd8d3c', '#fc4e2a', '#e31a1c', '#bd0026']
    cmap = mcolors.LinearSegmentedColormap.from_list('bacterial_growth', colors, N=256)
    return cmap

def analyze_flow_direction(flow, flow_magnitude):
    if np.any(flow_magnitude > 0):
        threshold = np.percentile(flow_magnitude[flow_magnitude > 0], 80)
    else:
        threshold = 0

    mask = flow_magnitude > threshold

    if not np.any(mask) or np.sum(mask) < 10:
        return "No significant flow"

    flow_x = flow[..., 0][mask]
    flow_y = flow[..., 1][mask]
    mag = flow_magnitude[mask]

    weighted_u = np.sum(flow_x * mag) / np.sum(mag)
    weighted_v = np.sum(flow_y * mag) / np.sum(mag)

    abs_u = abs(weighted_u)
    abs_v = abs(weighted_v)

    if abs_u > abs_v * 1.5:
        return "→ RIGHTWARD" if weighted_u > 0 else "← LEFTWARD"
    elif abs_v > abs_u * 1.5:
        return "↓ DOWNWARD" if weighted_v > 0 else "↑ UPWARD"
    else:
        if weighted_u > 0 and weighted_v < 0: return "↗ right-up"
        elif weighted_u < 0 and weighted_v < 0: return "↖ left-up"
        elif weighted_u > 0 and weighted_v > 0: return "↘ right-down"
        else: return "↙ left-down"

def create_large_vector_visualization(flow_magnitude, flow_vectors, title, condition, filename):
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.25)

    custom_cmap = create_custom_colormap()

    # Panel 1: Flow magnitude
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(flow_magnitude, cmap=custom_cmap, interpolation='bilinear')
    ax1.set_title(f'{title}\nFlow Magnitude Heatmap', fontsize=13, fontweight='bold')
    ax1.axis('off')
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046)
    cbar1.set_label('Flow Magnitude', fontsize=11)

    # Panel 2: LARGE vector field
    ax2 = fig.add_subplot(gs[0, 1])
    h, w = flow_magnitude.shape
    subsample = 20
    y_coords = np.arange(subsample//2, h, subsample)
    x_coords = np.arange(subsample//2, w, subsample)
    Y, X = np.meshgrid(y_coords, x_coords, indexing='ij')

    U = flow_vectors[Y, X, 0]
    V = flow_vectors[Y, X, 1]
    M = flow_magnitude[Y, X]

    ax2.imshow(flow_magnitude, cmap='gray', vmin=0, vmax=np.percentile(flow_magnitude, 95), alpha=0.5)
    quiver = ax2.quiver(X, Y, U, V, M, cmap='hot', scale=50, width=0.003, headwidth=3, headlength=4, alpha=0.8)

    ax2.set_title('Vector Field (Large Arrows)\nDirection & Magnitude', fontsize=13, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(quiver, ax=ax2, fraction=0.046)

    net_dir = analyze_flow_direction(flow_vectors, flow_magnitude)
    ax2.text(0.02, 0.02, f"Net: {net_dir}", transform=ax2.transAxes, fontsize=14, verticalalignment='bottom', fontweight='bold', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.95, edgecolor='black', linewidth=2))

    # Panel 3: Horizontal component
    ax3 = fig.add_subplot(gs[1, 0])
    flow_x = flow_vectors[..., 0]
    vmax_x = max(abs(flow_x.min()), abs(flow_x.max()))
    im3 = ax3.imshow(flow_x, cmap='RdBu_r', vmin=-vmax_x, vmax=vmax_x)
    ax3.set_title('Horizontal Component\n(Red=Right, Blue=Left)', fontsize=13, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046)

    # Panel 4: Vertical component
    ax4 = fig.add_subplot(gs[1, 1])
    flow_y = flow_vectors[..., 1]
    vmax_y = max(abs(flow_y.min()), abs(flow_y.max()))
    im4 = ax4.imshow(flow_y, cmap='RdBu_r', vmin=-vmax_y, vmax=vmax_y)
    ax4.set_title('Vertical Component\n(Red=Down, Blue=Up)', fontsize=13, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046)

    plt.suptitle(f'{condition} - {title}', fontsize=15, fontweight='bold', y=0.995)
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()

# --- 2. Robust Data Loading Logic ---

def find_image_sequence(base_path):
    if not base_path.exists(): return []
    images = sorted(list(base_path.glob('*.tiff')) + list(base_path.glob('*.tif')))
    if len(images) > 10: return images

    max_images = 0
    best_images = []
    aphase_folders = list(base_path.rglob('aphase'))
    search_dirs = aphase_folders if aphase_folders else list(base_path.rglob("*"))

    for p in search_dirs:
        if p.is_dir():
            imgs = sorted(list(p.glob('*.tiff')) + list(p.glob('*.tif')))
            if len(imgs) > max_images:
                max_images = len(imgs)
                best_images = imgs
    return best_images

def main():
    base_path = Path('.')
    output_dir = base_path / 'results_viz'
    output_dir.mkdir(exist_ok=True)

    folder_map = {'REF': 'REF_raw_data101_110', 'RIF': 'RIF10_raw_data201_210'}

    # FIXED: Extended range to include all frames up to 110
    key_frames = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]

    print("="*70)
    print("Generating Large Vector Visualizations")
    print("="*70)

    for condition, folder_name in folder_map.items():
        data_path = base_path / 'data' / folder_name
        img_files = find_image_sequence(data_path)

        if not img_files:
            print(f"Skipping {condition}: No images found in {folder_name}")
            continue

        print(f"Processing {condition} from {data_path}...")

        for frame_idx in key_frames:
            if frame_idx >= len(img_files) - 1: continue

            img1 = cv2.imread(str(img_files[frame_idx]), cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(str(img_files[frame_idx+1]), cv2.IMREAD_GRAYSCALE)

            if img1 is None: continue

            flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)

            outfile = output_dir / f'viz_{condition}_frame{frame_idx:03d}.png'
            create_large_vector_visualization(mag, flow, f'Frame {frame_idx}', condition, outfile)
            print(f"  Saved: {outfile.name}")

    print(f"\nVisualizations saved to {output_dir}/")

if __name__ == '__main__':
    main()