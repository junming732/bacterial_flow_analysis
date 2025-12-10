#!/usr/bin/env python3
"""
Script 2: Qualitative Visualization
Generates vector field heatmaps with large arrows to visualize flow dynamics.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

def create_custom_colormap():
    """Blue (low) -> Yellow -> Red (high) colormap."""
    colors = ['#08306b', '#2171b5', '#6baed6', '#c6dbef',
              '#ffffcc', '#ffeda0', '#fed976', '#feb24c',
              '#fd8d3c', '#fc4e2a', '#e31a1c', '#bd0026']
    return mcolors.LinearSegmentedColormap.from_list('bacterial_growth', colors, N=256)

def analyze_flow_direction(flow, flow_magnitude):
    """Determine dominant direction string for labels."""
    threshold = np.percentile(flow_magnitude[flow_magnitude > 0], 80) if np.any(flow_magnitude > 0) else 0
    mask = flow_magnitude > threshold
    if not np.any(mask): return "No significant flow"

    flow_x = flow[..., 0][mask]
    flow_y = flow[..., 1][mask]
    mag = flow_magnitude[mask]

    w_u = np.sum(flow_x * mag) / np.sum(mag)
    w_v = np.sum(flow_y * mag) / np.sum(mag)

    if abs(w_u) > abs(w_v) * 1.5:
        return "RIGHTWARD" if w_u > 0 else "LEFTWARD"
    elif abs(w_v) > abs(w_u) * 1.5:
        return "DOWNWARD" if w_v > 0 else "UPWARD"
    return "Mixed Direction"

def create_vector_viz(flow_mag, flow, title, filename):
    """Create 2x2 visualization panel."""
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.2, wspace=0.2)
    custom_cmap = create_custom_colormap()

    # 1. Magnitude Heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(flow_mag, cmap=custom_cmap)
    ax1.set_title(f'{title}\nMagnitude Heatmap', fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    # 2. Large Vector Field
    ax2 = fig.add_subplot(gs[0, 1])
    h, w = flow_mag.shape
    subsample = 20 # Sparser arrows for visibility
    y, x = np.meshgrid(np.arange(subsample//2, h, subsample),
                       np.arange(subsample//2, w, subsample), indexing='ij')

    U = flow[y, x, 0]
    V = flow[y, x, 1]
    M = flow_mag[y, x]

    ax2.imshow(flow_mag, cmap='gray', alpha=0.5)
    ax2.quiver(x, y, U, V, M, cmap='hot', scale=50, width=0.003, headwidth=4)
    ax2.set_title('Vector Field (Large Arrows)', fontweight='bold')
    ax2.axis('off')

    # Label Net Direction
    net_dir = analyze_flow_direction(flow, flow_mag)
    ax2.text(0.02, 0.02, f"Net: {net_dir}", transform=ax2.transAxes,
             bbox=dict(facecolor='yellow', alpha=0.8), fontweight='bold')

    # 3 & 4. Components (Horizontal/Vertical)
    for idx, (component, label) in enumerate([(0, 'Horizontal'), (1, 'Vertical')]):
        ax = fig.add_subplot(gs[1, idx])
        comp_data = flow[..., component]
        vmax = max(abs(comp_data.min()), abs(comp_data.max()))
        im = ax.imshow(comp_data, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(f'{label} Component', fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()

def main():
    base_path = Path('.')
    output_dir = base_path / 'results_viz'
    output_dir.mkdir(exist_ok=True)

    # Map condition names to specific folder names
    folder_map = {
        'REF': 'REF_raw_data101_110',
        'RIF': 'RIF10_raw_data201_210'
    }

    # Frames to visualize (Key phases)
    key_frames = [0, 10, 20, 30, 40, 50, 60]

    for condition, folder_name in folder_map.items():
        data_path = base_path / 'data' / folder_name
        if not data_path.exists():
            print(f"Skipping {condition}: Folder {folder_name} not found.")
            continue

        img_files = sorted(list(data_path.glob('*.tiff')) + list(data_path.glob('*.tif')))
        print(f"Visualizing {condition} from {folder_name}...")

        for frame_idx in key_frames:
            if frame_idx >= len(img_files) - 1: continue

            img1 = cv2.imread(str(img_files[frame_idx]), cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(str(img_files[frame_idx+1]), cv2.IMREAD_GRAYSCALE)
            if img1 is None: continue

            flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)

            outfile = output_dir / f'viz_{condition}_frame{frame_idx:03d}.png'
            create_vector_viz(mag, flow, f'{condition} Frame {frame_idx}', outfile)

    print(f"Visualizations saved to {output_dir}/")

if __name__ == '__main__':
    main()