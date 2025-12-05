#!/usr/bin/env python3
"""
Enhanced bacterial growth visualization with LARGE visible vector fields
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def create_custom_colormap():
    colors = ['#08306b', '#2171b5', '#6baed6', '#c6dbef',
              '#ffffcc', '#ffeda0', '#fed976', '#feb24c',
              '#fd8d3c', '#fc4e2a', '#e31a1c', '#bd0026']
    cmap = mcolors.LinearSegmentedColormap.from_list('bacterial_growth', colors, N=256)
    return cmap


def analyze_flow_direction(flow, flow_magnitude):
    """Analyze net flow direction - weighted by magnitude"""
    # Only look at regions with SUBSTANTIAL flow (top 20%)
    threshold = np.percentile(flow_magnitude[flow_magnitude > 0], 80) if np.any(flow_magnitude > 0) else 0
    mask = flow_magnitude > threshold

    if not np.any(mask) or np.sum(mask) < 10:
        return "No significant flow"

    # Get flow components where there's actual significant movement
    flow_x = flow[..., 0][mask]  # Positive = RIGHT, Negative = LEFT
    flow_y = flow[..., 1][mask]  # Positive = DOWN, Negative = UP
    mag = flow_magnitude[mask]

    # WEIGHT BY MAGNITUDE - larger movements count more!
    weighted_u = np.sum(flow_x * mag) / np.sum(mag)
    weighted_v = np.sum(flow_y * mag) / np.sum(mag)

    # Determine dominant direction
    abs_u = abs(weighted_u)
    abs_v = abs(weighted_v)

    # If horizontal component dominates (>1.5x the vertical)
    if abs_u > abs_v * 1.5:
        if weighted_u > 0:
            return "→ RIGHTWARD"  # Positive U = RIGHT
        else:
            return "← LEFTWARD"   # Negative U = LEFT
    # If vertical component dominates
    elif abs_v > abs_u * 1.5:
        if weighted_v > 0:
            return "↓ DOWNWARD"   # Positive V = DOWN
        else:
            return "↑ UPWARD"     # Negative V = UP
    # Mixed direction - combine both
    else:
        if weighted_u > 0 and weighted_v < 0:
            return "↗ right-up"    # +U, -V
        elif weighted_u < 0 and weighted_v < 0:
            return "↖ left-up"     # -U, -V
        elif weighted_u > 0 and weighted_v > 0:
            return "↘ right-down"  # +U, +V
        else:
            return "↙ left-down"   # -U, +V


def create_large_vector_visualization(flow_magnitude, flow_vectors, title, condition):
    """
    Create 2x2 visualization with LARGE visible vectors
    """
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.25)

    custom_cmap = create_custom_colormap()

    # Panel 1: Flow magnitude with custom colormap
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(flow_magnitude, cmap=custom_cmap, interpolation='bilinear')
    ax1.set_title(f'{title}\nFlow Magnitude Heatmap', fontsize=13, fontweight='bold')
    ax1.axis('off')
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046)
    cbar1.set_label('Flow Magnitude', fontsize=11)

    # Panel 2: LARGE vector field (like vector_analysis.py but bigger)
    ax2 = fig.add_subplot(gs[0, 1])
    h, w = flow_magnitude.shape

    # Sparse sampling for very large visible arrows
    subsample = 20
    y_coords = np.arange(subsample//2, h, subsample)
    x_coords = np.arange(subsample//2, w, subsample)

    Y, X = np.meshgrid(y_coords, x_coords, indexing='ij')

    U = flow_vectors[Y, X, 0]
    V = flow_vectors[Y, X, 1]
    M = flow_magnitude[Y, X]

    # Dark background for contrast (like the working version)
    ax2.imshow(flow_magnitude, cmap='gray', vmin=0, vmax=np.percentile(flow_magnitude, 95), alpha=0.5)

    # LARGE visible arrows with settings from working version
    quiver = ax2.quiver(X, Y, U, V, M,
                       cmap='hot',
                       scale=50,
                       width=0.003,
                       headwidth=3,
                       headlength=4,
                       alpha=0.8)

    ax2.set_title('Vector Field (Large Arrows)\nDirection & Magnitude', fontsize=13, fontweight='bold')
    ax2.axis('off')
    cbar2 = plt.colorbar(quiver, ax=ax2, fraction=0.046)
    cbar2.set_label('Flow Magnitude', fontsize=11)

    # Net direction - clean display
    net_dir = analyze_flow_direction(flow_vectors, flow_magnitude)

    ax2.text(0.02, 0.02, f"Net: {net_dir}",
            transform=ax2.transAxes,
            fontsize=14,
            verticalalignment='bottom',
            fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow',
                     alpha=0.95, edgecolor='black', linewidth=2))

    # Panel 3: Horizontal component (left-right)
    ax3 = fig.add_subplot(gs[1, 0])
    flow_x = flow_vectors[..., 0]
    vmax_x = max(abs(flow_x.min()), abs(flow_x.max()))

    im3 = ax3.imshow(flow_x, cmap='RdBu_r', vmin=-vmax_x, vmax=vmax_x)
    ax3.set_title('Horizontal Component\n(Red=Right, Blue=Left)', fontsize=13, fontweight='bold')
    ax3.axis('off')
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046)
    cbar3.set_label('Horizontal Velocity', fontsize=11)

    # Calculate mean for significant flow only (top 20%)
    sig_mask = flow_magnitude > np.percentile(flow_magnitude[flow_magnitude>0], 80)
    mean_h = np.mean(flow_x[sig_mask])

    h_text = f"{'RIGHTWARD →' if mean_h > 0 else '← LEFTWARD'}"
    ax3.text(0.98, 0.02, h_text,
            transform=ax3.transAxes,
            fontsize=12,
            verticalalignment='bottom',
            horizontalalignment='right',
            fontweight='bold',
            bbox=dict(boxstyle='round',
                     facecolor='lightcoral' if mean_h > 0 else 'lightblue',
                     alpha=0.9))

    # Panel 4: Vertical component (up-down)
    ax4 = fig.add_subplot(gs[1, 1])
    flow_y = flow_vectors[..., 1]
    vmax_y = max(abs(flow_y.min()), abs(flow_y.max()))

    im4 = ax4.imshow(flow_y, cmap='RdBu_r', vmin=-vmax_y, vmax=vmax_y)
    ax4.set_title('Vertical Component\n(Red=Down, Blue=Up)', fontsize=13, fontweight='bold')
    ax4.axis('off')
    cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046)
    cbar4.set_label('Vertical Velocity', fontsize=11)

    # Calculate mean for significant flow only (top 20%)
    sig_mask = flow_magnitude > np.percentile(flow_magnitude[flow_magnitude>0], 80)
    mean_v = np.mean(flow_y[sig_mask])

    v_text = f"{'DOWNWARD ↓' if mean_v > 0 else '↑ UPWARD'}"
    ax4.text(0.98, 0.02, v_text,
            transform=ax4.transAxes,
            fontsize=12,
            verticalalignment='bottom',
            horizontalalignment='right',
            fontweight='bold',
            bbox=dict(boxstyle='round',
                     facecolor='lightcoral' if mean_v > 0 else 'lightblue',
                     alpha=0.9))

    plt.suptitle(f'{condition} - {title}', fontsize=15, fontweight='bold', y=0.995)

    return fig


def main():
    base_path = Path('/home/junming/nobackup_junming')
    output_path = Path('/home/junming/private/rearch_methodology')

    # New output directory with descriptive name
    viz_dir = output_path / 'large_vector_visualizations'
    viz_dir.mkdir(exist_ok=True)

    print("="*70)
    print("Creating visualizations with LARGE VISIBLE vectors")
    print("="*70)

    conditions = {
        'REF': ('REF_raw_data101_110', range(101, 111)),
        'RIF10': ('RIF10_raw_data201_210', range(201, 211))
    }

    folders = {
        'REF': 'REF_raw_data101_110',
        'RIF10': 'RIF10_raw_data201_210'
    }

    for condition in ['REF', 'RIF10']:
        print(f"\n{condition}:")
        for pos in conditions[condition][1]:
            data_path = base_path / folders[condition] / f'Pos{pos}' / 'aphase'

            if not data_path.exists():
                print(f"  Skipping Pos{pos} - not found")
                continue

            img_files = sorted(data_path.glob('img_*.tiff'))

            if len(img_files) < 2:
                continue

            print(f"  Processing Pos{pos}...")

            # Key frames spanning early, mid, late phases
            # Early: 0-20, Mid: 40-60, Late: 80-110
            key_frames = [0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]

            for frame_idx in key_frames:
                if frame_idx >= len(img_files) - 1:
                    continue

                img1 = cv2.imread(str(img_files[frame_idx]), cv2.IMREAD_GRAYSCALE)
                img2 = cv2.imread(str(img_files[frame_idx + 1]), cv2.IMREAD_GRAYSCALE)

                if img1 is None or img2 is None:
                    continue

                # Compute optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    img1, img2, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )

                flow_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

                # Create visualization
                fig = create_large_vector_visualization(
                    flow_mag,
                    flow,
                    f'Pos{pos} Frame {frame_idx}',
                    condition
                )

                filename = f'large_vectors_{condition}_Pos{pos}_frame{frame_idx:03d}.png'
                fig.savefig(viz_dir / filename, dpi=150, bbox_inches='tight')
                plt.close(fig)

                print(f"    ✓ Frame {frame_idx}")

    print("\n" + "="*70)
    print(f"Results saved to: {viz_dir}")
    print(f"Generated {len(list(viz_dir.glob('*.png')))} visualization files")
    print("\nFeatures:")
    print("  - LARGE visible vectors (subsample=30)")
    print("  - Clear directional indicators (arrows)")
    print("  - Horizontal/vertical component decomposition")
    print("  - Net direction clearly labeled")
    print("="*70)


if __name__ == '__main__':
    main()