#!/usr/bin/env python3
"""
Time-series analysis of flow direction
One comprehensive plot per position showing directional trends over time
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def compute_weighted_direction(flow, flow_magnitude, threshold_percentile=80):
    """Compute magnitude-weighted flow direction"""
    threshold = np.percentile(flow_magnitude[flow_magnitude > 0], threshold_percentile) if np.any(flow_magnitude > 0) else 0
    mask = flow_magnitude > threshold

    if not np.any(mask) or np.sum(mask) < 10:
        return 0, 0, 0

    flow_x = flow[..., 0][mask]
    flow_y = flow[..., 1][mask]
    mag = flow_magnitude[mask]

    # Magnitude-weighted direction
    weighted_u = np.sum(flow_x * mag) / np.sum(mag)
    weighted_v = np.sum(flow_y * mag) / np.sum(mag)
    avg_mag = np.mean(mag)

    return weighted_u, weighted_v, avg_mag


def create_directional_timeseries(condition, position, base_path, output_path):
    """Create comprehensive time-series plot for one position"""

    # Correct path construction
    if condition == 'REF':
        folder = 'REF_raw_data101_110'
    else:
        folder = 'RIF10_raw_data201_210'

    data_path = base_path / folder / f'Pos{position}' / 'aphase'

    if not data_path.exists():
        print(f"  Skipping {condition} Pos{position} - not found")
        return

    img_files = sorted(data_path.glob('img_*.tiff'))

    if len(img_files) < 2:
        print(f"  Skipping {condition} Pos{position} - insufficient frames")
        return

    # Analyze all frames (or subsample for speed)
    max_frames = min(len(img_files) - 1, 110)

    times = []
    flow_u = []  # Horizontal (left-right)
    flow_v = []  # Vertical (up-down)
    magnitudes = []

    print(f"  Processing {condition} Pos{position}: {max_frames} frames...")

    for frame_idx in range(0, max_frames, 2):  # Every 2nd frame for speed
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

        # Get weighted direction
        u, v, mag = compute_weighted_direction(flow, flow_mag, threshold_percentile=80)

        times.append(frame_idx)
        flow_u.append(u)
        flow_v.append(v)
        magnitudes.append(mag)

    if len(times) < 5:
        print(f"  Insufficient data for {condition} Pos{position}")
        return

    # Create comprehensive plot
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)

    # Panel 1: Horizontal flow (left-right) over time
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(times, flow_u, 'o-', linewidth=2, markersize=4, color='#2171b5', label='Horizontal flow')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.fill_between(times, 0, flow_u, where=[u > 0 for u in flow_u],
                     alpha=0.3, color='red', label='Rightward')
    ax1.fill_between(times, 0, flow_u, where=[u < 0 for u in flow_u],
                     alpha=0.3, color='blue', label='Leftward')

    ax1.set_xlabel('Time (frame)', fontsize=12)
    ax1.set_ylabel('Horizontal Flow\n(+Right, -Left)', fontsize=12)
    ax1.set_title(f'{condition} Pos{position} - Horizontal Direction Over Time',
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Add stats
    mean_u = np.mean(flow_u)
    dominant_h = "RIGHTWARD →" if mean_u > 0 else "← LEFTWARD"
    ax1.text(0.02, 0.98, f"Mean: {mean_u:.3f}\nDominant: {dominant_h}",
            transform=ax1.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    # Panel 2: Vertical flow (up-down) over time
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(times, flow_v, 's-', linewidth=2, markersize=4, color='#238b45', label='Vertical flow')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.fill_between(times, 0, flow_v, where=[v > 0 for v in flow_v],
                     alpha=0.3, color='red', label='Downward')
    ax2.fill_between(times, 0, flow_v, where=[v < 0 for v in flow_v],
                     alpha=0.3, color='blue', label='Upward')

    ax2.set_xlabel('Time (frame)', fontsize=12)
    ax2.set_ylabel('Vertical Flow\n(+Down, -Up)', fontsize=12)
    ax2.set_title('Vertical Direction Over Time', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)

    mean_v = np.mean(flow_v)
    dominant_v = "DOWNWARD ↓" if mean_v > 0 else "↑ UPWARD"
    ax2.text(0.02, 0.98, f"Mean: {mean_v:.3f}\nDominant: {dominant_v}",
            transform=ax2.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    # Panel 3: Flow magnitude over time
    ax3 = fig.add_subplot(gs[2, :])
    ax3.plot(times, magnitudes, 'o-', linewidth=2, markersize=4, color='#e31a1c')
    ax3.set_xlabel('Time (frame)', fontsize=12)
    ax3.set_ylabel('Average Flow Magnitude', fontsize=12)
    ax3.set_title('Flow Magnitude Over Time', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Panel 4: 2D trajectory (cumulative)
    ax4 = fig.add_subplot(gs[3, 0])
    cumsum_u = np.cumsum(flow_u)
    cumsum_v = np.cumsum(flow_v)

    # Color by time
    scatter = ax4.scatter(cumsum_u, cumsum_v, c=times, cmap='viridis',
                         s=30, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax4.plot(cumsum_u, cumsum_v, 'k-', alpha=0.3, linewidth=1)

    ax4.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax4.axvline(x=0, color='gray', linestyle='--', linewidth=1)
    ax4.set_xlabel('Cumulative Horizontal (Right→)', fontsize=11)
    ax4.set_ylabel('Cumulative Vertical (Down↓)', fontsize=11)
    ax4.set_title('Cumulative Flow Trajectory', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='Time (frame)')

    # Panel 5: Direction vector plot (polar-like)
    ax5 = fig.add_subplot(gs[3, 1])

    # Calculate angles and magnitudes
    angles = [np.arctan2(v, u) * 180 / np.pi for u, v in zip(flow_u, flow_v)]
    mags = [np.sqrt(u**2 + v**2) for u, v in zip(flow_u, flow_v)]

    scatter2 = ax5.scatter(angles, mags, c=times, cmap='viridis',
                          s=30, alpha=0.7, edgecolors='black', linewidth=0.5)

    ax5.set_xlabel('Direction (degrees)\n0°=Right, 90°=Down, ±180°=Left', fontsize=10)
    ax5.set_ylabel('Flow Magnitude', fontsize=11)
    ax5.set_title('Direction vs Magnitude', fontsize=13, fontweight='bold')
    ax5.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Right')
    ax5.axvline(x=180, color='blue', linestyle='--', linewidth=1, alpha=0.5, label='Left')
    ax5.axvline(x=-180, color='blue', linestyle='--', linewidth=1, alpha=0.5)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax5, label='Time (frame)')

    plt.suptitle(f'{condition} - Position {position} - Complete Directional Analysis',
                fontsize=16, fontweight='bold')

    # Save
    filename = output_path / f'timeseries_{condition}_Pos{position}.png'
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"    ✓ Saved {filename.name}")


def main():
    base_path = Path('/home/junming/nobackup_junming')
    output_path = Path('/home/junming/private/rearch_methodology')

    timeseries_dir = output_path / 'directional_timeseries'
    timeseries_dir.mkdir(exist_ok=True)

    print("="*70)
    print("Creating directional time-series plots")
    print("="*70)

    conditions = {
        'REF': range(101, 111),
        'RIF10': range(201, 211)
    }

    for condition, positions in conditions.items():
        print(f"\n{condition}:")
        for pos in positions:
            create_directional_timeseries(condition, pos, base_path, timeseries_dir)

    print("\n" + "="*70)
    print(f"Results saved to: {timeseries_dir}")
    print(f"Generated {len(list(timeseries_dir.glob('*.png')))} time-series plots")
    print("\nEach plot shows:")
    print("  1. Horizontal flow over time (left-right tendency)")
    print("  2. Vertical flow over time (up-down tendency)")
    print("  3. Flow magnitude evolution")
    print("  4. Cumulative trajectory")
    print("  5. Direction distribution")
    print("="*70)


if __name__ == '__main__':
    main()