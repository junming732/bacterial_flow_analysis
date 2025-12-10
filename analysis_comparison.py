#!/usr/bin/env python3
"""
Script 1: Quantitative Analysis
Compares horizontal net flow between untreated (REF) and treated (RIF) samples.
Generates time-series plots showing the point of divergence.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

def compute_weighted_direction(flow, flow_magnitude, threshold_percentile=80):
    """Computes the net flow direction weighted by magnitude for top 20% active pixels."""
    if np.any(flow_magnitude > 0):
        threshold = np.percentile(flow_magnitude[flow_magnitude > 0], threshold_percentile)
    else:
        threshold = 0

    mask = flow_magnitude > threshold

    if not np.any(mask) or np.sum(mask) < 10:
        return 0, 0, 0

    flow_x = flow[..., 0][mask]
    flow_y = flow[..., 1][mask]
    mag = flow_magnitude[mask]

    # Weighted average direction
    weighted_u = np.sum(flow_x * mag) / np.sum(mag)
    weighted_v = np.sum(flow_y * mag) / np.sum(mag)
    avg_mag = np.mean(mag)

    return weighted_u, weighted_v, avg_mag

def analyze_directory(data_path, condition_name):
    """Iterates through images and calculates flow metrics."""
    img_files = sorted(list(data_path.glob('*.tiff')) + list(data_path.glob('*.tif')))

    if len(img_files) < 2:
        print(f"Warning: Not enough images found in {data_path}")
        return []

    print(f"Quantifying {condition_name} from {data_path.name} ({len(img_files)} frames)...")
    results = []

    # Analyze every 2nd frame
    max_frames = min(len(img_files) - 1, 110)

    for frame_idx in range(0, max_frames, 2):
        img1 = cv2.imread(str(img_files[frame_idx]), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(img_files[frame_idx + 1]), cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None: continue

        flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

        u, v, mag = compute_weighted_direction(flow, flow_mag)

        results.append({
            'condition': condition_name,
            'frame': frame_idx,
            'flow_u': u,
            'flow_v': v,
            'magnitude': mag
        })

    return results

def plot_comparison(ref_df, rif_df, output_file='results_comparison.png'):
    """Plots horizontal flow comparison and difference."""
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 1], hspace=0.3)

    # Panel 1: Trajectories
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(ref_df['frame'], ref_df['flow_u'], 'o-', linewidth=2, color='#2171b5', label='REF (Untreated)')
    ax1.plot(rif_df['frame'], rif_df['flow_u'], 's-', linewidth=2, color='#e31a1c', label='RIF (Treated)')
    ax1.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Net Horizontal Flow (+Right / -Left)', fontsize=12)
    ax1.set_title('Bacterial Growth Direction: Quantitative Comparison', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Difference
    ax2 = fig.add_subplot(gs[1])
    min_len = min(len(ref_df), len(rif_df))
    frames = ref_df['frame'].values[:min_len]
    diff = rif_df['flow_u'].values[:min_len] - ref_df['flow_u'].values[:min_len]

    noise_threshold = np.std(diff[0:10]) * 2 if len(diff) > 10 else 0.5

    ax2.plot(frames, diff, 'o-', color='purple', linewidth=2)
    ax2.fill_between(frames, -noise_threshold, noise_threshold, color='gray', alpha=0.2, label='Noise Baseline')

    significant_indices = np.where(np.abs(diff) > noise_threshold)[0]
    if len(significant_indices) > 0:
        first_sig = frames[significant_indices[0]]
        ax2.axvline(first_sig, color='orange', linestyle='--', linewidth=2)
        ax2.text(first_sig + 2, max(diff)*0.8, f'Divergence\nFrame {first_sig}',
                 bbox=dict(facecolor='orange', alpha=0.3))

    ax2.set_xlabel('Time (Frame)', fontsize=12)
    ax2.set_ylabel('Difference (RIF - REF)', fontsize=12)
    ax2.axhline(0, color='black', linestyle='--')
    ax2.grid(True, alpha=0.3)

    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Quantitative plots saved to {output_file}")

def main():
    base_path = Path('.')
    # Updated to match specific folder names
    ref_path = base_path / 'data' / 'REF_raw_data101_110'
    rif_path = base_path / 'data' / 'RIF10_raw_data201_210'

    if not ref_path.exists() or not rif_path.exists():
        print(f"Error: Data folders not found.")
        print(f"Looking for: {ref_path} and {rif_path}")
        return

    ref_data = analyze_directory(ref_path, 'REF')
    rif_data = analyze_directory(rif_path, 'RIF')

    if ref_data and rif_data:
        plot_comparison(pd.DataFrame(ref_data), pd.DataFrame(rif_data))

if __name__ == '__main__':
    main()