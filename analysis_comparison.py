#!/usr/bin/env python3
"""
REF vs RIF10 Comparison - Time Series Analysis
Shows when antibiotic effects become detectable.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def compute_weighted_direction(flow, flow_magnitude, threshold_percentile=80):
    """Compute magnitude-weighted flow direction"""
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

    # Magnitude-weighted direction
    weighted_u = np.sum(flow_x * mag) / np.sum(mag)
    weighted_v = np.sum(flow_y * mag) / np.sum(mag)
    avg_mag = np.mean(mag)

    return weighted_u, weighted_v, avg_mag

def find_images_in_pos(pos_path):
    """Helper to find images whether they are in 'aphase' or directly in Pos folder"""
    # 1. Check for 'aphase' subdirectory (standard structure)
    aphase_path = pos_path / 'aphase'
    if aphase_path.exists():
        images = sorted(list(aphase_path.glob('*.tiff')) + list(aphase_path.glob('*.tif')))
        if len(images) > 10:
            return images

    # 2. If no 'aphase', check the Pos folder directly
    images = sorted(list(pos_path.glob('*.tiff')) + list(pos_path.glob('*.tif')))
    return images

def analyze_all_positions(condition, base_path):
    """Analyze all positions for one condition and return time-series data"""

    # Define folder names based on your structure
    if condition == 'REF':
        folder_name = 'REF_raw_data101_110'
    else:
        folder_name = 'RIF10_raw_data201_210'

    data_root = base_path / 'data' / folder_name

    if not data_root.exists():
        print(f"Error: Could not find folder {data_root}")
        return pd.DataFrame()

    # Find all Pos folders (Pos101, Pos201, etc.)
    pos_folders = sorted([p for p in data_root.glob('Pos*') if p.is_dir()])

    if not pos_folders:
        print(f"Warning: No 'Pos' folders found in {data_root}")
        return pd.DataFrame()

    all_data = []

    for pos_path in pos_folders:
        pos_name = pos_path.name
        img_files = find_images_in_pos(pos_path)

        if len(img_files) < 2:
            continue

        print(f"  Processing {condition} {pos_name} ({len(img_files)} frames)...")

        max_frames = min(len(img_files) - 1, 110)

        for frame_idx in range(0, max_frames, 2):  # Every 2nd frame
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

            all_data.append({
                'condition': condition,
                'position': pos_name,
                'frame': frame_idx,
                'flow_u': u,
                'flow_v': v,
                'magnitude': mag
            })

    return pd.DataFrame(all_data)

def create_comparison_plot(ref_df, rif_df, output_file):
    """Create comprehensive REF vs RIF10 comparison (5 Panels)"""

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)

    # --- Panel 1: Horizontal flow comparison (MAIN PANEL) ---
    ax1 = fig.add_subplot(gs[0, :])

    ref_grouped = ref_df.groupby('frame')['flow_u'].agg(['mean', 'std'])
    rif_grouped = rif_df.groupby('frame')['flow_u'].agg(['mean', 'std'])

    ax1.plot(ref_grouped.index, ref_grouped['mean'], 'o-', linewidth=3, markersize=6, color='#2171b5', label='REF (Untreated)', alpha=0.8)
    ax1.fill_between(ref_grouped.index, ref_grouped['mean'] - ref_grouped['std'], ref_grouped['mean'] + ref_grouped['std'], alpha=0.2, color='#2171b5')

    ax1.plot(rif_grouped.index, rif_grouped['mean'], 's-', linewidth=3, markersize=6, color='#e31a1c', label='RIF10 (Treated)', alpha=0.8)
    ax1.fill_between(rif_grouped.index, rif_grouped['mean'] - rif_grouped['std'], rif_grouped['mean'] + rif_grouped['std'], alpha=0.2, color='#e31a1c')

    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax1.set_ylabel('Horizontal Flow (+ Right, - Left)', fontsize=14, fontweight='bold')
    ax1.set_title('HORIZONTAL DIRECTION: REF vs RIF10 Over Time', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12, loc='best')
    ax1.grid(True, alpha=0.3)

    # Mark early detection window
    ax1.axvspan(0, 20, alpha=0.1, color='yellow')
    ax1.text(10, ax1.get_ylim()[1]*0.9, 'Early phase', ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # --- Panel 2: Vertical flow comparison ---
    ax2 = fig.add_subplot(gs[1, :])
    ref_grouped_v = ref_df.groupby('frame')['flow_v'].agg(['mean', 'std'])
    rif_grouped_v = rif_df.groupby('frame')['flow_v'].agg(['mean', 'std'])

    ax2.plot(ref_grouped_v.index, ref_grouped_v['mean'], 'o-', linewidth=3, markersize=6, color='#2171b5', label='REF', alpha=0.8)
    ax2.fill_between(ref_grouped_v.index, ref_grouped_v['mean'] - ref_grouped_v['std'], ref_grouped_v['mean'] + ref_grouped_v['std'], alpha=0.2, color='#2171b5')

    ax2.plot(rif_grouped_v.index, rif_grouped_v['mean'], 's-', linewidth=3, markersize=6, color='#e31a1c', label='RIF10', alpha=0.8)
    ax2.fill_between(rif_grouped_v.index, rif_grouped_v['mean'] - rif_grouped_v['std'], rif_grouped_v['mean'] + rif_grouped_v['std'], alpha=0.2, color='#e31a1c')

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax2.set_ylabel('Vertical Flow (+ Down, - Up)', fontsize=14, fontweight='bold')
    ax2.set_title('VERTICAL DIRECTION: REF vs RIF10 Over Time', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Flow magnitude comparison ---
    ax3 = fig.add_subplot(gs[2, :])
    ref_grouped_m = ref_df.groupby('frame')['magnitude'].agg(['mean', 'std'])
    rif_grouped_m = rif_df.groupby('frame')['magnitude'].agg(['mean', 'std'])

    ax3.plot(ref_grouped_m.index, ref_grouped_m['mean'], 'o-', linewidth=3, markersize=6, color='#2171b5', label='REF', alpha=0.8)
    ax3.fill_between(ref_grouped_m.index, ref_grouped_m['mean'] - ref_grouped_m['std'], ref_grouped_m['mean'] + ref_grouped_m['std'], alpha=0.2, color='#2171b5')

    ax3.plot(rif_grouped_m.index, rif_grouped_m['mean'], 's-', linewidth=3, markersize=6, color='#e31a1c', label='RIF10', alpha=0.8)
    ax3.fill_between(rif_grouped_m.index, rif_grouped_m['mean'] - rif_grouped_m['std'], rif_grouped_m['mean'] + rif_grouped_m['std'], alpha=0.2, color='#e31a1c')

    ax3.set_ylabel('Flow Magnitude', fontsize=14, fontweight='bold')
    ax3.set_title('FLOW MAGNITUDE (Movement Intensity)', fontsize=16, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # --- Panel 4: Difference plot (RIF10 - REF) ---
    ax4 = fig.add_subplot(gs[3, 0])

    common_frames = sorted(set(ref_grouped.index) & set(rif_grouped.index))
    diff_h = []
    diff_std = []

    for f in common_frames:
        ref_vals = ref_df[ref_df['frame'] == f]['flow_u'].values
        rif_vals = rif_df[rif_df['frame'] == f]['flow_u'].values

        # Mean difference and combined standard error
        mean_diff = rif_vals.mean() - ref_vals.mean()
        se_diff = np.sqrt((ref_vals.std()**2/len(ref_vals)) + (rif_vals.std()**2/len(rif_vals))) if len(ref_vals)>1 and len(rif_vals)>1 else 0

        diff_h.append(mean_diff)
        diff_std.append(se_diff * 2) # 95% CI

    ax4.plot(common_frames, diff_h, 'o-', linewidth=3, markersize=6, color='purple')
    ax4.fill_between(common_frames, np.array(diff_h) - np.array(diff_std), np.array(diff_h) + np.array(diff_std), alpha=0.3, color='purple')
    ax4.axhline(0, color='black', linestyle='--')
    ax4.set_xlabel('Time (frame)', fontsize=12, fontweight='bold')
    ax4.set_title('DIFFERENCE (RIF10 - REF)', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # --- Panel 5: Stats ---
    ax5 = fig.add_subplot(gs[3, 1])
    ax5.axis('off')

    summary_text = (
        f"STATISTICAL SUMMARY\n{'='*30}\n"
        f"REF Avg Horizontal:  {ref_df['flow_u'].mean():.4f}\n"
        f"RIF10 Avg Horizontal: {rif_df['flow_u'].mean():.4f}\n\n"
        f"REF Avg Magnitude:   {ref_df['magnitude'].mean():.4f}\n"
        f"RIF10 Avg Magnitude:  {rif_df['magnitude'].mean():.4f}\n"
        f"{'='*30}\n"
        f"Interpretation:\n"
        f"{'⚠ RIF10 HIGHER FLOW' if rif_df['magnitude'].mean() > ref_df['magnitude'].mean() else 'Similar Flow'}"
    )
    ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, fontsize=10, verticalalignment='top', family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('REF vs RIF10 - Comprehensive Analysis', fontsize=18, fontweight='bold')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to {output_file}")

def main():
    # Use current directory for reproducibility
    base_path = Path('.')

    print("="*70)
    print("REF vs RIF10 COMPARISON ANALYSIS")
    print("="*70)

    # Analyze REF
    ref_df = analyze_all_positions('REF', base_path)

    # Analyze RIF10
    rif_df = analyze_all_positions('RIF10', base_path)

    if len(ref_df) == 0 or len(rif_df) == 0:
        print("ERROR: Insufficient data. Please check 'data' folder structure.")
        return

    # Generate results
    create_comparison_plot(ref_df, rif_df, 'results_comparison.png')

    # Save CSVs
    ref_df.to_csv('results_ref.csv', index=False)
    rif_df.to_csv('results_rif.csv', index=False)
    print("CSV data saved.")

if __name__ == '__main__':
    main()