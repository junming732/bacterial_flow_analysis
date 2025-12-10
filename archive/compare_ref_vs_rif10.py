#!/usr/bin/env python3
"""
REF vs RIF10 Comparison - Time Series Analysis
Shows when antibiotic effects become detectable
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


def analyze_all_positions(condition, positions, base_path):
    """Analyze all positions for one condition and return time-series data"""

    if condition == 'REF':
        folder = 'REF_raw_data101_110'
    else:
        folder = 'RIF10_raw_data201_210'

    all_data = []

    for pos in positions:
        data_path = base_path / folder / f'Pos{pos}' / 'aphase'

        if not data_path.exists():
            continue

        img_files = sorted(data_path.glob('img_*.tiff'))

        if len(img_files) < 2:
            continue

        print(f"  Processing {condition} Pos{pos}...")

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
                'position': pos,
                'frame': frame_idx,
                'flow_u': u,
                'flow_v': v,
                'magnitude': mag
            })

    return pd.DataFrame(all_data)


def create_comparison_plot(ref_df, rif_df, output_path):
    """Create comprehensive REF vs RIF10 comparison"""

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)

    # Panel 1: Horizontal flow comparison (MAIN PANEL)
    ax1 = fig.add_subplot(gs[0, :])

    # Calculate mean and std for each timepoint
    ref_grouped = ref_df.groupby('frame')['flow_u'].agg(['mean', 'std', 'count'])
    rif_grouped = rif_df.groupby('frame')['flow_u'].agg(['mean', 'std', 'count'])

    # Plot REF
    ax1.plot(ref_grouped.index, ref_grouped['mean'], 'o-',
             linewidth=3, markersize=6, color='#2171b5', label='REF (Untreated)', alpha=0.8)
    ax1.fill_between(ref_grouped.index,
                     ref_grouped['mean'] - ref_grouped['std'],
                     ref_grouped['mean'] + ref_grouped['std'],
                     alpha=0.2, color='#2171b5')

    # Plot RIF10
    ax1.plot(rif_grouped.index, rif_grouped['mean'], 's-',
             linewidth=3, markersize=6, color='#e31a1c', label='RIF10 (Treated)', alpha=0.8)
    ax1.fill_between(rif_grouped.index,
                     rif_grouped['mean'] - rif_grouped['std'],
                     rif_grouped['mean'] + rif_grouped['std'],
                     alpha=0.2, color='#e31a1c')

    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax1.set_xlabel('Time (frame)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Horizontal Flow (+ Right, - Left)', fontsize=14, fontweight='bold')
    ax1.set_title('HORIZONTAL DIRECTION: REF vs RIF10 Over Time',
                 fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12, loc='best')
    ax1.grid(True, alpha=0.3)

    # Mark early detection window
    ax1.axvspan(0, 20, alpha=0.1, color='yellow', label='Early detection window')
    ax1.text(10, ax1.get_ylim()[1]*0.9, 'Early phase\n(0-20 frames)',
            ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # Panel 2: Vertical flow comparison
    ax2 = fig.add_subplot(gs[1, :])

    ref_grouped_v = ref_df.groupby('frame')['flow_v'].agg(['mean', 'std'])
    rif_grouped_v = rif_df.groupby('frame')['flow_v'].agg(['mean', 'std'])

    ax2.plot(ref_grouped_v.index, ref_grouped_v['mean'], 'o-',
             linewidth=3, markersize=6, color='#2171b5', label='REF', alpha=0.8)
    ax2.fill_between(ref_grouped_v.index,
                     ref_grouped_v['mean'] - ref_grouped_v['std'],
                     ref_grouped_v['mean'] + ref_grouped_v['std'],
                     alpha=0.2, color='#2171b5')

    ax2.plot(rif_grouped_v.index, rif_grouped_v['mean'], 's-',
             linewidth=3, markersize=6, color='#e31a1c', label='RIF10', alpha=0.8)
    ax2.fill_between(rif_grouped_v.index,
                     rif_grouped_v['mean'] - rif_grouped_v['std'],
                     rif_grouped_v['mean'] + rif_grouped_v['std'],
                     alpha=0.2, color='#e31a1c')

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax2.set_xlabel('Time (frame)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Vertical Flow (+ Down, - Up)', fontsize=14, fontweight='bold')
    ax2.set_title('VERTICAL DIRECTION: REF vs RIF10 Over Time', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12, loc='best')
    ax2.grid(True, alpha=0.3)

    ax2.axvspan(0, 20, alpha=0.1, color='yellow')

    # Panel 3: Flow magnitude comparison
    ax3 = fig.add_subplot(gs[2, :])

    ref_grouped_m = ref_df.groupby('frame')['magnitude'].agg(['mean', 'std'])
    rif_grouped_m = rif_df.groupby('frame')['magnitude'].agg(['mean', 'std'])

    ax3.plot(ref_grouped_m.index, ref_grouped_m['mean'], 'o-',
             linewidth=3, markersize=6, color='#2171b5', label='REF', alpha=0.8)
    ax3.fill_between(ref_grouped_m.index,
                     ref_grouped_m['mean'] - ref_grouped_m['std'],
                     ref_grouped_m['mean'] + ref_grouped_m['std'],
                     alpha=0.2, color='#2171b5')

    ax3.plot(rif_grouped_m.index, rif_grouped_m['mean'], 's-',
             linewidth=3, markersize=6, color='#e31a1c', label='RIF10', alpha=0.8)
    ax3.fill_between(rif_grouped_m.index,
                     rif_grouped_m['mean'] - rif_grouped_m['std'],
                     rif_grouped_m['mean'] + rif_grouped_m['std'],
                     alpha=0.2, color='#e31a1c')

    ax3.set_xlabel('Time (frame)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Flow Magnitude', fontsize=14, fontweight='bold')
    ax3.set_title('FLOW MAGNITUDE: REF vs RIF10 Over Time\n(Shows overall movement intensity)',
                 fontsize=16, fontweight='bold')
    ax3.legend(fontsize=12, loc='best')
    ax3.grid(True, alpha=0.3)

    ax3.axvspan(0, 20, alpha=0.1, color='yellow')

    # Add annotation for higher RIF10 magnitude
    ref_mean_mag = ref_grouped_m['mean'].mean()
    rif_mean_mag = rif_grouped_m['mean'].mean()
    ratio = rif_mean_mag / ref_mean_mag

    ax3.text(0.98, 0.98, f"RIF10/REF ratio: {ratio:.2f}x\n{'⚠ RIF10 HIGHER!' if ratio > 1.2 else 'Normal'}",
            transform=ax3.transAxes, fontsize=11, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='orange' if ratio > 1.2 else 'lightgreen', alpha=0.8))

    # Panel 4: Difference plot (RIF10 - REF) for horizontal with confidence bands
    ax4 = fig.add_subplot(gs[3, 0])

    # Align by frame and calculate difference with uncertainty
    common_frames = sorted(set(ref_grouped.index) & set(rif_grouped.index))
    diff_h = []
    diff_std = []

    for f in common_frames:
        # Get all individual measurements at this frame
        ref_vals = ref_df[ref_df['frame'] == f]['flow_u'].values
        rif_vals = rif_df[rif_df['frame'] == f]['flow_u'].values

        # Calculate difference of means
        mean_diff = rif_vals.mean() - ref_vals.mean()

        # Calculate standard error of difference (accounts for both variabilities)
        # SE_diff = sqrt(SE_ref^2 + SE_rif^2)
        se_ref = ref_vals.std() / np.sqrt(len(ref_vals)) if len(ref_vals) > 1 else 0
        se_rif = rif_vals.std() / np.sqrt(len(rif_vals)) if len(rif_vals) > 1 else 0
        se_diff = np.sqrt(se_ref**2 + se_rif**2)

        diff_h.append(mean_diff)
        diff_std.append(se_diff * 2)  # 2 SE = ~95% confidence

    # Plot with confidence bands
    ax4.plot(common_frames, diff_h, 'o-', linewidth=3, markersize=6, color='purple', alpha=0.8)
    ax4.fill_between(common_frames,
                     np.array(diff_h) - np.array(diff_std),
                     np.array(diff_h) + np.array(diff_std),
                     alpha=0.3, color='purple', label='95% confidence')

    ax4.axhline(y=0, color='black', linestyle='--', linewidth=2)
    ax4.fill_between(common_frames, 0, diff_h, where=[d > 0 for d in diff_h],
                     alpha=0.2, color='red')
    ax4.fill_between(common_frames, 0, diff_h, where=[d < 0 for d in diff_h],
                     alpha=0.2, color='blue')

    ax4.set_xlabel('Time (frame)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Difference (RIF10 - REF)', fontsize=12, fontweight='bold')
    ax4.set_title('Horizontal Direction DIFFERENCE\n(Positive = RIF10 more right)\nWith 95% Confidence Bands',
                 fontsize=13, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # Mark when difference is statistically significant
    sig_frames = [f for f, d, s in zip(common_frames, diff_h, diff_std) if abs(d) > s]
    if sig_frames:
        ax4.text(0.98, 0.98, f"Significant difference\nfirst detected at\nframe {min(sig_frames)}",
                transform=ax4.transAxes, fontsize=10, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    # Panel 5: Statistical summary
    ax5 = fig.add_subplot(gs[3, 1])
    ax5.axis('off')

    # Calculate statistics
    ref_mean_u = ref_df['flow_u'].mean()
    rif_mean_u = rif_df['flow_u'].mean()
    ref_mean_v = ref_df['flow_v'].mean()
    rif_mean_v = rif_df['flow_v'].mean()

    # Early phase (0-20 frames)
    ref_early = ref_df[ref_df['frame'] <= 20]
    rif_early = rif_df[rif_df['frame'] <= 20]

    # Late phase (80+ frames)
    ref_late = ref_df[ref_df['frame'] >= 80]
    rif_late = rif_df[rif_df['frame'] >= 80]

    summary_text = f"""
    STATISTICAL SUMMARY
    {'='*50}

    OVERALL AVERAGES:
    REF Horizontal:  {ref_mean_u:>8.4f} ({'Right' if ref_mean_u > 0 else 'Left'})
    RIF10 Horizontal: {rif_mean_u:>8.4f} ({'Right' if rif_mean_u > 0 else 'Left'})

    REF Vertical:    {ref_mean_v:>8.4f} ({'Down' if ref_mean_v > 0 else 'Up'})
    RIF10 Vertical:   {rif_mean_v:>8.4f} ({'Down' if rif_mean_v > 0 else 'Up'})

    EARLY PHASE (frames 0-20):
    REF Magnitude:   {ref_early['magnitude'].mean():>8.4f}
    RIF10 Magnitude:  {rif_early['magnitude'].mean():>8.4f}
    Ratio: {rif_early['magnitude'].mean()/ref_early['magnitude'].mean():.2f}x

    LATE PHASE (frames 80+):
    REF Magnitude:   {ref_late['magnitude'].mean():>8.4f}
    RIF10 Magnitude:  {rif_late['magnitude'].mean():>8.4f}
    Ratio: {rif_late['magnitude'].mean()/ref_late['magnitude'].mean():.2f}x

    INTERPRETATION:
    {'='*50}
    """

    if rif_mean_mag > ref_mean_mag * 1.3:
        summary_text += "\n⚠ RIF10 shows HIGHER flow than REF!"
        summary_text += "\n  Possible causes:"
        summary_text += "\n  - Cell death/lysis (not growth)"
        summary_text += "\n  - Resistant populations"
        summary_text += "\n  - Measurement artifacts"
    elif abs(rif_mean_u - ref_mean_u) > 0.1:
        summary_text += "\n⚠ Different growth directions!"
        summary_text += "\n  This is unexpected for same species"
    else:
        summary_text += "\n✓ Similar patterns between REF and RIF10"

    ax5.text(0.05, 0.95, summary_text,
            transform=ax5.transAxes,
            fontsize=9,
            verticalalignment='top',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    plt.suptitle('REF (Untreated) vs RIF10 (Rifampicin-Treated) - Directional Comparison',
                fontsize=18, fontweight='bold')

    return fig


def main():
    base_path = Path('/home/junming/nobackup_junming')
    output_path = Path('/home/junming/private/rearch_methodology')

    comparison_dir = output_path / 'ref_vs_rif10_comparison'
    comparison_dir.mkdir(exist_ok=True)

    print("="*70)
    print("REF vs RIF10 COMPARISON ANALYSIS")
    print("="*70)

    # Analyze all REF positions
    print("\nAnalyzing REF positions (101-110)...")
    ref_df = analyze_all_positions('REF', range(101, 111), base_path)

    # Analyze all RIF10 positions
    print("\nAnalyzing RIF10 positions (201-210)...")
    rif_df = analyze_all_positions('RIF10', range(201, 211), base_path)

    if len(ref_df) == 0 or len(rif_df) == 0:
        print("ERROR: Insufficient data for comparison")
        return

    print(f"\nCollected data:")
    print(f"  REF: {len(ref_df)} measurements from {ref_df['position'].nunique()} positions")
    print(f"  RIF10: {len(rif_df)} measurements from {rif_df['position'].nunique()} positions")

    # Create comparison plot
    print("\nGenerating comparison plot...")
    fig = create_comparison_plot(ref_df, rif_df, comparison_dir)

    filename = comparison_dir / 'REF_vs_RIF10_timeseries_comparison.png'
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Save data
    ref_df.to_csv(comparison_dir / 'REF_timeseries_data.csv', index=False)
    rif_df.to_csv(comparison_dir / 'RIF10_timeseries_data.csv', index=False)

    print(f"\n{'='*70}")
    print(f"Results saved to: {comparison_dir}")
    print(f"Files created:")
    print(f"  - REF_vs_RIF10_timeseries_comparison.png")
    print(f"  - REF_timeseries_data.csv")
    print(f"  - RIF10_timeseries_data.csv")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()