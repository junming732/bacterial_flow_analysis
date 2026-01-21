#!/usr/bin/env python3
"""
REF vs RIF10 Comparison - Time Series Analysis
V4: Final Layout Fix - Centered Legend and Aggregated Difference Shading.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def compute_weighted_direction(flow, flow_magnitude):
    """Magnitude-weighted direction with Double-Threshold Outlier Rejection."""
    valid_mags = flow_magnitude[flow_magnitude > 0.001]
    if valid_mags.size < 10: return 0, 0, 0

    lower_bound = np.percentile(valid_mags, 50)
    upper_bound = np.percentile(valid_mags, 95)
    upper_bound = min(upper_bound, 10.0)

    mask = (flow_magnitude >= lower_bound) & (flow_magnitude <= upper_bound)
    if not np.any(mask): return 0, 0, 0

    flow_x = flow[..., 0][mask]
    flow_y = flow[..., 1][mask]
    mag = flow_magnitude[mask]

    return np.sum(flow_x * mag) / np.sum(mag), np.sum(flow_y * mag) / np.sum(mag), np.mean(mag)

def find_images_in_pos(pos_path):
    aphase = pos_path / 'aphase'
    if aphase.exists():
        imgs = sorted(list(aphase.glob('*.tiff')) + list(aphase.glob('*.tif')))
        if len(imgs) > 5: return imgs
    return sorted(list(pos_path.glob('*.tiff')) + list(pos_path.glob('*.tif')))

def analyze_all_positions(condition, base_path):
    folder = 'REF_raw_data101_110' if condition == 'REF' else 'RIF10_raw_data201_210'
    root = base_path / 'data' / folder
    if not root.exists(): return pd.DataFrame()

    all_data = []
    for pos_path in sorted([p for p in root.glob('Pos*') if p.is_dir()]):
        imgs = find_images_in_pos(pos_path)
        if len(imgs) < 2: continue
        print(f"  Analyzing {condition} {pos_path.name}...")
        for idx in range(0, min(len(imgs)-1, 110), 2):
            i1, i2 = cv2.imread(str(imgs[idx]), 0), cv2.imread(str(imgs[idx+1]), 0)
            if i1 is None or i2 is None: continue
            flow = cv2.calcOpticalFlowFarneback(i1, i2, None, 0.5, 3, 31, 3, 5, 1.2, 0)
            u, v, m = compute_weighted_direction(flow, np.sqrt(flow[...,0]**2 + flow[...,1]**2))
            all_data.append({'condition': condition, 'frame': idx, 'flow_u': u, 'flow_v': v, 'magnitude': m})
    return pd.DataFrame(all_data)

def create_comparison_plot(ref_df, rif_df, output_file):
    fig = plt.figure(figsize=(16, 18))
    # Increased top margin to prevent title overlap
    plt.subplots_adjust(top=0.92, bottom=0.08, hspace=0.4)
    gs = fig.add_gridspec(4, 2)

    # Aggregate Plotter
    def plot_agg(ax, df, col, color, lbl, marker):
        stats = df.groupby('frame')[col].agg(['mean', 'std'])
        ln, = ax.plot(stats.index, stats['mean'], f'{marker}-', lw=2.5, color=color, label=lbl, alpha=0.8)
        ax.fill_between(stats.index, stats['mean']-stats['std'], stats['mean']+stats['std'], alpha=0.15, color=color)
        return ln

    # 1. Horizontal
    ax1 = fig.add_subplot(gs[0, :])
    l1 = plot_agg(ax1, ref_df, 'flow_u', '#2171b5', 'REF (Untreated)', 'o')
    l2 = plot_agg(ax1, rif_df, 'flow_u', '#e31a1c', 'RIF10 (Treated)', 's')
    ax1.axhline(0, color='black', lw=1); ax1.grid(True, alpha=0.2)
    ax1.set_title('HORIZONTAL VITALITY SHIFT (Aggregate of 10 Positions)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Flow U (-Left / +Right)', fontweight='bold')

    # 2. Vertical
    ax2 = fig.add_subplot(gs[1, :])
    plot_agg(ax2, ref_df, 'flow_v', '#2171b5', 'REF', 'o')
    plot_agg(ax2, rif_df, 'flow_v', '#e31a1c', 'RIF10', 's')
    ax2.axhline(0, color='black', lw=1); ax2.grid(True, alpha=0.2)
    ax2.set_title('VERTICAL VALIDATION (Zero Flow = No Stage Drift)', fontsize=13)

    # 3. Magnitude
    ax3 = fig.add_subplot(gs[2, :])
    plot_agg(ax3, ref_df, 'magnitude', '#2171b5', 'REF', 'o')
    plot_agg(ax3, rif_df, 'magnitude', '#e31a1c', 'RIF10', 's')
    ax3.set_title('FLOW MAGNITUDE (Movement Intensity)', fontsize=13); ax3.grid(True, alpha=0.2)

    # 4. Aggregate Difference
    ax4 = fig.add_subplot(gs[3, 0])
    common = sorted(set(ref_df['frame']) & set(rif_df['frame']))
    d_m, d_s = [], []
    for f in common:
        r_u, t_u = ref_df[ref_df['frame']==f]['flow_u'], rif_df[rif_df['frame']==f]['flow_u']
        d_m.append(t_u.mean() - r_u.mean())
        d_s.append(np.sqrt(r_u.std()**2 + t_u.std()**2))
    ax4.plot(common, d_m, color='purple', lw=2.5, label='$\Delta$ (Treated - Ref)')
    ax4.fill_between(common, np.array(d_m)-np.array(d_s), np.array(d_m)+np.array(d_s), color='purple', alpha=0.2)
    ax4.axhline(0, color='black', ls='--'); ax4.grid(True, alpha=0.2)
    ax4.set_title('AGGREGATE DIFFERENCE IN FLOW', fontsize=12, fontweight='bold')

    # 5. Stats
    ax5 = fig.add_subplot(gs[3, 1]); ax5.axis('off')
    summary = (f"STATISTICAL SUMMARY\n{'='*25}\nAlgorithm: Farneback\nWindow: 31px (Stabilized)\n"
               f"Outlier Rejection: 50-95%\n\nNet Shift: {'RIGHTWARD' if rif_df['flow_u'].mean() > ref_df['flow_u'].mean() else 'NONE'}")
    ax5.text(0.1, 0.8, summary, family='monospace', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # GLOBAL LEGEND AT THE BOTTOM
    fig.legend(handles=[l1, l2], loc='lower center', ncol=2, fontsize=12, frameon=True, shadow=True)

    plt.suptitle('REF vs RIF10 - Phenotypic Susceptibility via Optical Flow', fontsize=18, fontweight='bold')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Aggregated Comparison plot saved to {output_file}")

def main():
    base = Path('.')
    ref, rif = analyze_all_positions('REF', base), analyze_all_positions('RIF10', base)
    if not ref.empty and not rif.empty:
        create_comparison_plot(ref, rif, 'results_comparison_final.png')

if __name__ == '__main__': main()