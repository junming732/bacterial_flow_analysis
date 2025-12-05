#!/usr/bin/env python3
"""
Enhanced Visualization for Bacterial Growth Heterogeneity
Features unique, intuitive colormaps and innovative spatial representations
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
from pathlib import Path
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')


def create_custom_colormap():
    """
    Create a unique colormap for bacterial growth
    Blue (no growth) -> Cyan -> Yellow -> Orange -> Red (high growth)
    More intuitive than standard hot colormap
    """
    colors = ['#08306b', '#2171b5', '#6baed6', '#c6dbef',  # Blues (low)
              '#ffffcc', '#ffeda0', '#fed976', '#feb24c',  # Yellows (medium)
              '#fd8d3c', '#fc4e2a', '#e31a1c', '#bd0026']  # Reds (high)
    n_bins = 256
    cmap = mcolors.LinearSegmentedColormap.from_list('bacterial_growth', colors, N=n_bins)
    return cmap


def create_diverging_colormap():
    """
    Diverging colormap for showing differences
    Blue (below average) -> White (average) -> Red (above average)
    """
    colors = ['#053061', '#2166ac', '#4393c3', '#92c5de', '#d1e5f0',
              '#f7f7f7',
              '#fddbc7', '#f4a582', '#d6604d', '#b2182b', '#67001f']
    cmap = mcolors.LinearSegmentedColormap.from_list('growth_diverging', colors, N=256)
    return cmap


def detect_spatial_clusters(flow_magnitude, threshold_percentile=90):
    """
    Detect and label distinct growth clusters
    Returns labeled regions and their properties
    """
    threshold = np.percentile(flow_magnitude[flow_magnitude > 0],
                             threshold_percentile) if np.any(flow_magnitude > 0) else 0

    if threshold == 0:
        return None, []

    hotspot_mask = flow_magnitude > threshold
    labeled, num_features = ndimage.label(hotspot_mask)

    if num_features == 0:
        return None, []

    cluster_info = []
    for i in range(1, num_features + 1):
        cluster_mask = labeled == i
        cluster_size = np.sum(cluster_mask)

        if cluster_size > 10:  # Ignore tiny clusters
            coords = np.argwhere(cluster_mask)
            center_y, center_x = coords.mean(axis=0)
            max_flow = flow_magnitude[cluster_mask].max()
            mean_flow = flow_magnitude[cluster_mask].mean()

            cluster_info.append({
                'id': i,
                'center': (center_x, center_y),
                'size': cluster_size,
                'max_flow': max_flow,
                'mean_flow': mean_flow
            })

    return labeled, cluster_info


def analyze_growth_front_directional(flow_magnitude):
    """
    Analyze directional growth (left-right expansion)
    More appropriate for bacterial lawns growing across substrate
    """
    h, w = flow_magnitude.shape

    # Calculate flow profile along horizontal axis (growth direction)
    horizontal_profile = []
    for x in range(w):
        column = flow_magnitude[:, x]
        if np.any(column > 0):
            horizontal_profile.append({
                'position': x / w,  # Normalize 0-1
                'mean_flow': np.mean(column[column > 0]),
                'max_flow': np.max(column),
                'active_pixels': np.sum(column > 0)
            })

    if len(horizontal_profile) == 0:
        return None

    # Find growth front (where bacteria are most active)
    max_flow_pos = max(horizontal_profile, key=lambda x: x['mean_flow'])

    # Classify regions
    front_threshold = max_flow_pos['position'] - 0.2  # 20% before peak

    return {
        'horizontal_profile': horizontal_profile,
        'front_position': max_flow_pos['position'],
        'front_intensity': max_flow_pos['mean_flow']
    }


def detect_scattered_colonies(flow_magnitude, front_position, threshold_percentile=90):
    """
    Detect colonies that have broken away from main growth front
    These could be resistant populations with different growth dynamics
    """
    h, w = flow_magnitude.shape

    # Define the main front region (20% of width around front_position)
    front_start = int((front_position - 0.1) * w)
    front_end = int((front_position + 0.1) * w)

    # Get hotspots
    threshold = np.percentile(flow_magnitude[flow_magnitude > 0],
                             threshold_percentile) if np.any(flow_magnitude > 0) else 0

    if threshold == 0:
        return [], []

    hotspot_mask = flow_magnitude > threshold
    labeled, num_features = ndimage.label(hotspot_mask)

    front_clusters = []
    scattered_clusters = []

    for i in range(1, num_features + 1):
        cluster_mask = labeled == i
        coords = np.argwhere(cluster_mask)

        if len(coords) < 10:
            continue

        center_y, center_x = coords.mean(axis=0)

        cluster_info = {
            'center': (center_x, center_y),
            'size': len(coords),
            'mean_flow': flow_magnitude[cluster_mask].mean()
        }

        # Classify as front or scattered
        if front_start <= center_x <= front_end:
            front_clusters.append(cluster_info)
        else:
            scattered_clusters.append(cluster_info)

    return front_clusters, scattered_clusters


def create_innovative_visualization(flow_magnitude, flow_vectors, title, condition):
    """
    Create a unique 2x2 visualization panel
    """
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.25)

    custom_cmap = create_custom_colormap()
    diverging_cmap = create_diverging_colormap()

    # Panel 1: Main flow magnitude with custom colormap and cluster annotations
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(flow_magnitude, cmap=custom_cmap, interpolation='bilinear')

    labeled, clusters = detect_spatial_clusters(flow_magnitude, threshold_percentile=90)

    if clusters:
        # Annotate clusters
        for cluster in sorted(clusters, key=lambda x: x['mean_flow'], reverse=True)[:5]:
            circle = Circle(cluster['center'], 20,
                          fill=False, edgecolor='white', linewidth=2, linestyle='--')
            ax1.add_patch(circle)
            ax1.text(cluster['center'][0], cluster['center'][1] - 25,
                    f"{cluster['mean_flow']:.2f}",
                    color='white', fontsize=9, ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

    ax1.set_title(f'{title}\nSpatial Flow Distribution (Top 5 Hotspots Labeled)',
                  fontsize=12, fontweight='bold')
    ax1.axis('off')
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046)
    cbar1.set_label('Flow Magnitude', fontsize=10)

    # Panel 2: Horizontal growth profile (directional expansion)
    ax2 = fig.add_subplot(gs[0, 1])

    growth_data = analyze_growth_front_directional(flow_magnitude)

    if growth_data:
        horizontal_profile = growth_data['horizontal_profile']
        positions = [p['position'] for p in horizontal_profile]
        means = [p['mean_flow'] for p in horizontal_profile]
        active = [p['active_pixels'] for p in horizontal_profile]

        # Plot flow profile
        ax2_twin = ax2.twinx()

        line1 = ax2.plot(positions, means, 'o-', linewidth=2.5, markersize=6,
                        color='#2171b5', label='Mean flow')
        ax2.axvline(x=growth_data['front_position'], color='red',
                   linestyle='--', linewidth=2, label='Growth front', alpha=0.7)

        # Plot active pixels
        line2 = ax2_twin.plot(positions, active, 's-', linewidth=2, markersize=4,
                             color='#238b45', alpha=0.6, label='Active pixels')

        ax2.set_xlabel('Horizontal Position\n(0=right/empty, 1=left/colonized)', fontsize=11)
        ax2.set_ylabel('Mean Flow Magnitude', fontsize=11, color='#2171b5')
        ax2_twin.set_ylabel('Active Pixels', fontsize=11, color='#238b45')
        ax2.set_title('Directional Growth Analysis\n(Expansion from Right to Left)',
                      fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')

        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, fontsize=9, loc='upper left')

        # Add interpretation
        front_pos_pct = growth_data['front_position'] * 100
        interpretation = f"Growth front at {front_pos_pct:.0f}% position\n"
        interpretation += f"Front intensity: {growth_data['front_intensity']:.3f}"

        ax2.text(0.98, 0.02, interpretation,
                transform=ax2.transAxes,
                fontsize=10,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        ax2.text(0.5, 0.5, 'No flow detected',
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Directional Growth Analysis', fontsize=12, fontweight='bold')

    # Panel 3: Spatial heterogeneity heatmap (deviation from mean)
    ax3 = fig.add_subplot(gs[1, 0])

    mean_flow = np.mean(flow_magnitude)
    deviation = flow_magnitude - mean_flow

    vmax = max(abs(deviation.min()), abs(deviation.max()))
    im3 = ax3.imshow(deviation, cmap=diverging_cmap,
                     vmin=-vmax, vmax=vmax, interpolation='bilinear')

    ax3.set_title('Spatial Heterogeneity Map\n(Deviation from Mean Growth)',
                  fontsize=12, fontweight='bold')
    ax3.axis('off')
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046)
    cbar3.set_label('Deviation from Mean\n(Blue=Below, Red=Above)', fontsize=9)

    # Panel 4: Vector field showing flow direction (large, visible vectors)
    ax4 = fig.add_subplot(gs[1, 1])

    h, w = flow_magnitude.shape

    # Much sparser sampling for very visible arrows
    subsample = 25
    y_coords = np.arange(subsample//2, h, subsample)
    x_coords = np.arange(subsample//2, w, subsample)

    Y, X = np.meshgrid(y_coords, x_coords, indexing='ij')

    # Get flow vectors at subsampled points
    U = flow_vectors[Y, X, 0]
    V = flow_vectors[Y, X, 1]
    M = flow_magnitude[Y, X]

    # Show background as grayscale for better contrast
    ax4.imshow(flow_magnitude, cmap='gray', alpha=0.4)

    # Plot large, prominent vectors
    quiver = ax4.quiver(X, Y, U, V, M,
                       cmap='hot',
                       scale=20,
                       scale_units='xy',
                       width=0.004,
                       headwidth=4,
                       headlength=5,
                       headaxislength=4.5,
                       alpha=1.0)

    ax4.set_title('Flow Vector Field\n(Arrows show growth direction & magnitude)',
                 fontsize=12, fontweight='bold')
    ax4.axis('off')

    # Add colorbar
    cbar = plt.colorbar(quiver, ax=ax4, fraction=0.046, pad=0.04)
    cbar.set_label('Flow Magnitude', fontsize=9)

    # Calculate and display net direction with better logic
    # Only use significant flow regions
    significant_mask = M > np.percentile(M[M > 0], 30) if np.any(M > 0) else M > 0

    if np.any(significant_mask):
        mean_u = np.mean(U[significant_mask])
        mean_v = np.mean(V[significant_mask])

        if abs(mean_u) > abs(mean_v):
            direction = "→ RIGHTWARD" if mean_u > 0 else "← LEFTWARD"
        else:
            direction = "↓ DOWNWARD" if mean_v > 0 else "↑ UPWARD"
    else:
        direction = "No significant flow"

    ax4.text(0.02, 0.02, f"Net direction:\n{direction}",
            transform=ax4.transAxes,
            fontsize=11,
            verticalalignment='bottom',
            fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9, edgecolor='black', linewidth=2))

    plt.suptitle(f'{condition} - {title}',
                fontsize=14, fontweight='bold', y=0.995)

    return fig


def create_side_by_side_comparison(ref_flow, rif_flow, frame_idx):
    """
    Enhanced side-by-side comparison with radial profiles
    """
    # Resize to same dimensions if needed
    if ref_flow.shape != rif_flow.shape:
        target_shape = (min(ref_flow.shape[0], rif_flow.shape[0]),
                       min(ref_flow.shape[1], rif_flow.shape[1]))
        ref_flow = cv2.resize(ref_flow, (target_shape[1], target_shape[0]))
        rif_flow = cv2.resize(rif_flow, (target_shape[1], target_shape[0]))

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)

    custom_cmap = create_custom_colormap()

    vmax = max(np.max(ref_flow), np.max(rif_flow))

    # REF visualizations
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(ref_flow, cmap=custom_cmap, vmax=vmax, interpolation='bilinear')
    ax1.set_title(f'REF (Untreated)\nFrame {frame_idx}', fontsize=11, fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    ax2 = fig.add_subplot(gs[0, 1])
    ref_growth = analyze_growth_front_directional(ref_flow)
    if ref_growth:
        ref_profile = ref_growth['horizontal_profile']
        ref_positions = [p['position'] for p in ref_profile]
        ref_means = [p['mean_flow'] for p in ref_profile]
        ax2.plot(ref_positions, ref_means, 'o-', linewidth=2.5, color='#2171b5')
        ax2.axvline(x=ref_growth['front_position'], color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax2.set_xlabel('Horizontal Position')
        ax2.set_ylabel('Mean Flow')
        ax2.set_title('REF Growth Profile', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)

    # RIF10 visualizations
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(rif_flow, cmap=custom_cmap, vmax=vmax, interpolation='bilinear')
    ax3.set_title(f'RIF10 (Treated)\nFrame {frame_idx}', fontsize=11, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046)

    ax4 = fig.add_subplot(gs[0, 3])
    rif_growth = analyze_growth_front_directional(rif_flow)
    if rif_growth:
        rif_profile = rif_growth['horizontal_profile']
        rif_positions = [p['position'] for p in rif_profile]
        rif_means = [p['mean_flow'] for p in rif_profile]
        ax4.plot(rif_positions, rif_means, 'o-', linewidth=2.5, color='#e31a1c')
        ax4.axvline(x=rif_growth['front_position'], color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax4.set_xlabel('Horizontal Position')
        ax4.set_ylabel('Mean Flow')
        ax4.set_title('RIF10 Growth Profile', fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3)

    # Comparison plots
    ax5 = fig.add_subplot(gs[1, :2])
    if ref_growth and rif_growth:
        ax5.plot(ref_positions, ref_means, 'o-', linewidth=2.5,
                 color='#2171b5', label='REF (Untreated)', markersize=8)
        ax5.plot(rif_positions, rif_means, 's-', linewidth=2.5,
                 color='#e31a1c', label='RIF10 (Treated)', markersize=8)
        ax5.set_xlabel('Horizontal Position (0=right, 1=left)', fontsize=12)
        ax5.set_ylabel('Mean Flow Magnitude', fontsize=12)
        ax5.set_title('Direct Comparison: Growth Front Profiles', fontsize=12, fontweight='bold')
        ax5.legend(fontsize=11)
        ax5.grid(True, alpha=0.3)

    # Difference map
    ax6 = fig.add_subplot(gs[1, 2:])
    difference = rif_flow - ref_flow
    vmax_diff = max(abs(difference.min()), abs(difference.max()))

    diverging_cmap = create_diverging_colormap()
    im6 = ax6.imshow(difference, cmap=diverging_cmap,
                     vmin=-vmax_diff, vmax=vmax_diff, interpolation='bilinear')
    ax6.set_title('Difference Map (RIF10 - REF)\nRed=Higher in RIF10, Blue=Higher in REF',
                  fontsize=12, fontweight='bold')
    ax6.axis('off')
    cbar6 = plt.colorbar(im6, ax=ax6, fraction=0.046)
    cbar6.set_label('Flow Difference', fontsize=10)

    plt.suptitle(f'Comprehensive Comparison: REF vs RIF10 - Frame {frame_idx}',
                fontsize=14, fontweight='bold')

    return fig


def main():
    base_path = Path('/home/junming/nobackup_junming')
    output_path = Path('/home/junming/private/rearch_methodology')

    innovative_dir = output_path / 'innovative_visualizations'
    innovative_dir.mkdir(exist_ok=True)

    print("Generating innovative visualizations...")

    # Process example positions
    positions_to_viz = {
        'REF': [101, 102],
        'RIF10': [201, 202]
    }

    folders = {
        'REF': 'REF_raw_data101_110',
        'RIF10': 'RIF10_raw_data201_210'
    }

    flow_storage = {}

    # Generate innovative single visualizations
    for condition in ['REF', 'RIF10']:
        flow_storage[condition] = {}
        for pos in positions_to_viz[condition]:
            data_path = base_path / folders[condition] / f'Pos{pos}' / 'aphase'

            if not data_path.exists():
                continue

            img_files = sorted(data_path.glob('img_*.tiff'))

            # Generate visualizations for early (0-20), mid (40-60), and late (80-100) timepoints
            key_frames = [0, 5, 10, 15, 20, 40, 60, 80, 100]

            for frame_idx in key_frames:
                if frame_idx < len(img_files) - 1:
                    img1 = cv2.imread(str(img_files[frame_idx]), cv2.IMREAD_GRAYSCALE)
                    img2 = cv2.imread(str(img_files[frame_idx + 1]), cv2.IMREAD_GRAYSCALE)

                    if img1 is not None and img2 is not None:
                        flow = cv2.calcOpticalFlowFarneback(
                            img1, img2, None,
                            pyr_scale=0.5, levels=3, winsize=15,
                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                        )
                        flow_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

                        if frame_idx not in flow_storage[condition]:
                            flow_storage[condition][frame_idx] = {}
                        flow_storage[condition][frame_idx][pos] = (flow_mag, flow)

                        fig = create_innovative_visualization(
                            flow_mag,
                            flow,
                            f'Pos{pos} Frame {frame_idx}',
                            condition
                        )
                        fig.savefig(
                            innovative_dir / f'innovative_{condition}_Pos{pos}_frame{frame_idx:03d}.png',
                            dpi=150, bbox_inches='tight'
                        )
                        plt.close(fig)
                        print(f"  Created innovative viz for {condition} Pos{pos} Frame {frame_idx}")

    # Generate comparison visualizations
    print("\nGenerating comparison visualizations...")
    comparison_frames = [0, 10, 20, 40, 60, 80, 100]
    for frame_idx in comparison_frames:
        if frame_idx in flow_storage['REF'] and frame_idx in flow_storage['RIF10']:
            ref_pos = list(flow_storage['REF'][frame_idx].keys())[0]
            rif_pos = list(flow_storage['RIF10'][frame_idx].keys())[0]

            ref_flow_mag, ref_flow = flow_storage['REF'][frame_idx][ref_pos]
            rif_flow_mag, rif_flow = flow_storage['RIF10'][frame_idx][rif_pos]

            fig = create_side_by_side_comparison(ref_flow_mag, rif_flow_mag, frame_idx)
            fig.savefig(
                innovative_dir / f'comparison_innovative_frame{frame_idx:03d}.png',
                dpi=150, bbox_inches='tight'
            )
            plt.close(fig)
            print(f"  Created comparison for Frame {frame_idx}")

    print(f"\nAll visualizations saved to {innovative_dir}")
    print("\n" + "="*70)
    print("DIAGNOSTIC: Investigating RIF10 > REF Paradox")
    print("="*70)
    print("\nFrom your previous analysis, RIF10 showed 65% MORE flow than REF.")
    print("This is unexpected because rifampicin should SUPPRESS growth.\n")
    print("Possible explanations:")
    print("1. Optical flow detects cell death/lysis (dying cells move more)")
    print("2. RIF10 contains resistant populations growing faster")
    print("3. Dataset labels might be swapped")
    print("4. Optical flow detects medium convection, not bacterial growth\n")
    print("Check the heatmaps:")
    print("- Do RIF10 images show more scattered hotspots?")
    print("- Is RIF10 front more irregular/patchy?")
    print("- Compare early frames (0-20) vs late frames (80-100)")
    print("="*70)
    print("\nFeatures:")
    print("  - Custom colormap (blue->cyan->yellow->red)")
    print("  - Directional growth profiles (left-right expansion)")
    print("  - Annotated top hotspots with flow values")
    print("  - Cluster size vs intensity analysis")
    print("  - Diverging colormap for spatial heterogeneity")
    print("  - Direct difference maps (RIF10 - REF)")
    print(f"\nGenerated {len(list(innovative_dir.glob('*.png')))} visualization files")


if __name__ == '__main__':
    main()