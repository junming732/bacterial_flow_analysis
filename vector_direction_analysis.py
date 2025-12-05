#!/usr/bin/env python3
"""
Optical Flow Vector Analysis for Bacterial Growth
Analyzes directional components to distinguish:
- True directional growth (coordinated vectors)
- Cell death/artifacts (random/chaotic vectors)
- Resistant populations (different flow directions)
"""

import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def analyze_flow_direction(flow, flow_magnitude, threshold_percentile=50):
    """
    Analyze directional properties of optical flow

    Returns:
        - Dominant direction (angle in degrees)
        - Directional coherence (how aligned vectors are)
        - Left vs right bias
        - Up vs down bias
    """
    # Only analyze significant flow (above threshold)
    threshold = np.percentile(flow_magnitude[flow_magnitude > 0],
                             threshold_percentile) if np.any(flow_magnitude > 0) else 0

    mask = flow_magnitude > threshold

    if not np.any(mask):
        return None

    # Extract flow components
    flow_x = flow[..., 0][mask]  # Horizontal component
    flow_y = flow[..., 1][mask]  # Vertical component

    # Calculate mean direction
    mean_x = np.mean(flow_x)
    mean_y = np.mean(flow_y)

    # Dominant angle (in degrees, 0=right, 90=down, 180=left, 270=up)
    dominant_angle = np.arctan2(mean_y, mean_x) * 180 / np.pi

    # Directional coherence: how aligned are the vectors?
    # Compute angle for each vector
    angles = np.arctan2(flow_y, flow_x)

    # Convert to unit vectors and compute mean resultant length
    mean_cos = np.mean(np.cos(angles))
    mean_sin = np.mean(np.sin(angles))
    coherence = np.sqrt(mean_cos**2 + mean_sin**2)  # 0=random, 1=perfectly aligned

    # Left-right bias (negative=left, positive=right)
    left_right_bias = mean_x

    # Up-down bias (negative=up, positive=down)
    up_down_bias = mean_y

    # Calculate proportion moving in dominant direction
    vectors_aligned = np.sum(np.cos(angles - np.arctan2(mean_y, mean_x)) > 0.5) / len(angles)

    return {
        'dominant_angle': dominant_angle,
        'coherence': coherence,
        'left_right_bias': left_right_bias,
        'up_down_bias': up_down_bias,
        'mean_magnitude': np.mean(flow_magnitude[mask]),
        'vectors_aligned': vectors_aligned
    }


def create_vector_field_plot(flow, flow_magnitude, title, subsample=20):
    """
    Create quiver plot showing flow vectors
    """
    h, w = flow.shape[:2]

    # Subsample for visualization
    y_coords = np.arange(0, h, subsample)
    x_coords = np.arange(0, w, subsample)

    Y, X = np.meshgrid(y_coords, x_coords, indexing='ij')

    # Get flow at subsampled points
    U = flow[Y, X, 0]
    V = flow[Y, X, 1]
    M = flow_magnitude[Y, X]

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: Vector field overlay on magnitude
    ax1 = axes[0]
    im1 = ax1.imshow(flow_magnitude, cmap='gray', alpha=0.5)

    # Color vectors by magnitude
    quiver = ax1.quiver(X, Y, U, V, M,
                       cmap='hot',
                       scale=50,
                       width=0.003,
                       headwidth=3,
                       headlength=4,
                       alpha=0.8)

    ax1.set_title(f'{title}\nVector Field (colored by magnitude)', fontsize=12, fontweight='bold')
    ax1.axis('off')
    plt.colorbar(quiver, ax=ax1, label='Flow Magnitude', fraction=0.046)

    # Panel 2: Horizontal component (left-right)
    ax2 = axes[1]
    flow_x = flow[..., 0]
    vmax = max(abs(flow_x.min()), abs(flow_x.max()))

    im2 = ax2.imshow(flow_x, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax2.set_title('Horizontal Flow Component\n(Red=Right, Blue=Left)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046)
    cbar2.set_label('Horizontal Velocity')

    # Panel 3: Vertical component (up-down)
    ax3 = axes[2]
    flow_y = flow[..., 1]
    vmax_y = max(abs(flow_y.min()), abs(flow_y.max()))

    im3 = ax3.imshow(flow_y, cmap='RdBu_r', vmin=-vmax_y, vmax=vmax_y)
    ax3.set_title('Vertical Flow Component\n(Red=Down, Blue=Up)', fontsize=12, fontweight='bold')
    ax3.axis('off')
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046)
    cbar3.set_label('Vertical Velocity')

    plt.tight_layout()
    return fig


def create_directional_analysis_plot(direction_data, title):
    """
    Create comprehensive directional analysis visualization
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Panel 1: Polar plot of flow direction distribution
    ax1 = fig.add_subplot(gs[0, 0], projection='polar')

    # Histogram of angles
    angles = []
    magnitudes = []
    for data in direction_data:
        if data is not None:
            angles.append(data['dominant_angle'] * np.pi / 180)
            magnitudes.append(data['mean_magnitude'])

    if angles:
        # Plot as points
        ax1.scatter(angles, magnitudes, s=100, alpha=0.6, c=magnitudes, cmap='hot')

        # Mean direction
        mean_angle = np.arctan2(np.mean([np.sin(a) for a in angles]),
                               np.mean([np.cos(a) for a in angles]))
        mean_mag = np.mean(magnitudes)
        ax1.arrow(0, 0, mean_angle, mean_mag,
                 width=0.1, head_width=0.3, head_length=mean_mag*0.1,
                 fc='red', ec='red', linewidth=3, alpha=0.8)

        ax1.set_theta_zero_location('E')
        ax1.set_theta_direction(1)
        ax1.set_title('Flow Direction Distribution\n(Red arrow = mean direction)',
                     fontsize=11, fontweight='bold', pad=20)

    # Panel 2: Coherence over time
    ax2 = fig.add_subplot(gs[0, 1])

    times = list(range(len(direction_data)))
    coherences = [d['coherence'] if d else 0 for d in direction_data]

    ax2.plot(times, coherences, 'o-', linewidth=2.5, markersize=8, color='#2171b5')
    ax2.axhline(y=0.7, color='red', linestyle='--', linewidth=2, alpha=0.7, label='High coherence')
    ax2.axhline(y=0.3, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Low coherence')

    ax2.set_xlabel('Time (frame)', fontsize=11)
    ax2.set_ylabel('Directional Coherence', fontsize=11)
    ax2.set_title('Flow Coherence Over Time\n(1=aligned, 0=random)', fontsize=11, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    # Panel 3: Left-right bias over time
    ax3 = fig.add_subplot(gs[0, 2])

    lr_bias = [d['left_right_bias'] if d else 0 for d in direction_data]

    ax3.plot(times, lr_bias, 'o-', linewidth=2.5, markersize=8, color='#238b45')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.fill_between(times, 0, lr_bias, where=[b > 0 for b in lr_bias],
                     alpha=0.3, color='red', label='Rightward')
    ax3.fill_between(times, 0, lr_bias, where=[b < 0 for b in lr_bias],
                     alpha=0.3, color='blue', label='Leftward')

    ax3.set_xlabel('Time (frame)', fontsize=11)
    ax3.set_ylabel('Left-Right Bias', fontsize=11)
    ax3.set_title('Horizontal Growth Direction\n(+Right, -Left)', fontsize=11, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel 4: Summary statistics table
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis('off')

    if direction_data and any(d is not None for d in direction_data):
        # Calculate summary statistics
        valid_data = [d for d in direction_data if d is not None]

        avg_coherence = np.mean([d['coherence'] for d in valid_data])
        avg_lr_bias = np.mean([d['left_right_bias'] for d in valid_data])
        avg_ud_bias = np.mean([d['up_down_bias'] for d in valid_data])
        avg_magnitude = np.mean([d['mean_magnitude'] for d in valid_data])

        # Determine dominant direction
        if abs(avg_lr_bias) > abs(avg_ud_bias):
            if avg_lr_bias > 0:
                direction_text = "RIGHTWARD →"
            else:
                direction_text = "LEFTWARD ←"
        else:
            if avg_ud_bias > 0:
                direction_text = "DOWNWARD ↓"
            else:
                direction_text = "UPWARD ↑"

        # Interpretation
        if avg_coherence > 0.7:
            coherence_text = "HIGH - Coordinated expansion (biological growth)"
        elif avg_coherence > 0.4:
            coherence_text = "MEDIUM - Mixed pattern"
        else:
            coherence_text = "LOW - Random/chaotic (artifacts or cell death)"

        summary_text = f"""
        DIRECTIONAL ANALYSIS SUMMARY
        {'='*60}

        Dominant Direction:        {direction_text}
        Average Coherence:         {avg_coherence:.3f} - {coherence_text}

        Left-Right Bias:           {avg_lr_bias:.4f} ({'Right' if avg_lr_bias > 0 else 'Left'})
        Up-Down Bias:              {avg_ud_bias:.4f} ({'Down' if avg_ud_bias > 0 else 'Up'})
        Average Flow Magnitude:    {avg_magnitude:.4f}

        INTERPRETATION:
        {'-'*60}
        """

        if avg_coherence > 0.6:
            summary_text += "\n✓ Flow is highly directional - consistent with biological growth"
            if abs(avg_lr_bias) > abs(avg_ud_bias):
                summary_text += f"\n✓ Clear horizontal expansion ({'rightward' if avg_lr_bias > 0 else 'leftward'})"
        else:
            summary_text += "\n⚠ Flow is not directional - may be artifacts, cell death, or convection"
            summary_text += "\n⚠ This pattern is NOT consistent with coordinated bacterial growth"

        ax4.text(0.1, 0.9, summary_text,
                transform=ax4.transAxes,
                fontsize=11,
                verticalalignment='top',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle(f'{title} - Directional Flow Analysis',
                fontsize=14, fontweight='bold')

    return fig


def main():
    base_path = Path('/home/junming/nobackup_junming')
    output_path = Path('/home/junming/private/rearch_methodology')

    vector_dir = output_path / 'vector_analysis'
    vector_dir.mkdir(exist_ok=True)

    print("Analyzing optical flow vectors for directional information...")
    print("="*70)

    conditions = {
        'REF': ('REF_raw_data101_110', [101, 102]),
        'RIF10': ('RIF10_raw_data201_210', [201, 202])
    }

    all_results = []

    for condition, (folder, positions) in conditions.items():
        print(f"\n{condition}:")

        for pos in positions:
            data_path = base_path / folder / f'Pos{pos}' / 'aphase'

            if not data_path.exists():
                print(f"  Skipping Pos{pos} - not found")
                continue

            img_files = sorted(data_path.glob('img_*.tiff'))

            if len(img_files) < 2:
                continue

            print(f"  Processing Pos{pos} ({len(img_files)} frames)...")

            direction_data = []

            # Analyze every 5th frame to get temporal evolution
            sample_frames = list(range(0, min(len(img_files)-1, 100), 5))

            for frame_idx in sample_frames:
                img1 = cv2.imread(str(img_files[frame_idx]), cv2.IMREAD_GRAYSCALE)
                img2 = cv2.imread(str(img_files[frame_idx + 1]), cv2.IMREAD_GRAYSCALE)

                if img1 is None or img2 is None:
                    direction_data.append(None)
                    continue

                # Compute optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    img1, img2, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )

                flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

                # Analyze direction
                dir_info = analyze_flow_direction(flow, flow_magnitude, threshold_percentile=50)
                direction_data.append(dir_info)

                if dir_info:
                    all_results.append({
                        'condition': condition,
                        'position': pos,
                        'frame': frame_idx,
                        **dir_info
                    })

                # Save vector field for key frames
                if frame_idx in [0, 20, 40, 60]:
                    fig = create_vector_field_plot(flow, flow_magnitude,
                                                   f'{condition} Pos{pos} Frame {frame_idx}')
                    fig.savefig(vector_dir / f'vectors_{condition}_Pos{pos}_frame{frame_idx:03d}.png',
                               dpi=150, bbox_inches='tight')
                    plt.close(fig)

            # Create temporal analysis plot
            if any(d is not None for d in direction_data):
                fig = create_directional_analysis_plot(direction_data,
                                                      f'{condition} Pos{pos}')
                fig.savefig(vector_dir / f'directional_analysis_{condition}_Pos{pos}.png',
                           dpi=150, bbox_inches='tight')
                plt.close(fig)

                print(f"    ✓ Generated vector analysis for Pos{pos}")

    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_path / 'vector_direction_analysis.csv', index=False)

        # Comparative analysis
        print("\n" + "="*70)
        print("COMPARATIVE DIRECTIONAL ANALYSIS: REF vs RIF10")
        print("="*70)

        ref_data = results_df[results_df['condition'] == 'REF']
        rif_data = results_df[results_df['condition'] == 'RIF10']

        print(f"\nREF (Untreated):")
        print(f"  Average coherence:     {ref_data['coherence'].mean():.3f}")
        print(f"  Average L-R bias:      {ref_data['left_right_bias'].mean():.4f}")
        print(f"  Average magnitude:     {ref_data['mean_magnitude'].mean():.4f}")

        print(f"\nRIF10 (Treated):")
        print(f"  Average coherence:     {rif_data['coherence'].mean():.3f}")
        print(f"  Average L-R bias:      {rif_data['left_right_bias'].mean():.4f}")
        print(f"  Average magnitude:     {rif_data['mean_magnitude'].mean():.4f}")

        print(f"\nKey Findings:")
        if abs(rif_data['coherence'].mean() - ref_data['coherence'].mean()) > 0.2:
            print(f"  ⚠ Large difference in coherence - different growth patterns!")

        if np.sign(ref_data['left_right_bias'].mean()) != np.sign(rif_data['left_right_bias'].mean()):
            print(f"  ⚠ OPPOSITE DIRECTIONS - REF and RIF10 grow in different directions!")
            print(f"     This is highly suspicious - check dataset labels!")

        if rif_data['mean_magnitude'].mean() > ref_data['mean_magnitude'].mean():
            print(f"  ⚠ RIF10 has MORE flow than REF")
            if rif_data['coherence'].mean() < 0.5:
                print(f"     Low coherence suggests this is NOT biological growth")
                print(f"     Likely: cell death, lysis, or measurement artifacts")

        print("\n" + "="*70)
        print(f"Results saved to {output_path}")
        print(f"Vector visualizations in {vector_dir}")
        print("="*70)


if __name__ == '__main__':
    main()