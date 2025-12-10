#!/usr/bin/env python3
"""
Spatial Heterogeneity Analysis for Bacterial Growth
Detects local growth hotspots and resistant populations that area metrics miss

Key metrics:
1. Coefficient of Variation (CV) - spatial heterogeneity
2. Moran's I - spatial autocorrelation (clustering)
3. Hotspot detection - local regions exceeding thresholds
4. Growth front analysis - expansion patterns
"""

import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import ndimage, stats
from skimage import measure
import warnings
warnings.filterwarnings('ignore')

class SpatialHeterogeneityAnalyzer:
    """
    Analyzes spatial patterns in bacterial growth to detect:
    - Resistant subpopulations
    - Non-uniform drug effects
    - Local growth hotspots
    """

    def __init__(self, hotspot_percentile=90):
        """
        Args:
            hotspot_percentile: Flow values above this percentile are "hotspots"
        """
        self.hotspot_percentile = hotspot_percentile

    def calculate_coefficient_of_variation(self, flow_magnitude):
        """
        CV = std/mean - measures spatial heterogeneity
        Higher CV = more heterogeneous growth (bad for antibiotics)
        """
        nonzero = flow_magnitude[flow_magnitude > 0]
        if len(nonzero) == 0:
            return 0.0
        return np.std(nonzero) / (np.mean(nonzero) + 1e-10)

    def calculate_morans_i(self, flow_magnitude, distance_threshold=50):
        """
        Moran's I - spatial autocorrelation
        I > 0: Clustered (hotspots grouped together - resistant colonies)
        I ≈ 0: Random
        I < 0: Dispersed
        """
        # Sample points for computational efficiency
        h, w = flow_magnitude.shape
        y_coords, x_coords = np.meshgrid(
            np.arange(0, h, 10),
            np.arange(0, w, 10),
            indexing='ij'
        )

        coords = np.column_stack([y_coords.ravel(), x_coords.ravel()])
        values = flow_magnitude[y_coords.ravel(), x_coords.ravel()]

        # Remove zeros
        mask = values > 0
        coords = coords[mask]
        values = values[mask]

        if len(values) < 10:
            return 0.0

        # Calculate spatial weights (inverse distance)
        n = len(values)
        mean_val = np.mean(values)

        # For efficiency, only consider neighbors within distance_threshold
        numerator = 0
        denominator = 0
        weight_sum = 0

        for i in range(min(n, 200)):  # Limit computation
            distances = np.sqrt(np.sum((coords - coords[i])**2, axis=1))
            weights = (distances < distance_threshold) & (distances > 0)

            if np.any(weights):
                w_sum = np.sum(weights)
                numerator += w_sum * (values[i] - mean_val) * np.sum(weights * (values - mean_val))
                weight_sum += w_sum

        denominator = np.sum((values - mean_val)**2)

        if denominator == 0 or weight_sum == 0:
            return 0.0

        morans_i = (n / weight_sum) * (numerator / denominator)
        return morans_i

    def detect_hotspots(self, flow_magnitude):
        """
        Detect local growth hotspots - potential resistant populations
        Returns:
            - Number of hotspot regions
            - Total hotspot area (fraction of image)
            - Max hotspot size
        """
        threshold = np.percentile(flow_magnitude[flow_magnitude > 0],
                                 self.hotspot_percentile) if np.any(flow_magnitude > 0) else 0

        if threshold == 0:
            return 0, 0.0, 0

        hotspot_mask = flow_magnitude > threshold

        # Label connected components
        labeled, num_features = ndimage.label(hotspot_mask)

        if num_features == 0:
            return 0, 0.0, 0

        # Calculate properties
        total_pixels = np.prod(flow_magnitude.shape)
        hotspot_area = np.sum(hotspot_mask) / total_pixels

        sizes = ndimage.sum(hotspot_mask, labeled, range(1, num_features + 1))
        max_hotspot = np.max(sizes) / total_pixels if len(sizes) > 0 else 0

        return num_features, hotspot_area, max_hotspot

    def calculate_spatial_entropy(self, flow_magnitude, n_bins=20):
        """
        Shannon entropy of spatial flow distribution
        Higher entropy = more uniform (good for antibiotics)
        Lower entropy = concentrated growth (resistant colonies)
        """
        # Divide image into grid
        h, w = flow_magnitude.shape
        cell_h = h // n_bins
        cell_w = w // n_bins

        cells = []
        for i in range(n_bins):
            for j in range(n_bins):
                cell = flow_magnitude[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                cells.append(np.mean(cell))

        cells = np.array(cells)
        cells = cells[cells > 0]

        if len(cells) == 0:
            return 0.0

        # Normalize to probability distribution
        probs = cells / np.sum(cells)

        # Calculate entropy
        entropy = -np.sum(probs * np.log2(probs + 1e-10))

        return entropy

    def analyze_growth_front(self, flow_magnitude):
        """
        Analyze if growth is at edges (normal expansion) or interior (resistant colonies)
        """
        # Create edge mask (outer 20% of image)
        h, w = flow_magnitude.shape
        edge_mask = np.zeros((h, w), dtype=bool)
        border = int(min(h, w) * 0.2)
        edge_mask[:border, :] = True
        edge_mask[-border:, :] = True
        edge_mask[:, :border] = True
        edge_mask[:, -border:] = True

        edge_flow = np.mean(flow_magnitude[edge_mask])
        interior_flow = np.mean(flow_magnitude[~edge_mask])

        # Ratio > 1: growth at edges (normal)
        # Ratio < 1: growth in interior (resistant colonies forming)
        edge_to_interior_ratio = edge_flow / (interior_flow + 1e-10)

        return edge_to_interior_ratio

    def analyze_frame(self, flow_magnitude):
        """
        Complete spatial analysis of one frame
        """
        results = {
            'cv': self.calculate_coefficient_of_variation(flow_magnitude),
            'morans_i': self.calculate_morans_i(flow_magnitude),
            'spatial_entropy': self.calculate_spatial_entropy(flow_magnitude),
            'edge_interior_ratio': self.analyze_growth_front(flow_magnitude)
        }

        n_hotspots, hotspot_area, max_hotspot = self.detect_hotspots(flow_magnitude)
        results['n_hotspots'] = n_hotspots
        results['hotspot_area'] = hotspot_area
        results['max_hotspot_size'] = max_hotspot

        return results


def compute_optical_flow_direct(img1, img2):
    """Compute dense optical flow between two frames"""
    flow = cv2.calcOpticalFlowFarneback(
        img1, img2,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    return magnitude, flow


def create_spatial_heatmap(flow_magnitude, title, vmax=None):
    """Create detailed spatial heatmap with hotspot overlay"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Flow magnitude heatmap
    im1 = axes[0].imshow(flow_magnitude, cmap='hot', vmax=vmax)
    axes[0].set_title(f'{title}\nFlow Magnitude')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)

    # 2. Hotspot detection (>90th percentile)
    threshold = np.percentile(flow_magnitude[flow_magnitude > 0], 90) if np.any(flow_magnitude > 0) else 0
    hotspot_mask = flow_magnitude > threshold

    axes[1].imshow(flow_magnitude, cmap='gray', alpha=0.3)
    axes[1].imshow(hotspot_mask, cmap='Reds', alpha=0.7)
    axes[1].set_title(f'Hotspots (>90th percentile)\nThreshold: {threshold:.3f}')
    axes[1].axis('off')

    # 3. Edge vs Interior analysis
    h, w = flow_magnitude.shape
    border = int(min(h, w) * 0.2)
    edge_mask = np.zeros((h, w), dtype=bool)
    edge_mask[:border, :] = True
    edge_mask[-border:, :] = True
    edge_mask[:, :border] = True
    edge_mask[:, -border:] = True

    viz = np.zeros((h, w, 3))
    viz[edge_mask] = [0, 1, 0]  # Green for edges
    viz[~edge_mask] = [0, 0, 1]  # Blue for interior

    axes[2].imshow(viz, alpha=0.3)
    axes[2].imshow(flow_magnitude, cmap='hot', alpha=0.7)
    axes[2].set_title('Growth Location\nGreen=Edge, Blue=Interior')
    axes[2].axis('off')

    plt.tight_layout()
    return fig


def analyze_time_series(data_path, condition, position, analyzer):
    """Analyze spatial heterogeneity across entire time series"""
    img_files = sorted(data_path.glob(f'img_*.tiff'))

    if len(img_files) < 2:
        return None, None

    results = []
    heatmaps = []

    for i in range(len(img_files) - 1):
        img1 = cv2.imread(str(img_files[i]), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(img_files[i + 1]), cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            continue

        # Compute optical flow
        flow_mag, flow = compute_optical_flow_direct(img1, img2)

        # Spatial analysis
        spatial_metrics = analyzer.analyze_frame(flow_mag)

        results.append({
            'condition': condition,
            'position': position,
            'time': i,
            'mean_flow': np.mean(flow_mag),
            **spatial_metrics
        })

        # Save heatmaps for select frames
        if i in [0, 5, 10, 20, 40]:
            heatmaps.append((i, flow_mag))

    return pd.DataFrame(results), heatmaps


def main():
    # Setup paths
    base_path = Path('/home/junming/nobackup_junming')
    output_path = Path('/home/junming/private/rearch_methodology')
    output_path.mkdir(exist_ok=True)

    # Initialize analyzer
    analyzer = SpatialHeterogeneityAnalyzer(hotspot_percentile=90)

    # Process all conditions
    conditions = {
        'REF': ('REF_raw_data101_110', range(101, 111)),
        'RIF10': ('RIF10_raw_data201_210', range(201, 211))
    }

    all_results = []

    print("Analyzing spatial heterogeneity...")

    for condition, (folder, positions) in conditions.items():
        for pos in positions:
            data_path = base_path / folder / f'Pos{pos}' / 'aphase'

            if not data_path.exists():
                print(f"  Skipping {condition} Pos{pos} - not found")
                continue

            print(f"  Processing {condition} Pos{pos}...")
            df, heatmaps = analyze_time_series(data_path, condition, pos, analyzer)

            if df is not None:
                all_results.append(df)

                # Create example heatmap
                if pos in [101, 201] and len(heatmaps) > 0:
                    for frame_idx, flow_mag in heatmaps[:2]:  # First 2 timepoints
                        fig = create_spatial_heatmap(
                            flow_mag,
                            f'{condition} Pos{pos} Frame {frame_idx}',
                            vmax=1.0
                        )
                        fig.savefig(
                            output_path / f'spatial_heterogeneity_{condition}_Pos{pos}_frame{frame_idx}.png',
                            dpi=150,
                            bbox_inches='tight'
                        )
                        plt.close(fig)

    # Combine results
    results_df = pd.concat(all_results, ignore_index=True)
    results_df.to_csv(output_path / 'spatial_heterogeneity_metrics.csv', index=False)

    # Statistical comparison
    print("\n" + "="*70)
    print("SPATIAL HETEROGENEITY ANALYSIS")
    print("="*70)

    ref_data = results_df[results_df['condition'] == 'REF']
    rif_data = results_df[results_df['condition'] == 'RIF10']

    metrics = ['cv', 'morans_i', 'spatial_entropy', 'edge_interior_ratio',
               'n_hotspots', 'hotspot_area', 'max_hotspot_size']

    comparison = []

    for metric in metrics:
        ref_vals = ref_data[metric].values
        rif_vals = rif_data[metric].values

        t_stat, p_val = stats.ttest_ind(ref_vals, rif_vals)

        comparison.append({
            'metric': metric,
            'REF_mean': np.mean(ref_vals),
            'REF_std': np.std(ref_vals),
            'RIF10_mean': np.mean(rif_vals),
            'RIF10_std': np.std(rif_vals),
            'fold_change': np.mean(rif_vals) / (np.mean(ref_vals) + 1e-10),
            'p_value': p_val
        })

        print(f"\n{metric.upper()}:")
        print(f"  REF:   {np.mean(ref_vals):.4f} ± {np.std(ref_vals):.4f}")
        print(f"  RIF10: {np.mean(rif_vals):.4f} ± {np.std(rif_vals):.4f}")
        print(f"  Fold change: {np.mean(rif_vals) / (np.mean(ref_vals) + 1e-10):.2f}x")
        print(f"  P-value: {p_val:.4e}")

    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv(output_path / 'spatial_comparison.csv', index=False)

    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.ravel()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Box plot comparison
        data_to_plot = [
            ref_data[metric].values,
            rif_data[metric].values
        ]

        bp = ax.boxplot(data_to_plot, labels=['REF', 'RIF10'],
                       patch_artist=True, widths=0.6)

        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')

        # Add individual points
        for i, data in enumerate(data_to_plot):
            y = data
            x = np.random.normal(i+1, 0.04, len(data))
            ax.scatter(x, y, alpha=0.3, s=20, c='black')

        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()}\np={comparison[idx]["p_value"]:.4f}')
        ax.grid(True, alpha=0.3)

    # Time evolution plots
    ax = axes[7]
    for condition, color in [('REF', 'blue'), ('RIF10', 'red')]:
        data = results_df[results_df['condition'] == condition]
        grouped = data.groupby('time')['cv'].agg(['mean', 'std'])
        ax.plot(grouped.index, grouped['mean'], label=condition, color=color, linewidth=2)
        ax.fill_between(grouped.index,
                        grouped['mean'] - grouped['std'],
                        grouped['mean'] + grouped['std'],
                        alpha=0.2, color=color)
    ax.set_xlabel('Time (frames)')
    ax.set_ylabel('Coefficient of Variation')
    ax.set_title('CV Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[8]
    for condition, color in [('REF', 'blue'), ('RIF10', 'red')]:
        data = results_df[results_df['condition'] == condition]
        grouped = data.groupby('time')['n_hotspots'].agg(['mean', 'std'])
        ax.plot(grouped.index, grouped['mean'], label=condition, color=color, linewidth=2)
        ax.fill_between(grouped.index,
                        grouped['mean'] - grouped['std'],
                        grouped['mean'] + grouped['std'],
                        alpha=0.2, color=color)
    ax.set_xlabel('Time (frames)')
    ax.set_ylabel('Number of Hotspots')
    ax.set_title('Hotspots Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path / 'spatial_heterogeneity_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("\n" + "="*70)
    print("INTERPRETATION GUIDE:")
    print("="*70)
    print("\nFor effective antibiotics, we expect:")
    print("  - LOW CV (uniform growth suppression)")
    print("  - LOW Moran's I (no clustering of resistant colonies)")
    print("  - HIGH spatial entropy (uniform distribution)")
    print("  - HIGH edge/interior ratio (no interior growth)")
    print("  - FEW hotspots (no resistant populations)")
    print("\nFor ineffective antibiotics:")
    print("  - HIGH CV (patchy resistant colonies)")
    print("  - HIGH Moran's I (clustered resistant growth)")
    print("  - LOW spatial entropy (concentrated growth)")
    print("  - LOW edge/interior ratio (interior resistant colonies)")
    print("  - MANY hotspots (surviving populations)")
    print("="*70)

    print(f"\nResults saved to {output_path}")
    print("Files created:")
    print("  - spatial_heterogeneity_metrics.csv")
    print("  - spatial_comparison.csv")
    print("  - spatial_heterogeneity_comparison.png")
    print("  - spatial_heterogeneity_[condition]_[pos]_frame[n].png")


if __name__ == '__main__':
    main()