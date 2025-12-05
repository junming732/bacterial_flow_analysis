#!/usr/bin/env python3
"""
Enhanced Spatial Heterogeneity Analysis
Generates comprehensive heatmaps for all positions and timepoints
Optimized for CPU with parallel processing
"""

import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import ndimage, stats
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

class SpatialHeterogeneityAnalyzer:

    def __init__(self, hotspot_percentile=90):
        self.hotspot_percentile = hotspot_percentile

    def calculate_coefficient_of_variation(self, flow_magnitude):
        nonzero = flow_magnitude[flow_magnitude > 0]
        if len(nonzero) == 0:
            return 0.0
        return np.std(nonzero) / (np.mean(nonzero) + 1e-10)

    def calculate_morans_i(self, flow_magnitude, distance_threshold=50):
        h, w = flow_magnitude.shape
        y_coords, x_coords = np.meshgrid(
            np.arange(0, h, 10),
            np.arange(0, w, 10),
            indexing='ij'
        )

        coords = np.column_stack([y_coords.ravel(), x_coords.ravel()])
        values = flow_magnitude[y_coords.ravel(), x_coords.ravel()]

        mask = values > 0
        coords = coords[mask]
        values = values[mask]

        if len(values) < 10:
            return 0.0

        n = len(values)
        mean_val = np.mean(values)

        numerator = 0
        denominator = 0
        weight_sum = 0

        for i in range(min(n, 200)):
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
        threshold = np.percentile(flow_magnitude[flow_magnitude > 0],
                                 self.hotspot_percentile) if np.any(flow_magnitude > 0) else 0

        if threshold == 0:
            return 0, 0.0, 0

        hotspot_mask = flow_magnitude > threshold
        labeled, num_features = ndimage.label(hotspot_mask)

        if num_features == 0:
            return 0, 0.0, 0

        total_pixels = np.prod(flow_magnitude.shape)
        hotspot_area = np.sum(hotspot_mask) / total_pixels

        sizes = ndimage.sum(hotspot_mask, labeled, range(1, num_features + 1))
        max_hotspot = np.max(sizes) / total_pixels if len(sizes) > 0 else 0

        return num_features, hotspot_area, max_hotspot

    def calculate_spatial_entropy(self, flow_magnitude, n_bins=20):
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

        probs = cells / np.sum(cells)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))

        return entropy

    def analyze_growth_front(self, flow_magnitude):
        h, w = flow_magnitude.shape
        edge_mask = np.zeros((h, w), dtype=bool)
        border = int(min(h, w) * 0.2)
        edge_mask[:border, :] = True
        edge_mask[-border:, :] = True
        edge_mask[:, :border] = True
        edge_mask[:, -border:] = True

        edge_flow = np.mean(flow_magnitude[edge_mask])
        interior_flow = np.mean(flow_magnitude[~edge_mask])

        edge_to_interior_ratio = edge_flow / (interior_flow + 1e-10)

        return edge_to_interior_ratio

    def analyze_frame(self, flow_magnitude):
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
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im1 = axes[0].imshow(flow_magnitude, cmap='hot', vmax=vmax)
    axes[0].set_title(f'{title}\nFlow Magnitude')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)

    threshold = np.percentile(flow_magnitude[flow_magnitude > 0], 90) if np.any(flow_magnitude > 0) else 0
    hotspot_mask = flow_magnitude > threshold

    axes[1].imshow(flow_magnitude, cmap='gray', alpha=0.3)
    axes[1].imshow(hotspot_mask, cmap='Reds', alpha=0.7)
    axes[1].set_title(f'Hotspots (>90th percentile)\nThreshold: {threshold:.3f}')
    axes[1].axis('off')

    h, w = flow_magnitude.shape
    border = int(min(h, w) * 0.2)
    edge_mask = np.zeros((h, w), dtype=bool)
    edge_mask[:border, :] = True
    edge_mask[-border:, :] = True
    edge_mask[:, :border] = True
    edge_mask[:, -border:] = True

    viz = np.zeros((h, w, 3))
    viz[edge_mask] = [0, 1, 0]
    viz[~edge_mask] = [0, 0, 1]

    axes[2].imshow(viz, alpha=0.3)
    axes[2].imshow(flow_magnitude, cmap='hot', alpha=0.7)
    axes[2].set_title('Growth Location\nGreen=Edge, Blue=Interior')
    axes[2].axis('off')

    plt.tight_layout()
    return fig


def create_comparison_figure(ref_flow, rif_flow, frame_idx):
    """Create side-by-side comparison of REF vs RIF10"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    vmax = max(np.max(ref_flow), np.max(rif_flow))

    # REF row
    im1 = axes[0, 0].imshow(ref_flow, cmap='hot', vmax=vmax)
    axes[0, 0].set_title(f'REF Frame {frame_idx}\nFlow Magnitude')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)

    ref_threshold = np.percentile(ref_flow[ref_flow > 0], 90) if np.any(ref_flow > 0) else 0
    ref_hotspot = ref_flow > ref_threshold
    axes[0, 1].imshow(ref_flow, cmap='gray', alpha=0.3)
    axes[0, 1].imshow(ref_hotspot, cmap='Reds', alpha=0.7)
    axes[0, 1].set_title(f'REF Hotspots\nThreshold: {ref_threshold:.3f}')
    axes[0, 1].axis('off')

    h, w = ref_flow.shape
    border = int(min(h, w) * 0.2)
    edge_mask = np.zeros((h, w), dtype=bool)
    edge_mask[:border, :] = True
    edge_mask[-border:, :] = True
    edge_mask[:, :border] = True
    edge_mask[:, -border:] = True

    viz = np.zeros((h, w, 3))
    viz[edge_mask] = [0, 1, 0]
    viz[~edge_mask] = [0, 0, 1]
    axes[0, 2].imshow(viz, alpha=0.3)
    axes[0, 2].imshow(ref_flow, cmap='hot', alpha=0.7)
    axes[0, 2].set_title('REF Growth Location')
    axes[0, 2].axis('off')

    # RIF10 row
    im2 = axes[1, 0].imshow(rif_flow, cmap='hot', vmax=vmax)
    axes[1, 0].set_title(f'RIF10 Frame {frame_idx}\nFlow Magnitude')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    rif_threshold = np.percentile(rif_flow[rif_flow > 0], 90) if np.any(rif_flow > 0) else 0
    rif_hotspot = rif_flow > rif_threshold
    axes[1, 1].imshow(rif_flow, cmap='gray', alpha=0.3)
    axes[1, 1].imshow(rif_hotspot, cmap='Reds', alpha=0.7)
    axes[1, 1].set_title(f'RIF10 Hotspots\nThreshold: {rif_threshold:.3f}')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(viz, alpha=0.3)
    axes[1, 2].imshow(rif_flow, cmap='hot', alpha=0.7)
    axes[1, 2].set_title('RIF10 Growth Location')
    axes[1, 2].axis('off')

    plt.tight_layout()
    return fig


def process_position(args):
    """Process one position - for parallel execution"""
    condition, folder, pos, base_path, analyzer, save_heatmaps = args

    data_path = base_path / folder / f'Pos{pos}' / 'aphase'

    if not data_path.exists():
        return None

    img_files = sorted(data_path.glob(f'img_*.tiff'))

    if len(img_files) < 2:
        return None

    results = []
    heatmap_data = []

    for i in range(len(img_files) - 1):
        img1 = cv2.imread(str(img_files[i]), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(img_files[i + 1]), cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            continue

        flow_mag, flow = compute_optical_flow_direct(img1, img2)
        spatial_metrics = analyzer.analyze_frame(flow_mag)

        results.append({
            'condition': condition,
            'position': pos,
            'time': i,
            'mean_flow': np.mean(flow_mag),
            **spatial_metrics
        })

        if save_heatmaps:
            heatmap_data.append((i, flow_mag))

    return {
        'results': pd.DataFrame(results),
        'heatmaps': heatmap_data,
        'condition': condition,
        'position': pos
    }


def main():
    base_path = Path('/home/junming/nobackup_junming')
    output_path = Path('/home/junming/private/rearch_methodology')
    output_path.mkdir(exist_ok=True)

    # Create subdirectories for organization
    heatmap_dir = output_path / 'heatmaps'
    heatmap_dir.mkdir(exist_ok=True)
    comparison_dir = output_path / 'comparisons'
    comparison_dir.mkdir(exist_ok=True)

    analyzer = SpatialHeterogeneityAnalyzer(hotspot_percentile=90)

    conditions = {
        'REF': ('REF_raw_data101_110', range(101, 111)),
        'RIF10': ('RIF10_raw_data201_210', range(201, 211))
    }

    print("Analyzing spatial heterogeneity with comprehensive heatmap generation...")
    print(f"Using {cpu_count()} CPU cores for parallel processing")

    # Prepare tasks for parallel processing
    tasks = []
    for condition, (folder, positions) in conditions.items():
        for pos in positions:
            # Save heatmaps for first 3 positions of each condition
            save_heatmaps = pos in [101, 102, 103, 201, 202, 203]
            tasks.append((condition, folder, pos, base_path, analyzer, save_heatmaps))

    # Process in parallel
    with Pool(processes=min(cpu_count(), 8)) as pool:
        position_results = pool.map(process_position, tasks)

    # Collect results
    all_results = []
    heatmap_storage = {'REF': {}, 'RIF10': {}}

    for result in position_results:
        if result is not None:
            all_results.append(result['results'])
            if len(result['heatmaps']) > 0:
                heatmap_storage[result['condition']][result['position']] = result['heatmaps']
            print(f"  Processed {result['condition']} Pos{result['position']}")

    print(f"\nCollected data from {len(all_results)} positions")
    print(f"REF positions with heatmaps: {sorted(heatmap_storage['REF'].keys())}")
    print(f"RIF10 positions with heatmaps: {sorted(heatmap_storage['RIF10'].keys())}")

    # Save individual heatmaps
    print("\nGenerating individual heatmaps...")
    for condition in ['REF', 'RIF10']:
        for pos, heatmaps in heatmap_storage[condition].items():
            # Save every 5th frame plus key frames
            key_frames = [0, 5, 10, 15, 20, 30, 40, 50]
            for frame_idx, flow_mag in heatmaps:
                if frame_idx in key_frames:
                    fig = create_spatial_heatmap(
                        flow_mag,
                        f'{condition} Pos{pos} Frame {frame_idx}',
                        vmax=1.0
                    )
                    fig.savefig(
                        heatmap_dir / f'spatial_{condition}_Pos{pos}_frame{frame_idx:03d}.png',
                        dpi=150,
                        bbox_inches='tight'
                    )
                    plt.close(fig)

    # Create comparison figures
    print("\nGenerating comparison figures...")
    if len(heatmap_storage['REF']) > 0 and len(heatmap_storage['RIF10']) > 0:
        ref_positions = sorted(heatmap_storage['REF'].keys())
        rif_positions = sorted(heatmap_storage['RIF10'].keys())

        for ref_pos, rif_pos in zip(ref_positions, rif_positions):
            ref_heatmaps = dict(heatmap_storage['REF'][ref_pos])
            rif_heatmaps = dict(heatmap_storage['RIF10'][rif_pos])

            common_frames = set(ref_heatmaps.keys()) & set(rif_heatmaps.keys())
            key_frames = [0, 5, 10, 20, 40]

            for frame_idx in key_frames:
                if frame_idx in common_frames:
                    fig = create_comparison_figure(
                        ref_heatmaps[frame_idx],
                        rif_heatmaps[frame_idx],
                        frame_idx
                    )
                    fig.savefig(
                        comparison_dir / f'comparison_REF{ref_pos}_vs_RIF{rif_pos}_frame{frame_idx:03d}.png',
                        dpi=150,
                        bbox_inches='tight'
                    )
                    plt.close(fig)
    else:
        print("  Skipping comparisons - missing REF or RIF10 data")

    # Statistical analysis (same as before)
    results_df = pd.concat(all_results, ignore_index=True)
    results_df.to_csv(output_path / 'spatial_heterogeneity_detailed.csv', index=False)

    print("\n" + "="*70)
    print("COMPREHENSIVE SPATIAL HETEROGENEITY ANALYSIS")
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
    comparison_df.to_csv(output_path / 'spatial_comparison_detailed.csv', index=False)

    print("\n" + "="*70)
    print(f"Results saved to {output_path}")
    print("\nGenerated files:")
    print(f"  - {len(list(heatmap_dir.glob('*.png')))} individual heatmaps in heatmaps/")
    print(f"  - {len(list(comparison_dir.glob('*.png')))} comparison figures in comparisons/")
    print("  - spatial_heterogeneity_detailed.csv")
    print("  - spatial_comparison_detailed.csv")
    print("="*70)


if __name__ == '__main__':
    main()