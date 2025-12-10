#!/usr/bin/env python3

import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class FlowAnalyzer:

    def __init__(self, data_path='/home/junming/nobackup_junming', output_path='/home/junming/private/rearch_methodology'):
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.flow_params = {
            'pyr_scale': 0.5,
            'levels': 3,
            'winsize': 15,
            'iterations': 3,
            'poly_n': 5,
            'poly_sigma': 1.2,
            'flags': 0
        }

    def load_time_series(self, condition, position):
        """
        Load images for a given condition and position.

        condition: 'REF' or 'RIF10'
        position: 101-110 for REF, 201-210 for RIF10
        """
        if condition == 'REF':
            raw_dir = self.data_path / 'REF_raw_data101_110' / f'Pos{position}' / 'aphase'
            mask_dir = self.data_path / 'REF_masks101_110' / f'Pos{position}' / 'PreprocessedPhaseMasks'
        else:
            raw_dir = self.data_path / 'RIF10_raw_data201_210' / f'Pos{position}' / 'aphase'
            mask_dir = self.data_path / 'RIF10_masks201_210' / f'Pos{position}' / 'PreprocessedPhaseMasks'

        if not raw_dir.exists():
            print(f"Directory not found: {raw_dir}")
            return None, None

        image_files = sorted(raw_dir.glob('img_*.tiff'))
        mask_files = sorted(mask_dir.glob('MASK_img_*.tif'))

        images = []
        masks = []

        for img_path in image_files:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)

        for mask_path in mask_files:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                masks.append(mask)

        return images, masks

    def compute_optical_flow(self, img1, img2):
        """Calculate dense optical flow between two frames."""
        flow = cv2.calcOpticalFlowFarneback(
            img1, img2,
            None,
            **self.flow_params
        )
        return flow

    def extract_boundary_pixels(self, mask):
        """Get boundary pixels from mask."""
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(mask, kernel, iterations=1)
        boundary = mask - eroded
        return boundary > 0

    def analyze_boundary_flow(self, flow, boundary_mask):
        """Calculate flow metrics at colony boundary."""
        if not np.any(boundary_mask):
            return {'speed': 0, 'coherence': 0, 'radial_speed': 0}

        u = flow[:,:,0][boundary_mask]
        v = flow[:,:,1][boundary_mask]

        speeds = np.sqrt(u**2 + v**2)
        mean_speed = np.mean(speeds)

        if len(u) > 1:
            flow_angles = np.arctan2(v, u)
            coherence = np.abs(np.mean(np.exp(1j * flow_angles)))
        else:
            coherence = 0

        y_coords, x_coords = np.where(boundary_mask)
        if len(x_coords) > 0:
            center_y, center_x = np.mean(y_coords), np.mean(x_coords)
            radial_u = u * (x_coords[boundary_mask[y_coords, x_coords]] - center_x)
            radial_v = v * (y_coords[boundary_mask[y_coords, x_coords]] - center_y)
            distances = np.sqrt((x_coords[boundary_mask[y_coords, x_coords]] - center_x)**2 +
                              (y_coords[boundary_mask[y_coords, x_coords]] - center_y)**2)
            radial_speed = np.sum(radial_u + radial_v) / (np.sum(distances) + 1e-10)
        else:
            radial_speed = 0

        return {
            'speed': mean_speed,
            'coherence': coherence,
            'radial_speed': radial_speed
        }

    def analyze_position(self, condition, position, max_frames=30):
        """Analyze single position time series."""
        print(f"Analyzing {condition} Pos{position}...")

        images, masks = self.load_time_series(condition, position)

        if images is None or len(images) < 2:
            print(f"Not enough images for {condition} Pos{position}")
            return None

        n_frames = min(len(images), max_frames)
        results = []

        for t in range(n_frames - 1):
            flow = self.compute_optical_flow(images[t], images[t+1])

            if t < len(masks):
                boundary = self.extract_boundary_pixels(masks[t])
                metrics = self.analyze_boundary_flow(flow, boundary)
            else:
                metrics = {'speed': 0, 'coherence': 0, 'radial_speed': 0}

            results.append({
                'time': t,
                'condition': condition,
                'position': position,
                'boundary_speed': metrics['speed'],
                'coherence': metrics['coherence'],
                'radial_speed': metrics['radial_speed']
            })

        return pd.DataFrame(results)

    def run_full_analysis(self, max_frames=30):
        """Analyze all positions for both conditions."""
        all_results = []

        ref_positions = range(101, 111)
        rif_positions = range(201, 211)

        print("Processing REF (untreated) samples...")
        for pos in ref_positions:
            df = self.analyze_position('REF', pos, max_frames)
            if df is not None:
                all_results.append(df)

        print("\nProcessing RIF10 (treated) samples...")
        for pos in rif_positions:
            df = self.analyze_position('RIF10', pos, max_frames)
            if df is not None:
                all_results.append(df)

        if not all_results:
            print("No data collected")
            return None

        combined_df = pd.concat(all_results, ignore_index=True)
        return combined_df

    def statistical_analysis(self, df, window_end=15):
        """Compare treated vs untreated within time window."""
        window_data = df[df['time'] < window_end]

        ref_data = window_data[window_data['condition'] == 'REF']['boundary_speed']
        rif_data = window_data[window_data['condition'] == 'RIF10']['boundary_speed']

        if len(ref_data) < 2 or len(rif_data) < 2:
            return None

        t_stat, p_value = stats.ttest_ind(ref_data, rif_data)

        pooled_std = np.sqrt(((len(ref_data) - 1) * np.var(ref_data, ddof=1) +
                             (len(rif_data) - 1) * np.var(rif_data, ddof=1)) /
                            (len(ref_data) + len(rif_data) - 2))

        effect_size = (np.mean(ref_data) - np.mean(rif_data)) / pooled_std if pooled_std > 0 else 0

        return {
            'window_end': window_end,
            't_statistic': t_stat,
            'p_value': p_value,
            'effect_size': effect_size,
            'ref_mean': np.mean(ref_data),
            'rif_mean': np.mean(rif_data),
            'ref_std': np.std(ref_data),
            'rif_std': np.std(rif_data)
        }

    def plot_results(self, df, output_filename='flow_analysis_results.png'):
        """Create visualization of results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: boundary speed over time
        ax = axes[0, 0]
        for condition in ['REF', 'RIF10']:
            data = df[df['condition'] == condition]
            grouped = data.groupby('time')['boundary_speed'].agg(['mean', 'std'])
            ax.plot(grouped.index, grouped['mean'], label=condition, linewidth=2)
            ax.fill_between(grouped.index,
                           grouped['mean'] - grouped['std'],
                           grouped['mean'] + grouped['std'],
                           alpha=0.3)
        ax.axvline(x=15, color='red', linestyle='--', alpha=0.5, label='Detection window')
        ax.set_xlabel('Time point')
        ax.set_ylabel('Boundary speed (pixels/frame)')
        ax.set_title('Colony Boundary Expansion Speed')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: coherence over time
        ax = axes[0, 1]
        for condition in ['REF', 'RIF10']:
            data = df[df['condition'] == condition]
            grouped = data.groupby('time')['coherence'].agg(['mean', 'std'])
            ax.plot(grouped.index, grouped['mean'], label=condition, linewidth=2)
            ax.fill_between(grouped.index,
                           grouped['mean'] - grouped['std'],
                           grouped['mean'] + grouped['std'],
                           alpha=0.3)
        ax.set_xlabel('Time point')
        ax.set_ylabel('Directional coherence')
        ax.set_title('Growth Coordination')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: statistical power over time
        ax = axes[1, 0]
        windows = range(5, 31, 2)
        p_values = []
        effect_sizes = []

        for w in windows:
            stats_result = self.statistical_analysis(df, window_end=w)
            if stats_result:
                p_values.append(stats_result['p_value'])
                effect_sizes.append(stats_result['effect_size'])

        ax.plot(list(windows)[:len(p_values)], p_values, marker='o', label='p-value')
        ax.axhline(y=0.05, color='red', linestyle='--', label='p=0.05')
        ax.set_xlabel('Detection window (frames)')
        ax.set_ylabel('p-value')
        ax.set_title('Statistical Significance vs Window Size')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # Plot 4: effect sizes
        ax = axes[1, 1]
        ax.plot(list(windows)[:len(effect_sizes)], effect_sizes, marker='s', color='purple')
        ax.axhline(y=0.5, color='orange', linestyle='--', label='Medium effect')
        ax.axhline(y=0.8, color='red', linestyle='--', label='Large effect')
        ax.set_xlabel('Detection window (frames)')
        ax.set_ylabel("Cohen's d")
        ax.set_title('Effect Size vs Window Size')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_path / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nResults saved to: {output_path}")

        return fig

def main():
    print("Bacterial Colony Optical Flow Analysis")
    print("=" * 50)

    analyzer = FlowAnalyzer()

    print("\nRunning analysis on first 30 frames...")
    df = analyzer.run_full_analysis(max_frames=30)

    if df is None:
        print("Analysis failed")
        return

    print(f"\nCollected {len(df)} measurements")
    print(f"Conditions: {df['condition'].unique()}")
    print(f"Positions analyzed: {df['position'].nunique()}")

    print("\nStatistical comparison (first 15 frames):")
    stats_result = analyzer.statistical_analysis(df, window_end=15)
    if stats_result:
        print(f"  REF mean speed: {stats_result['ref_mean']:.3f} ± {stats_result['ref_std']:.3f}")
        print(f"  RIF10 mean speed: {stats_result['rif_mean']:.3f} ± {stats_result['rif_std']:.3f}")
        print(f"  t-statistic: {stats_result['t_statistic']:.3f}")
        print(f"  p-value: {stats_result['p_value']:.4f}")
        print(f"  Effect size (Cohen's d): {stats_result['effect_size']:.3f}")

    print("\nGenerating plots...")
    analyzer.plot_results(df)

    csv_path = analyzer.output_path / 'flow_analysis_data.csv'
    df.to_csv(csv_path, index=False)
    print(f"Data saved to: {csv_path}")

if __name__ == '__main__':
    main()