#!/usr/bin/env python3

import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DirectFlowAnalyzer:

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
        if condition == 'REF':
            raw_dir = self.data_path / 'REF_raw_data101_110' / f'Pos{position}' / 'aphase'
        else:
            raw_dir = self.data_path / 'RIF10_raw_data201_210' / f'Pos{position}' / 'aphase'

        if not raw_dir.exists():
            return None

        image_files = sorted(raw_dir.glob('img_*.tiff'))
        images = []
        for img_path in image_files:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)

        return images

    def compute_optical_flow(self, img1, img2):
        flow = cv2.calcOpticalFlowFarneback(img1, img2, None, **self.flow_params)
        return flow

    def analyze_position(self, condition, position, max_frames=30):
        print(f"Analyzing {condition} Pos{position}...")

        images = self.load_time_series(condition, position)

        if images is None or len(images) < 2:
            print(f"Not enough images")
            return None

        n_frames = min(len(images), max_frames)
        results = []

        for t in range(n_frames - 1):
            flow = self.compute_optical_flow(images[t], images[t+1])
            magnitude = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)

            # Just measure flow statistics directly
            results.append({
                'time': t,
                'condition': condition,
                'position': position,
                'mean_flow': magnitude.mean(),
                'median_flow': np.median(magnitude),
                'p75_flow': np.percentile(magnitude, 75),
                'p90_flow': np.percentile(magnitude, 90),
                'p95_flow': np.percentile(magnitude, 95),
                'p99_flow': np.percentile(magnitude, 99),
                'max_flow': magnitude.max(),
                'std_flow': magnitude.std()
            })

        return pd.DataFrame(results)

    def run_full_analysis(self, max_frames=30):
        all_results = []

        ref_positions = range(101, 111)
        rif_positions = range(201, 211)

        print("Processing REF (untreated)...")
        for pos in ref_positions:
            df = self.analyze_position('REF', pos, max_frames)
            if df is not None:
                all_results.append(df)

        print("\nProcessing RIF10 (treated)...")
        for pos in rif_positions:
            df = self.analyze_position('RIF10', pos, max_frames)
            if df is not None:
                all_results.append(df)

        if not all_results:
            return None

        return pd.concat(all_results, ignore_index=True)

    def plot_results(self, df, output_filename='flow_results_direct.png'):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        metrics = [
            ('mean_flow', 'Mean Flow Speed'),
            ('median_flow', 'Median Flow Speed'),
            ('p75_flow', '75th Percentile Flow'),
            ('p90_flow', '90th Percentile Flow'),
            ('p99_flow', '99th Percentile Flow'),
            ('max_flow', 'Maximum Flow')
        ]

        for idx, (metric, title) in enumerate(metrics):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]

            for condition in ['REF', 'RIF10']:
                data = df[df['condition'] == condition]
                grouped = data.groupby('time')[metric].agg(['mean', 'std'])

                ax.plot(grouped.index, grouped['mean'], label=condition, linewidth=2, marker='o', markersize=3)
                ax.fill_between(grouped.index,
                               grouped['mean'] - grouped['std'],
                               grouped['mean'] + grouped['std'],
                               alpha=0.3)

            ax.set_xlabel('Time point')
            ax.set_ylabel('Flow magnitude (pixels/frame)')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_path / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Results saved to: {output_path}")
        plt.close()

    def create_flow_heatmaps(self, condition, position, frames=[5, 10, 15, 20], output_filename=None):
        images = self.load_time_series(condition, position)

        if images is None or max(frames) + 1 >= len(images):
            return None

        n_frames = len(frames)
        fig, axes = plt.subplots(1, n_frames, figsize=(5*n_frames, 5))

        if n_frames == 1:
            axes = [axes]

        vmax = None
        magnitudes = []

        for frame_num in frames:
            flow = self.compute_optical_flow(images[frame_num], images[frame_num + 1])
            magnitude = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
            magnitudes.append(magnitude)

        vmax = max(np.percentile(m, 99) for m in magnitudes)

        for idx, (frame_num, magnitude) in enumerate(zip(frames, magnitudes)):
            ax = axes[idx]
            im = ax.imshow(magnitude, cmap='hot', interpolation='nearest', vmin=0, vmax=vmax)
            ax.set_title(f'{condition} Pos{position}\nFrame {frame_num}\nMean: {magnitude.mean():.2f}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Speed')

        plt.tight_layout()

        if output_filename is None:
            output_filename = f'heatmap_direct_{condition}_Pos{position}.png'

        output_path = self.output_path / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {output_path}")
        plt.close()

    def create_comparison_heatmaps(self, ref_position=101, rif_position=201, frames=[5, 10, 15, 20]):
        ref_images = self.load_time_series('REF', ref_position)
        rif_images = self.load_time_series('RIF10', rif_position)

        if ref_images is None or rif_images is None:
            return None

        n_frames = len(frames)
        fig, axes = plt.subplots(2, n_frames, figsize=(6*n_frames, 12))

        if n_frames == 1:
            axes = axes.reshape(-1, 1)

        all_magnitudes = []

        for frame_num in frames:
            if frame_num + 1 < len(ref_images):
                ref_flow = self.compute_optical_flow(ref_images[frame_num], ref_images[frame_num + 1])
                ref_magnitude = np.sqrt(ref_flow[:,:,0]**2 + ref_flow[:,:,1]**2)
                all_magnitudes.append(ref_magnitude)

            if frame_num + 1 < len(rif_images):
                rif_flow = self.compute_optical_flow(rif_images[frame_num], rif_images[frame_num + 1])
                rif_magnitude = np.sqrt(rif_flow[:,:,0]**2 + rif_flow[:,:,1]**2)
                all_magnitudes.append(rif_magnitude)

        vmax = max(np.percentile(m, 99) for m in all_magnitudes)

        for idx, frame_num in enumerate(frames):
            if frame_num + 1 >= len(ref_images) or frame_num + 1 >= len(rif_images):
                continue

            ref_flow = self.compute_optical_flow(ref_images[frame_num], ref_images[frame_num + 1])
            rif_flow = self.compute_optical_flow(rif_images[frame_num], rif_images[frame_num + 1])

            ref_magnitude = np.sqrt(ref_flow[:,:,0]**2 + ref_flow[:,:,1]**2)
            rif_magnitude = np.sqrt(rif_flow[:,:,0]**2 + rif_flow[:,:,1]**2)

            ax = axes[0, idx]
            im = ax.imshow(ref_magnitude, cmap='hot', interpolation='nearest', vmin=0, vmax=vmax)
            ax.set_title(f'REF Pos{ref_position}\nFrame {frame_num}\nMean: {ref_magnitude.mean():.2f}', fontsize=12)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Speed')

            ax = axes[1, idx]
            im = ax.imshow(rif_magnitude, cmap='hot', interpolation='nearest', vmin=0, vmax=vmax)
            ax.set_title(f'RIF10 Pos{rif_position}\nFrame {frame_num}\nMean: {rif_magnitude.mean():.2f}', fontsize=12)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Speed')

        fig.suptitle('Growth Pattern Comparison: Untreated vs Rifampicin', fontsize=14, y=0.995)
        plt.tight_layout()

        output_path = self.output_path / 'comparison_heatmaps_direct.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Comparison saved to: {output_path}")
        plt.close()

def main():
    print("Direct Optical Flow Analysis - No Masks")
    print("="*50)

    analyzer = DirectFlowAnalyzer()

    print("\nRunning analysis...")
    df = analyzer.run_full_analysis(max_frames=30)

    if df is None:
        print("Analysis failed")
        return

    print(f"\nCollected {len(df)} measurements")

    print("\nSummary by condition:")
    summary = df.groupby('condition').agg({
        'mean_flow': ['mean', 'std'],
        'p75_flow': ['mean', 'std'],
        'p90_flow': ['mean', 'std'],
        'max_flow': ['mean', 'std']
    })
    print(summary)

    print("\nGenerating plots...")
    analyzer.plot_results(df)

    print("\nGenerating heatmaps...")
    analyzer.create_comparison_heatmaps(ref_position=101, rif_position=201, frames=[5, 10, 15, 20])
    analyzer.create_flow_heatmaps('REF', 101, frames=[5, 10, 15, 20])
    analyzer.create_flow_heatmaps('RIF10', 201, frames=[5, 10, 15, 20])

    csv_path = analyzer.output_path / 'flow_analysis_direct.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nData saved to: {csv_path}")

    print("\nDone!")

if __name__ == '__main__':
    main()