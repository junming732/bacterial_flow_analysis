#!/usr/bin/env python3

import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

class FixedFlowAnalyzer:

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

    def get_bacteria_mask(self, img):
        """
        Bacteria appear as bright/high-contrast features on right side.
        Use adaptive thresholding to find them.
        """
        # Focus on right half where bacteria are
        right_half = img[:, img.shape[1]//2:]

        # Bacteria have high local contrast
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)

        # Dilate edges to get regions
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # Keep only regions on right side
        mask = np.zeros_like(img, dtype=bool)
        mask[:, img.shape[1]//3:] = dilated[:, img.shape[1]//3:] > 0

        # Clean up
        mask = ndimage.binary_closing(mask, iterations=2)
        mask = ndimage.binary_opening(mask, iterations=1)

        return mask

    def analyze_position(self, condition, position, max_frames=30):
        print(f"Analyzing {condition} Pos{position}...")

        images = self.load_time_series(condition, position)

        if images is None or len(images) < 2:
            print(f"Not enough images for {condition} Pos{position}")
            return None

        n_frames = min(len(images), max_frames)
        results = []

        for t in range(n_frames - 1):
            bacteria_mask = self.get_bacteria_mask(images[t])
            flow = self.compute_optical_flow(images[t], images[t+1])
            magnitude = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)

            # Flow in bacteria regions
            bacteria_flow = magnitude[bacteria_mask]

            # High activity regions
            high_activity = magnitude > 1.0
            bacteria_high_activity = bacteria_mask & high_activity

            mean_flow_bacteria = bacteria_flow.mean() if len(bacteria_flow) > 0 else 0
            max_flow_bacteria = bacteria_flow.max() if len(bacteria_flow) > 0 else 0
            p75_flow_bacteria = np.percentile(bacteria_flow, 75) if len(bacteria_flow) > 0 else 0
            p90_flow_bacteria = np.percentile(bacteria_flow, 90) if len(bacteria_flow) > 0 else 0

            # Overall flow statistics
            mean_flow_all = magnitude.mean()
            max_flow_all = magnitude.max()

            # Count pixels
            bacteria_pixels = np.sum(bacteria_mask)
            high_activity_bacteria_pixels = np.sum(bacteria_high_activity)

            results.append({
                'time': t,
                'condition': condition,
                'position': position,
                'mean_flow_bacteria': mean_flow_bacteria,
                'max_flow_bacteria': max_flow_bacteria,
                'p75_flow_bacteria': p75_flow_bacteria,
                'p90_flow_bacteria': p90_flow_bacteria,
                'mean_flow_all': mean_flow_all,
                'max_flow_all': max_flow_all,
                'bacteria_pixels': bacteria_pixels,
                'high_activity_pixels': high_activity_bacteria_pixels
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

    def plot_results(self, df, output_filename='flow_results_fixed.png'):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        metrics = [
            ('mean_flow_bacteria', 'Mean Flow Speed in Bacteria Regions'),
            ('p75_flow_bacteria', '75th Percentile Flow in Bacteria'),
            ('p90_flow_bacteria', '90th Percentile Flow in Bacteria'),
            ('max_flow_bacteria', 'Max Flow in Bacteria'),
            ('bacteria_pixels', 'Detected Bacteria Pixels'),
            ('high_activity_pixels', 'High Activity Bacteria Pixels')
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
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_path / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Results saved to: {output_path}")
        plt.close()

    def create_diagnostic(self, condition, position, frame=15):
        images = self.load_time_series(condition, position)

        if images is None or frame + 1 >= len(images):
            return None

        bacteria_mask = self.get_bacteria_mask(images[frame])
        flow = self.compute_optical_flow(images[frame], images[frame + 1])
        magnitude = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        axes[0, 0].imshow(images[frame], cmap='gray')
        axes[0, 0].set_title(f'{condition} Pos{position} Frame {frame}\nOriginal')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(bacteria_mask, cmap='RdYlGn')
        axes[0, 1].set_title(f'Detected Bacteria\n{np.sum(bacteria_mask)} pixels')
        axes[0, 1].axis('off')

        im = axes[0, 2].imshow(magnitude, cmap='hot')
        axes[0, 2].set_title(f'Flow Magnitude\nMean: {magnitude.mean():.2f}')
        axes[0, 2].axis('off')
        plt.colorbar(im, ax=axes[0, 2], fraction=0.046)

        bacteria_flow = magnitude.copy()
        bacteria_flow[~bacteria_mask] = 0
        im = axes[1, 0].imshow(bacteria_flow, cmap='hot')
        axes[1, 0].set_title(f'Flow in Bacteria\nMean: {magnitude[bacteria_mask].mean():.2f}')
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

        overlay = cv2.cvtColor(images[frame], cv2.COLOR_GRAY2RGB)
        overlay[bacteria_mask, 1] = 255
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Bacteria Mask Overlay')
        axes[1, 1].axis('off')

        overlay2 = cv2.cvtColor(images[frame], cv2.COLOR_GRAY2RGB)
        flow_vis = (magnitude / magnitude.max() * 255).astype(np.uint8)
        flow_colored = cv2.applyColorMap(flow_vis, cv2.COLORMAP_HOT)
        high_flow = magnitude > 1.0
        overlay2[high_flow & bacteria_mask] = flow_colored[high_flow & bacteria_mask]
        axes[1, 2].imshow(overlay2)
        axes[1, 2].set_title('High Flow in Bacteria')
        axes[1, 2].axis('off')

        plt.tight_layout()
        output_path = self.output_path / f'diagnostic_fixed_{condition}_Pos{position}.png'
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"Diagnostic saved to: {output_path}")
        plt.close()

def main():
    print("Fixed Optical Flow Analysis")
    print("="*50)

    analyzer = FixedFlowAnalyzer()

    print("\nRunning analysis...")
    df = analyzer.run_full_analysis(max_frames=30)

    if df is None:
        print("Analysis failed")
        return

    print(f"\nCollected {len(df)} measurements")

    print("\nSummary by condition:")
    summary = df.groupby('condition').agg({
        'mean_flow_bacteria': ['mean', 'std'],
        'p75_flow_bacteria': ['mean', 'std'],
        'max_flow_bacteria': ['mean', 'std']
    })
    print(summary)

    print("\nGenerating plots...")
    analyzer.plot_results(df)

    print("\nGenerating diagnostics...")
    analyzer.create_diagnostic('REF', 101, frame=15)
    analyzer.create_diagnostic('RIF10', 201, frame=15)

    csv_path = analyzer.output_path / 'flow_analysis_fixed.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nData saved to: {csv_path}")

if __name__ == '__main__':
    main()