#!/usr/bin/env python3

import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

class SimpleFlowAnalyzer:

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
        """Load images for a given condition and position."""
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
        """Calculate dense optical flow."""
        flow = cv2.calcOpticalFlowFarneback(img1, img2, None, **self.flow_params)
        return flow

    def get_colony_mask(self, img, condition):
        """Get colony region using appropriate threshold for each condition."""
        if condition == 'REF':
            threshold = np.percentile(img, 40)
        else:
            threshold = np.percentile(img, 30)

        mask = img < threshold
        mask = ndimage.binary_opening(mask, iterations=2)
        mask = ndimage.binary_closing(mask, iterations=2)

        labeled, num_features = ndimage.label(mask)
        if num_features > 0:
            sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
            largest_label = np.argmax(sizes) + 1
            mask = labeled == largest_label

        return mask

    def get_active_region(self, flow, threshold=0.5):
        """Get regions with significant optical flow."""
        magnitude = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
        active = magnitude > threshold
        return active

    def analyze_position(self, condition, position, max_frames=30):
        """Analyze single position time series."""
        print(f"Analyzing {condition} Pos{position}...")

        images = self.load_time_series(condition, position)

        if images is None or len(images) < 2:
            print(f"Not enough images for {condition} Pos{position}")
            return None

        n_frames = min(len(images), max_frames)
        results = []

        for t in range(n_frames - 1):
            colony_mask = self.get_colony_mask(images[t], condition)
            flow = self.compute_optical_flow(images[t], images[t+1])
            active_mask = self.get_active_region(flow, threshold=0.3)

            magnitude = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)

            colony_flow = magnitude[colony_mask]
            active_flow = magnitude[active_mask]
            colony_active_flow = magnitude[colony_mask & active_mask]

            all_mean = magnitude.mean()
            colony_mean = colony_flow.mean() if len(colony_flow) > 0 else 0
            active_mean = active_flow.mean() if len(active_flow) > 0 else 0
            colony_active_mean = colony_active_flow.mean() if len(colony_active_flow) > 0 else 0

            all_75th = np.percentile(magnitude, 75)
            colony_75th = np.percentile(colony_flow, 75) if len(colony_flow) > 0 else 0

            active_pixels = np.sum(active_mask)
            colony_active_pixels = np.sum(colony_mask & active_mask)

            u = flow[:,:,0][colony_mask & active_mask]
            v = flow[:,:,1][colony_mask & active_mask]

            if len(u) > 1:
                flow_angles = np.arctan2(v, u)
                coherence = np.abs(np.mean(np.exp(1j * flow_angles)))
            else:
                coherence = 0

            results.append({
                'time': t,
                'condition': condition,
                'position': position,
                'all_mean_speed': all_mean,
                'colony_mean_speed': colony_mean,
                'active_mean_speed': active_mean,
                'colony_active_mean_speed': colony_active_mean,
                'colony_75th_speed': colony_75th,
                'all_75th_speed': all_75th,
                'active_pixels': active_pixels,
                'colony_active_pixels': colony_active_pixels,
                'coherence': coherence,
                'colony_area': np.sum(colony_mask)
            })

        return pd.DataFrame(results)

    def run_full_analysis(self, max_frames=30):
        """Analyze all positions."""
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

    def plot_results(self, df, output_filename='flow_results_v3.png'):
        """Plot comprehensive flow analysis."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        metrics = [
            ('colony_active_mean_speed', 'Mean Speed in Active Colony Regions'),
            ('colony_75th_speed', '75th Percentile Speed in Colony'),
            ('colony_active_pixels', 'Active Pixels in Colony'),
            ('coherence', 'Directional Coherence'),
            ('colony_mean_speed', 'Mean Speed in All Colony Regions'),
            ('active_mean_speed', 'Mean Speed in All Active Regions')
        ]

        for idx, (metric, title) in enumerate(metrics):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]

            for condition in ['REF', 'RIF10']:
                data = df[df['condition'] == condition]
                grouped = data.groupby('time')[metric].agg(['mean', 'std'])

                ax.plot(grouped.index, grouped['mean'], label=condition, linewidth=2, marker='o', markersize=4)
                ax.fill_between(grouped.index,
                               grouped['mean'] - grouped['std'],
                               grouped['mean'] + grouped['std'],
                               alpha=0.3)

            ax.set_xlabel('Time point', fontsize=10)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
            ax.set_title(title, fontsize=11)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_path / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Results saved to: {output_path}")
        plt.close()

    def create_diagnostic_heatmap(self, condition, position, frame=15, output_filename=None):
        """Create diagnostic showing colony mask, active regions, and flow."""
        images = self.load_time_series(condition, position)

        if images is None or frame + 1 >= len(images):
            return None

        colony_mask = self.get_colony_mask(images[frame], condition)
        flow = self.compute_optical_flow(images[frame], images[frame + 1])
        magnitude = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
        active_mask = self.get_active_region(flow, threshold=0.3)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        axes[0, 0].imshow(images[frame], cmap='gray')
        axes[0, 0].set_title(f'{condition} Pos{position} Frame {frame}\nOriginal Image')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(colony_mask, cmap='RdYlGn')
        axes[0, 1].set_title(f'Detected Colony\nArea: {np.sum(colony_mask)} pixels')
        axes[0, 1].axis('off')

        im = axes[0, 2].imshow(magnitude, cmap='hot')
        axes[0, 2].set_title(f'Optical Flow Magnitude\nMean: {magnitude.mean():.2f}')
        axes[0, 2].axis('off')
        plt.colorbar(im, ax=axes[0, 2], fraction=0.046)

        axes[1, 0].imshow(active_mask, cmap='RdYlGn')
        axes[1, 0].set_title(f'Active Regions (flow > 0.3)\nPixels: {np.sum(active_mask)}')
        axes[1, 0].axis('off')

        combined = colony_mask & active_mask
        axes[1, 1].imshow(combined, cmap='RdYlGn')
        axes[1, 1].set_title(f'Colony + Active\nPixels: {np.sum(combined)}')
        axes[1, 1].axis('off')

        overlay = images[frame].copy()
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
        flow_vis = magnitude.copy()
        flow_vis = (flow_vis / flow_vis.max() * 255).astype(np.uint8)
        flow_colored = cv2.applyColorMap(flow_vis, cv2.COLORMAP_HOT)
        overlay[combined] = flow_colored[combined]

        axes[1, 2].imshow(overlay)
        axes[1, 2].set_title('Flow in Active Colony Regions')
        axes[1, 2].axis('off')

        plt.tight_layout()

        if output_filename is None:
            output_filename = f'diagnostic_{condition}_Pos{position}_frame{frame}.png'

        output_path = self.output_path / output_filename
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"Diagnostic saved to: {output_path}")
        plt.close()

def main():
    print("Simple Optical Flow Analysis")
    print("="*50)

    analyzer = SimpleFlowAnalyzer()

    print("\nRunning analysis...")
    df = analyzer.run_full_analysis(max_frames=30)

    if df is None:
        print("Analysis failed")
        return

    print(f"\nCollected {len(df)} measurements")

    print("\nSummary statistics by condition:")
    summary = df.groupby('condition').agg({
        'colony_active_mean_speed': ['mean', 'std'],
        'colony_75th_speed': ['mean', 'std'],
        'colony_active_pixels': ['mean', 'std']
    })
    print(summary)

    print("\nGenerating plots...")
    analyzer.plot_results(df)

    print("\nGenerating diagnostic heatmaps...")
    analyzer.create_diagnostic_heatmap('REF', 101, frame=15)
    analyzer.create_diagnostic_heatmap('RIF10', 201, frame=15)

    csv_path = analyzer.output_path / 'flow_analysis_v3.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nData saved to: {csv_path}")

    print("\nDone!")

if __name__ == '__main__':
    main()