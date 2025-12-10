#!/usr/bin/env python3

import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage
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
        """Load images for a given condition and position."""
        if condition == 'REF':
            raw_dir = self.data_path / 'REF_raw_data101_110' / f'Pos{position}' / 'aphase'
        else:
            raw_dir = self.data_path / 'RIF10_raw_data201_210' / f'Pos{position}' / 'aphase'

        if not raw_dir.exists():
            print(f"Directory not found: {raw_dir}")
            return None

        image_files = sorted(raw_dir.glob('img_*.tiff'))

        images = []
        for img_path in image_files:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)

        return images

    def compute_optical_flow(self, img1, img2):
        """Calculate dense optical flow between two frames."""
        flow = cv2.calcOpticalFlowFarneback(
            img1, img2,
            None,
            **self.flow_params
        )
        return flow

    def detect_colony_region(self, img, threshold_percentile=30):
        """
        Detect colony region using intensity thresholding.
        Colonies are darker than background.
        """
        threshold = np.percentile(img, threshold_percentile)
        colony_mask = img < threshold

        colony_mask = ndimage.binary_opening(colony_mask, iterations=2)
        colony_mask = ndimage.binary_closing(colony_mask, iterations=2)

        labeled, num_features = ndimage.label(colony_mask)
        if num_features == 0:
            return np.zeros_like(img, dtype=bool)

        sizes = ndimage.sum(colony_mask, labeled, range(1, num_features + 1))
        largest_label = np.argmax(sizes) + 1
        colony_mask = labeled == largest_label

        return colony_mask

    def extract_boundary_pixels(self, mask):
        """Get boundary pixels from binary mask."""
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
        boundary = mask.astype(np.uint8) - eroded
        return boundary > 0

    def analyze_flow_metrics(self, flow, mask):
        """Calculate flow metrics within mask region."""
        if not np.any(mask):
            return {
                'mean_speed': 0,
                'median_speed': 0,
                'max_speed': 0,
                'coherence': 0,
                'expansion_rate': 0
            }

        u = flow[:,:,0][mask]
        v = flow[:,:,1][mask]

        speeds = np.sqrt(u**2 + v**2)
        mean_speed = np.mean(speeds)
        median_speed = np.median(speeds)
        max_speed = np.max(speeds)

        if len(u) > 1:
            flow_angles = np.arctan2(v, u)
            coherence = np.abs(np.mean(np.exp(1j * flow_angles)))
        else:
            coherence = 0

        y_coords, x_coords = np.where(mask)
        if len(x_coords) > 0:
            center_y, center_x = np.mean(y_coords), np.mean(x_coords)
            dx = x_coords - center_x
            dy = y_coords - center_y
            distances = np.sqrt(dx**2 + dy**2)

            radial_flow = (u * dx + v * dy) / (distances + 1e-10)
            expansion_rate = np.mean(radial_flow)
        else:
            expansion_rate = 0

        return {
            'mean_speed': mean_speed,
            'median_speed': median_speed,
            'max_speed': max_speed,
            'coherence': coherence,
            'expansion_rate': expansion_rate
        }

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
            colony_mask = self.detect_colony_region(images[t])
            boundary_mask = self.extract_boundary_pixels(colony_mask)

            flow = self.compute_optical_flow(images[t], images[t+1])

            colony_metrics = self.analyze_flow_metrics(flow, colony_mask)
            boundary_metrics = self.analyze_flow_metrics(flow, boundary_mask)

            results.append({
                'time': t,
                'condition': condition,
                'position': position,
                'colony_mean_speed': colony_metrics['mean_speed'],
                'colony_coherence': colony_metrics['coherence'],
                'colony_expansion': colony_metrics['expansion_rate'],
                'boundary_mean_speed': boundary_metrics['mean_speed'],
                'boundary_max_speed': boundary_metrics['max_speed'],
                'colony_area': np.sum(colony_mask)
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

    def create_flow_heatmaps(self, condition, position, frames=[5, 10, 15], output_filename=None):
        """Create heatmaps showing flow magnitude at different time points."""
        images = self.load_time_series(condition, position)

        if images is None or len(images) < max(frames) + 1:
            print(f"Not enough frames for {condition} Pos{position}")
            return None

        n_frames = len(frames)
        fig, axes = plt.subplots(2, n_frames, figsize=(5*n_frames, 10))

        if n_frames == 1:
            axes = axes.reshape(-1, 1)

        for idx, frame_num in enumerate(frames):
            colony_mask = self.detect_colony_region(images[frame_num])
            flow = self.compute_optical_flow(images[frame_num], images[frame_num + 1])
            magnitude = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)

            ax = axes[0, idx]
            im = ax.imshow(magnitude, cmap='hot', interpolation='nearest')
            ax.set_title(f'{condition} Pos{position}\nFrame {frame_num}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Speed')

            ax = axes[1, idx]
            im = ax.imshow(magnitude, cmap='hot', interpolation='nearest')
            boundary = self.extract_boundary_pixels(colony_mask)
            if np.any(boundary):
                ax.contour(boundary, colors='cyan', linewidths=2)
            ax.set_title(f'With detected boundary')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Speed')

        plt.tight_layout()

        if output_filename is None:
            output_filename = f'flow_heatmap_{condition}_Pos{position}.png'

        output_path = self.output_path / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {output_path}")
        plt.close()

        return fig

    def create_comparison_heatmaps(self, ref_position=101, rif_position=201, frames=[5, 10, 15]):
        """Create side-by-side comparison heatmaps for REF vs RIF10."""
        ref_images = self.load_time_series('REF', ref_position)
        rif_images = self.load_time_series('RIF10', rif_position)

        if ref_images is None or rif_images is None:
            print("Could not load comparison data")
            return None

        n_frames = len(frames)
        fig, axes = plt.subplots(2, n_frames, figsize=(6*n_frames, 12))

        if n_frames == 1:
            axes = axes.reshape(-1, 1)

        vmax = None

        for idx, frame_num in enumerate(frames):
            if frame_num + 1 >= len(ref_images) or frame_num + 1 >= len(rif_images):
                continue

            ref_flow = self.compute_optical_flow(ref_images[frame_num], ref_images[frame_num + 1])
            rif_flow = self.compute_optical_flow(rif_images[frame_num], rif_images[frame_num + 1])

            ref_magnitude = np.sqrt(ref_flow[:,:,0]**2 + ref_flow[:,:,1]**2)
            rif_magnitude = np.sqrt(rif_flow[:,:,0]**2 + rif_flow[:,:,1]**2)

            if vmax is None:
                vmax = max(np.percentile(ref_magnitude, 99), np.percentile(rif_magnitude, 99))

            ref_mask = self.detect_colony_region(ref_images[frame_num])
            rif_mask = self.detect_colony_region(rif_images[frame_num])

            ax = axes[0, idx]
            im = ax.imshow(ref_magnitude, cmap='hot', interpolation='nearest', vmin=0, vmax=vmax)
            boundary = self.extract_boundary_pixels(ref_mask)
            if np.any(boundary):
                ax.contour(boundary, colors='cyan', linewidths=2)
            ax.set_title(f'REF Pos{ref_position}\nFrame {frame_num}', fontsize=12)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Speed')

            ax = axes[1, idx]
            im = ax.imshow(rif_magnitude, cmap='hot', interpolation='nearest', vmin=0, vmax=vmax)
            boundary = self.extract_boundary_pixels(rif_mask)
            if np.any(boundary):
                ax.contour(boundary, colors='cyan', linewidths=2)
            ax.set_title(f'RIF10 Pos{rif_position}\nFrame {frame_num}', fontsize=12)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Speed')

        fig.suptitle('Growth Pattern Comparison: Untreated vs Rifampicin', fontsize=14, y=0.995)
        plt.tight_layout()

        output_path = self.output_path / 'flow_comparison_heatmaps.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Comparison heatmaps saved to: {output_path}")
        plt.close()

        return fig

    def plot_time_series(self, df, output_filename='flow_time_series.png'):
        """Plot flow metrics over time."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax = axes[0, 0]
        for condition in ['REF', 'RIF10']:
            data = df[df['condition'] == condition]
            grouped = data.groupby('time')['colony_mean_speed'].agg(['mean', 'std'])
            ax.plot(grouped.index, grouped['mean'], label=condition, linewidth=2)
            ax.fill_between(grouped.index,
                           grouped['mean'] - grouped['std'],
                           grouped['mean'] + grouped['std'],
                           alpha=0.3)
        ax.set_xlabel('Time point')
        ax.set_ylabel('Mean flow speed')
        ax.set_title('Colony Flow Speed Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        for condition in ['REF', 'RIF10']:
            data = df[df['condition'] == condition]
            grouped = data.groupby('time')['colony_expansion'].agg(['mean', 'std'])
            ax.plot(grouped.index, grouped['mean'], label=condition, linewidth=2)
            ax.fill_between(grouped.index,
                           grouped['mean'] - grouped['std'],
                           grouped['mean'] + grouped['std'],
                           alpha=0.3)
        ax.set_xlabel('Time point')
        ax.set_ylabel('Expansion rate')
        ax.set_title('Colony Expansion Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        for condition in ['REF', 'RIF10']:
            data = df[df['condition'] == condition]
            grouped = data.groupby('time')['boundary_mean_speed'].agg(['mean', 'std'])
            ax.plot(grouped.index, grouped['mean'], label=condition, linewidth=2)
            ax.fill_between(grouped.index,
                           grouped['mean'] - grouped['std'],
                           grouped['mean'] + grouped['std'],
                           alpha=0.3)
        ax.set_xlabel('Time point')
        ax.set_ylabel('Boundary flow speed')
        ax.set_title('Boundary Flow Speed')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        for condition in ['REF', 'RIF10']:
            data = df[df['condition'] == condition]
            grouped = data.groupby('time')['colony_coherence'].agg(['mean', 'std'])
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

        plt.tight_layout()
        output_path = self.output_path / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Time series plot saved to: {output_path}")

        return fig

def main():
    print("Bacterial Colony Optical Flow Analysis")
    print("="*50)

    analyzer = FlowAnalyzer()

    print("\nRunning analysis on first 30 frames...")
    df = analyzer.run_full_analysis(max_frames=30)

    if df is None:
        print("Analysis failed")
        return

    print(f"\nCollected {len(df)} measurements")
    print(f"Conditions: {df['condition'].unique()}")
    print(f"Positions analyzed: {df['position'].nunique()}")

    print("\nSummary statistics:")
    summary = df.groupby('condition').agg({
        'colony_mean_speed': ['mean', 'std'],
        'colony_expansion': ['mean', 'std'],
        'boundary_mean_speed': ['mean', 'std']
    })
    print(summary)

    print("\nGenerating plots...")
    analyzer.plot_time_series(df)

    print("\nGenerating flow heatmaps...")
    analyzer.create_comparison_heatmaps(ref_position=101, rif_position=201, frames=[5, 10, 15, 20])

    print("\nGenerating individual heatmaps...")
    analyzer.create_flow_heatmaps('REF', 101, frames=[5, 10, 15])
    analyzer.create_flow_heatmaps('RIF10', 201, frames=[5, 10, 15])

    csv_path = analyzer.output_path / 'flow_analysis_data.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nData saved to: {csv_path}")

    print("\nAnalysis complete!")

if __name__ == '__main__':
    main()