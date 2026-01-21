import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def process_position(pos_path, condition, output_dir):
    """Processes a single position folder and saves dual-threshold visualizations."""
    pos_name = pos_path.name
    aphase_path = pos_path / 'aphase'
    img_files = sorted(list(aphase_path.glob('*.tiff')) + list(aphase_path.glob('*.tif'))) if aphase_path.exists() else sorted(list(pos_path.glob('*.tiff')) + list(pos_path.glob('*.tif')))

    if len(img_files) < 2:
        return

    print(f"  Processing {condition} {pos_name}...")

    # Process every 10th frame
    for i in range(0, len(img_files) - 1, 10):
        img1 = cv2.imread(str(img_files[i]), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(img_files[i+1]), cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None: continue

        # 1. Calculate Optical Flow
        flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # 2. Calculate 80th Percentile Threshold
        # We only consider non-zero pixels to avoid skewing the percentile
        valid_mag = mag[mag > 0]
        threshold = np.percentile(valid_mag, 80) if valid_mag.size > 0 else 0

        # 3. Create Masked Vectors for the "Clean" plot
        u_masked = flow[..., 0].copy()
        v_masked = flow[..., 1].copy()
        u_masked[mag < threshold] = 0
        v_masked[mag < threshold] = 0

        # 4. Generate 6-Panel Comparison Plot
        fig, axs = plt.subplots(3, 2, figsize=(18, 18))
        step = 20 # Subsampling for quiver plot
        y, x = np.mgrid[step/2:img1.shape[0]:step, step/2:img1.shape[1]:step].reshape(2,-1).astype(int)

        # ROW 1: RAW DATA & CRAZY VECTORS
        im1 = axs[0, 0].imshow(mag, cmap='hot')
        axs[0, 0].set_title(f'Raw Magnitude (All) - {pos_name} Frame {i}')
        plt.colorbar(im1, ax=axs[0, 0])

        fx_raw, fy_raw = flow[y, x].T
        axs[0, 1].imshow(img1, cmap='gray')
        axs[0, 1].quiver(x, y, fx_raw, -fy_raw, color='cyan', scale=10)
        axs[0, 1].set_title('Vector Field: ALL (Includes "Crazy" Noise)')

        # ROW 2: COMPONENTS & CLEANED SIGNAL
        im3 = axs[1, 0].imshow(flow[..., 0], cmap='RdBu')
        axs[1, 0].set_title('Horizontal Flow (U)')
        plt.colorbar(im3, ax=axs[1, 0])

        fx_clean, fy_clean = u_masked[y, x], v_masked[y, x]
        axs[1, 1].imshow(img1, cmap='gray')
        axs[1, 1].quiver(x, y, fx_clean, -fy_clean, color='lime', scale=10)
        axs[1, 1].set_title('Vector Field: 80th Percentile (Cleaned Signal)')

        # ROW 3: STAGE DRIFT CHECK & THRESHOLD MASK
        im5 = axs[2, 0].imshow(flow[..., 1], cmap='PRGn')
        axs[2, 0].set_title('Vertical Flow (V) - Stage Drift Check')
        plt.colorbar(im5, ax=axs[2, 0])

        binary_mask = (mag >= threshold).astype(float)
        axs[2, 1].imshow(binary_mask, cmap='binary')
        axs[2, 1].set_title(f'Extraction Mask (Top 20% Motion - Threshold: {threshold:.2f})')

        # Save Result
        save_path = output_dir / f"compare_{condition}_{pos_name}_frame{i:03d}.png"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

def main():
    base_path = Path('.')
    output_dir = base_path / 'results_viz_comparison'
    output_dir.mkdir(exist_ok=True)
    conditions = {'REF': 'REF_raw_data101_110', 'RIF': 'RIF10_raw_data201_210'}

    for cond, folder in conditions.items():
        data_root = base_path / 'data' / folder
        if not data_root.exists(): continue
        for pos_path in sorted([p for p in data_root.glob('Pos*') if p.is_dir()]):
            process_position(pos_path, cond, output_dir)
    print(f"Comparison visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()