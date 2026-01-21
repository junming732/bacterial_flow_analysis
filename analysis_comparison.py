import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def process_position(pos_path, condition, output_dir):
    """Generates a comparison of Raw vs. Outlier-Rejected (Double-Threshold) vectors."""
    pos_name = pos_path.name
    aphase_path = pos_path / 'aphase'
    img_files = sorted(list(aphase_path.glob('*.tiff')) + list(aphase_path.glob('*.tif'))) if aphase_path.exists() else sorted(list(pos_path.glob('*.tiff')) + list(pos_path.glob('*.tif')))

    if len(img_files) < 2: return

    print(f"  Refining Analysis for {condition} {pos_name}...")

    for i in range(0, len(img_files) - 1, 10):
        img1 = cv2.imread(str(img_files[i]), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(img_files[i+1]), cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None: continue

        # 1. FARNEBACK WITH STABILIZED WINDOW
        # Winsize increased to 31 to act as a low-pass filter against flicker/noise
        flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 31, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # 2. DOUBLE-THRESHOLD LOGIC (Heuristic Outlier Rejection)
        # We remove the "Bottom 50%" (noise) and "Top 5%" (Crazy Hallucinations)
        valid_mag = mag[mag > 0.001]
        if valid_mag.size > 0:
            lower_bound = np.percentile(valid_mag, 50)
            upper_bound = np.percentile(valid_mag, 95)
            # Physical safety cap: nothing should move > 10 pixels in this setup
            upper_bound = min(upper_bound, 10.0)
        else:
            lower_bound, upper_bound = 0, 10

        mask = (mag >= lower_bound) & (mag <= upper_bound)

        # Apply mask
        u_raw, v_raw = flow[..., 0], flow[..., 1]
        u_clean = np.where(mask, u_raw, 0)
        v_clean = np.where(mask, v_raw, 0)

        # 3. SIX-PANEL DIAGNOSTIC PLOT
        fig, axs = plt.subplots(3, 2, figsize=(20, 24))
        step = 25
        y, x = np.mgrid[step/2:img1.shape[0]:step, step/2:img1.shape[1]:step].reshape(2,-1).astype(int)

        # Panel 1: Original Image
        axs[0, 0].imshow(img1, cmap='gray')
        axs[0, 0].set_title(f"Original Frame: {pos_name} ({i})")

        # Panel 2: RAW Vector Field (Cyan - The 'Crazy' Vectors)
        axs[0, 1].imshow(img1, cmap='gray')
        axs[0, 1].quiver(x, y, u_raw[y, x], -v_raw[y, x], color='cyan', scale=15)
        axs[0, 1].set_title("RAW Flow (Includes Outliers/Hallucinations)")

        # Panel 3: Horizontal Component (The 'Dying' Indicator)
        im3 = axs[1, 0].imshow(u_raw, cmap='RdBu', vmin=-1, vmax=1)
        axs[1, 0].set_title("Horizontal Flow (U): Red=Right(Dying) / Blue=Left(Growth)")
        plt.colorbar(im3, ax=axs[1, 0])

        # Panel 4: CLEANED Vector Field (Lime - The Signal)
        axs[1, 1].imshow(img1, cmap='gray')
        axs[1, 1].quiver(x, y, u_clean[y, x], -v_clean[y, x], color='lime', scale=15)
        axs[1, 1].set_title(f"CLEANED Flow (Outliers Removed)\nRange: [{lower_bound:.2f}, {upper_bound:.2f}]")

        # Panel 5: Vertical Flow (The Drift Check)
        im5 = axs[2, 0].imshow(v_raw, cmap='PRGn', vmin=-0.5, vmax=0.5)
        axs[2, 0].set_title("Vertical Flow (V): Should be neutral (No Stage Drift)")
        plt.colorbar(im5, ax=axs[2, 0])

        # Panel 6: The Rejection Mask
        axs[2, 1].imshow(mask, cmap='binary_r')
        axs[2, 1].set_title("Rejection Mask (White = Valid Biomass Motion)")

        plt.savefig(output_dir / f"final_diag_{condition}_{pos_name}_{i:03d}.png", bbox_inches='tight')
        plt.close()

def main():
    out = Path('results_final_diagnostics'); out.mkdir(exist_ok=True)
    paths = {'REF': 'data/REF_raw_data101_110', 'RIF': 'data/RIF10_raw_data201_210'}
    for cond, folder in paths.items():
        data = Path(folder)
        if not data.exists(): continue
        for pos in sorted([p for p in data.glob('Pos*') if p.is_dir()]):
            process_position(pos, cond, out)

if __name__ == "__main__": main()