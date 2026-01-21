import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def process_position(pos_path, condition, output_dir):
    pos_name = pos_path.name
    # Find images in 'aphase' or root
    aphase_path = pos_path / 'aphase'
    img_files = sorted(list(aphase_path.glob('*.tiff')) + list(aphase_path.glob('*.tif'))) if aphase_path.exists() else sorted(list(pos_path.glob('*.tiff')) + list(pos_path.glob('*.tif')))

    if len(img_files) < 2:
        print(f"      [Skipping] {pos_name}: Not enough frames.")
        return

    print(f"  --> Analyzing {condition} {pos_name}...")

    for i in range(0, len(img_files) - 1, 10):
        img1_raw = cv2.imread(str(img_files[i]), 0)
        img2_raw = cv2.imread(str(img_files[i+1]), 0)
        if img1_raw is None or img2_raw is None: continue

        # 1. OPTICAL FLOW (Farneback with stabilized window)
        flow = cv2.calcOpticalFlowFarneback(img1_raw, img2_raw, None, 0.5, 3, 31, 3, 5, 1.2, 0)
        u_raw, v_raw = flow[..., 0], flow[..., 1]
        mag = np.sqrt(u_raw**2 + v_raw**2)

        # 2. DOUBLE THRESHOLD (Keep 30th to 98th percentile)
        valid_mags = mag[mag > 0.0001]
        if valid_mags.size > 0:
            low, high = np.percentile(valid_mags, 50), np.percentile(valid_mags, 95)
            mask = (mag >= low) & (mag <= high)
        else:
            mask = np.zeros_like(mag, dtype=bool)

        u_clean = np.where(mask, u_raw, 0)
        v_clean = np.where(mask, v_raw, 0)

        # 3. THREE-PANEL VISUALIZATION (Source, Target, Vectors)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
        plt.subplots_adjust(wspace=0.1)

        # Metadata in titles
        info_str = f"{condition} | {pos_name} | Frame {i}"

        # Panel 1: Frame N
        ax1.imshow(img1_raw, cmap='gray')
        ax1.set_title(f"{info_str}\nSource (Frame N)", fontsize=12)
        ax1.axis('off')

        # Panel 2: Frame N+1
        ax2.imshow(img2_raw, cmap='gray')
        ax2.set_title(f"Target (Frame N+1)", fontsize=12)
        ax2.axis('off')

        # Panel 3: BIG VECTOR OVERLAY
        step = 25
        y, x = np.mgrid[step/2:img1_raw.shape[0]:step, step/2:img1_raw.shape[1]:step].reshape(2,-1).astype(int)
        u_p, v_p = u_clean[y, x], v_clean[y, x]

        # Strict Directional Coloring: Blue = Leftward (-) | Red = Rightward (+)
        arrow_colors = np.where(u_p > 0, '#e31a1c', '#2171b5')

        ax3.imshow(img1_raw, cmap='gray')
        # scale=0.03 makes the arrows MASSIVE and impossible to miss
        ax3.quiver(x, y, u_p, v_p, color=arrow_colors,
                   angles='xy', scale_units='xy', scale=0.03, width=0.005)
        ax3.set_title(f"Vector Field Analysis\nBlue: Growth (L) | Red: Retreat (R)", fontsize=12)
        ax3.axis('off')

        # Save with full naming info
        save_filename = f"{condition}_{pos_name}_frame_{i:03d}.png"
        plt.savefig(output_dir / save_filename, bbox_inches='tight', dpi=120)
        plt.close()
        print(f"      Saved: {save_filename}")

def main():
    base = Path('.'); out = base / 'final_validation_plots'
    out.mkdir(exist_ok=True)

    # Path mapping
    conditions = {
        'REF': 'REF_raw_data101_110',
        'RIF': 'RIF10_raw_data201_210'
    }

    print("\n" + "="*60)
    print("STARTING 3-PANEL VALIDATION (REF vs RIF)")
    print("="*60)

    for cond, folder in conditions.items():
        data_root = base / 'data' / folder
        if not data_root.exists():
            print(f"!! Error: {folder} not found.")
            continue

        # Find all Pos folders
        pos_folders = sorted([p for p in data_root.glob('Pos*') if p.is_dir()])
        for pos in pos_folders:
            process_position(pos, cond, out)

    print("\n" + "="*60)
    print(f"FINISH: Check output in {out.absolute()}")
    print("="*60)

if __name__ == "__main__":
    main()