import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def process_position(pos_path, condition, output_dir):
    pos_name = pos_path.name
    aphase_path = pos_path / 'aphase'
    img_files = sorted(list(aphase_path.glob('*.tiff')) + list(aphase_path.glob('*.tif'))) if aphase_path.exists() else sorted(list(pos_path.glob('*.tiff')) + list(pos_path.glob('*.tif')))

    if len(img_files) < 35:
        print(f"      [Skipping] {pos_name}: Not enough frames for shock analysis.")
        return []

    print(f"  --> Analyzing Shock Window for {pos_name}...")

    pos_movement_scores = []

    # Analyze every frame between 25 and 35
    for i in range(25, 35):
        img1_raw = cv2.imread(str(img_files[i]), 0)
        img2_raw = cv2.imread(str(img_files[i+1]), 0)
        if img1_raw is None or img2_raw is None: continue

        # 1. OPTICAL FLOW
        flow = cv2.calcOpticalFlowFarneback(img1_raw, img2_raw, None, 0.5, 3, 31, 3, 5, 1.2, 0)
        u_raw, v_raw = flow[..., 0], flow[..., 1]
        mag = np.sqrt(u_raw**2 + v_raw**2)

        # Record activity score for global plot (mean of top 50% movement)
        activity = np.mean(mag[mag > np.percentile(mag, 50)])
        pos_movement_scores.append(activity)

        # 2. THRESHOLDING
        valid_mags = mag[mag > 0.0001]
        if valid_mags.size > 0:
            low, high = np.percentile(valid_mags, 50), np.percentile(valid_mags, 95)
            mask = (mag >= low) & (mag <= high)
        else:
            mask = np.zeros_like(mag, dtype=bool)

        u_clean = np.where(mask, u_raw, 0)
        v_clean = np.where(mask, v_raw, 0)

        # 3. VISUALIZATION
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        plt.subplots_adjust(wspace=0.1)

        ax1.imshow(img1_raw, cmap='gray'); ax1.set_title(f"Frame {i}"); ax1.axis('off')
        ax2.imshow(img2_raw, cmap='gray'); ax2.set_title(f"Frame {i+1}"); ax2.axis('off')

        step = 25
        y, x = np.mgrid[step/2:img1_raw.shape[0]:step, step/2:img1_raw.shape[1]:step].reshape(2,-1).astype(int)
        u_p, v_p = u_clean[y, x], v_clean[y, x]
        arrow_colors = np.where(u_p > 0, '#e31a1c', '#2171b5')

        ax3.imshow(img1_raw, cmap='gray')
        ax3.quiver(x, y, u_p, v_p, color=arrow_colors, angles='xy', scale_units='xy', scale=0.03, width=0.005)
        ax3.set_title("Vector Analysis (Shock Window)"); ax3.axis('off')

        save_filename = f"SHOCK_{pos_name}_frame_{i:03d}.png"
        plt.savefig(output_dir / save_filename, bbox_inches='tight', dpi=100)
        plt.close()

    return pos_movement_scores

def main():
    base = Path('.'); out = base / 'shock_analysis_results'
    out.mkdir(exist_ok=True)

    # Focus ONLY on RIF folder
    rif_folder = base / 'data' / 'RIF10_raw_data201_210'
    pos_folders = sorted([p for p in rif_folder.glob('Pos*') if p.is_dir()])

    global_data = {}

    for pos in pos_folders:
        scores = process_position(pos, "RIF", out)
        if scores:
            global_data[pos.name] = scores

    # --- GENERATE FULL PICTURE SUMMARY ---
    plt.figure(figsize=(10, 6))
    for pos_name, scores in global_data.items():
        plt.plot(range(25, 35), scores, alpha=0.3, color='gray') # Individual positions

    # Calculate and plot the average of all positions
    all_scores = np.array(list(global_data.values()))
    mean_scores = np.mean(all_scores, axis=0)
    plt.plot(range(25, 35), mean_scores, color='red', linewidth=3, label='Global Average')

    plt.title('RIF Shock Event Detection (Frames 25-35)', fontsize=14)
    plt.xlabel('Frame Number')
    plt.ylabel('Vector Magnitude (Movement Intensity)')
    plt.axvline(x=30, color='black', linestyle='--')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(out / "GLOBAL_RIF_SHOCK_SUMMARY.png", dpi=150)
    print(f"\nAnalysis complete. Check {out.absolute()} for the summary plot.")

if __name__ == "__main__":
    main()