#!/usr/bin/env python3

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage

data_path = Path('/home/junming/nobackup_junming')
output_path = Path('/home/junming/private/rearch_methodology')

def test_colony_detection(condition, position, frame=10):
    """Visualize colony detection at different threshold levels."""

    if condition == 'REF':
        raw_dir = data_path / 'REF_raw_data101_110' / f'Pos{position}' / 'aphase'
    else:
        raw_dir = data_path / 'RIF10_raw_data201_210' / f'Pos{position}' / 'aphase'

    image_files = sorted(raw_dir.glob('img_*.tiff'))
    img = cv2.imread(str(image_files[frame]), cv2.IMREAD_GRAYSCALE)

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title(f'{condition} Pos{position} Frame {frame}\nOriginal')
    axes[0, 0].axis('off')

    axes[0, 1].hist(img.ravel(), bins=100, color='blue', alpha=0.7)
    axes[0, 1].set_title('Intensity Histogram')
    axes[0, 1].set_xlabel('Intensity')
    axes[0, 1].set_ylabel('Frequency')

    percentiles = [10, 20, 30, 40, 50, 70]

    for idx, percentile in enumerate(percentiles):
        row = (idx + 2) // 3
        col = (idx + 2) % 3

        threshold = np.percentile(img, percentile)
        mask = img < threshold

        mask = ndimage.binary_opening(mask, iterations=2)
        mask = ndimage.binary_closing(mask, iterations=2)

        labeled, num_features = ndimage.label(mask)
        if num_features > 0:
            sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
            largest_label = np.argmax(sizes) + 1
            mask = labeled == largest_label

        overlay = img.copy()
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
        overlay[mask] = [255, 0, 0]

        axes[row, col].imshow(overlay)
        axes[row, col].set_title(f'Threshold: {percentile}th percentile\nPixels: {np.sum(mask)}')
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(output_path / f'colony_detection_{condition}_Pos{position}.png', dpi=200, bbox_inches='tight')
    print(f"Saved: colony_detection_{condition}_Pos{position}.png")
    plt.close()

print("Testing colony detection on different samples...")
test_colony_detection('REF', 101, frame=10)
test_colony_detection('RIF10', 201, frame=10)
test_colony_detection('REF', 101, frame=20)
test_colony_detection('RIF10', 201, frame=20)

print("\nCheck the output images to see which threshold works best")
print("Then we can adjust the detection algorithm")