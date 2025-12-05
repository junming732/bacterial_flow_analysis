#!/usr/bin/env python3

import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

data_path = Path('/home/junming/nobackup_junming')
output_path = Path('/home/junming/private/rearch_methodology')

def test_image_loading():
    """Check if images load correctly."""
    print("Testing image loading...")

    raw_dir = data_path / 'REF_raw_data101_110' / 'Pos101' / 'aphase'
    mask_dir = data_path / 'REF_masks101_110' / 'Pos101' / 'PreprocessedPhaseMasks'

    print(f"Raw directory: {raw_dir}")
    print(f"Raw directory exists: {raw_dir.exists()}")

    if raw_dir.exists():
        image_files = sorted(raw_dir.glob('img_*.tiff'))
        print(f"Found {len(image_files)} raw images")

        if len(image_files) > 0:
            img = cv2.imread(str(image_files[0]), cv2.IMREAD_GRAYSCALE)
            print(f"First image shape: {img.shape if img is not None else 'Failed to load'}")
            print(f"First image dtype: {img.dtype if img is not None else 'N/A'}")
            print(f"First image range: [{img.min()}, {img.max()}]" if img is not None else "N/A")

            if img is not None and len(image_files) > 1:
                img2 = cv2.imread(str(image_files[1]), cv2.IMREAD_GRAYSCALE)
                print(f"Second image loaded: {img2 is not None}")
                if img2 is not None:
                    diff = np.abs(img.astype(float) - img2.astype(float))
                    print(f"Difference between frame 0 and 1: mean={diff.mean():.2f}, max={diff.max():.2f}")

    print(f"\nMask directory: {mask_dir}")
    print(f"Mask directory exists: {mask_dir.exists()}")

    if mask_dir.exists():
        mask_files = sorted(mask_dir.glob('MASK_img_*.tif'))
        print(f"Found {len(mask_files)} mask images")

        if len(mask_files) > 0:
            mask = cv2.imread(str(mask_files[0]), cv2.IMREAD_GRAYSCALE)
            print(f"First mask shape: {mask.shape if mask is not None else 'Failed to load'}")
            print(f"First mask unique values: {np.unique(mask) if mask is not None else 'N/A'}")
            print(f"First mask range: [{mask.min()}, {mask.max()}]" if mask is not None else "N/A")

def test_optical_flow():
    """Test optical flow calculation on actual data."""
    print("\n" + "="*50)
    print("Testing optical flow calculation...")

    raw_dir = data_path / 'REF_raw_data101_110' / 'Pos101' / 'aphase'
    image_files = sorted(raw_dir.glob('img_*.tiff'))

    if len(image_files) < 2:
        print("Not enough images for flow calculation")
        return

    img1 = cv2.imread(str(image_files[0]), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(str(image_files[1]), cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print("Failed to load images")
        return

    print(f"Image 1 shape: {img1.shape}, range: [{img1.min()}, {img1.max()}]")
    print(f"Image 2 shape: {img2.shape}, range: [{img2.min()}, {img2.max()}]")

    flow_params = {
        'pyr_scale': 0.5,
        'levels': 3,
        'winsize': 15,
        'iterations': 3,
        'poly_n': 5,
        'poly_sigma': 1.2,
        'flags': 0
    }

    flow = cv2.calcOpticalFlowFarneback(img1, img2, None, **flow_params)

    print(f"Flow shape: {flow.shape}")
    print(f"Flow u range: [{flow[:,:,0].min():.4f}, {flow[:,:,0].max():.4f}]")
    print(f"Flow v range: [{flow[:,:,1].min():.4f}, {flow[:,:,1].max():.4f}]")

    magnitude = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
    print(f"Flow magnitude: mean={magnitude.mean():.4f}, max={magnitude.max():.4f}, median={np.median(magnitude):.4f}")
    print(f"Non-zero flow pixels: {np.sum(magnitude > 0.1)}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(img1, cmap='gray')
    axes[0, 0].set_title('Frame 0')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img2, cmap='gray')
    axes[0, 1].set_title('Frame 1')
    axes[0, 1].axis('off')

    diff = np.abs(img2.astype(float) - img1.astype(float))
    im = axes[0, 2].imshow(diff, cmap='hot')
    axes[0, 2].set_title(f'Absolute Difference\nMean: {diff.mean():.2f}')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)

    im = axes[1, 0].imshow(flow[:,:,0], cmap='RdBu')
    axes[1, 0].set_title('Flow U (horizontal)')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

    im = axes[1, 1].imshow(flow[:,:,1], cmap='RdBu')
    axes[1, 1].set_title('Flow V (vertical)')
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

    im = axes[1, 2].imshow(magnitude, cmap='hot')
    axes[1, 2].set_title(f'Flow Magnitude\nMax: {magnitude.max():.2f}')
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046)

    plt.tight_layout()
    plt.savefig(output_path / 'flow_diagnostic.png', dpi=200, bbox_inches='tight')
    print(f"\nDiagnostic plot saved to: {output_path / 'flow_diagnostic.png'}")
    plt.close()

def test_mask_and_boundary():
    """Test mask loading and boundary extraction."""
    print("\n" + "="*50)
    print("Testing mask and boundary extraction...")

    mask_dir = data_path / 'REF_masks101_110' / 'Pos101' / 'PreprocessedPhaseMasks'
    mask_files = sorted(mask_dir.glob('MASK_img_*.tif'))

    if len(mask_files) < 1:
        print("No mask files found")
        return

    mask = cv2.imread(str(mask_files[0]), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print("Failed to load mask")
        return

    print(f"Mask shape: {mask.shape}")
    print(f"Mask unique values: {np.unique(mask)}")
    print(f"Mask non-zero pixels: {np.sum(mask > 0)}")

    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)
    boundary = mask - eroded

    print(f"Boundary pixels: {np.sum(boundary > 0)}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(mask, cmap='gray')
    axes[0].set_title(f'Mask\nNon-zero: {np.sum(mask > 0)}')
    axes[0].axis('off')

    axes[1].imshow(eroded, cmap='gray')
    axes[1].set_title('Eroded mask')
    axes[1].axis('off')

    axes[2].imshow(boundary, cmap='hot')
    axes[2].set_title(f'Boundary\nPixels: {np.sum(boundary > 0)}')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path / 'mask_diagnostic.png', dpi=200, bbox_inches='tight')
    print(f"Mask diagnostic saved to: {output_path / 'mask_diagnostic.png'}")
    plt.close()

def test_boundary_flow():
    """Test flow calculation at boundary pixels."""
    print("\n" + "="*50)
    print("Testing boundary flow extraction...")

    raw_dir = data_path / 'REF_raw_data101_110' / 'Pos101' / 'aphase'
    mask_dir = data_path / 'REF_masks101_110' / 'Pos101' / 'PreprocessedPhaseMasks'

    image_files = sorted(raw_dir.glob('img_*.tiff'))
    mask_files = sorted(mask_dir.glob('MASK_img_*.tif'))

    if len(image_files) < 2 or len(mask_files) < 1:
        print("Not enough files")
        return

    img1 = cv2.imread(str(image_files[0]), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(str(image_files[1]), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_files[0]), cv2.IMREAD_GRAYSCALE)

    flow_params = {
        'pyr_scale': 0.5,
        'levels': 3,
        'winsize': 15,
        'iterations': 3,
        'poly_n': 5,
        'poly_sigma': 1.2,
        'flags': 0
    }

    flow = cv2.calcOpticalFlowFarneback(img1, img2, None, **flow_params)
    magnitude = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)

    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)
    boundary = (mask - eroded) > 0

    print(f"Boundary pixels: {np.sum(boundary)}")
    print(f"Flow at boundary - mean: {magnitude[boundary].mean():.4f}, max: {magnitude[boundary].max():.4f}")
    print(f"Flow overall - mean: {magnitude.mean():.4f}, max: {magnitude.max():.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im = axes[0].imshow(magnitude, cmap='hot')
    axes[0].contour(boundary, colors='cyan', linewidths=2)
    axes[0].set_title('Flow magnitude with boundary')
    axes[0].axis('off')
    plt.colorbar(im, ax=axes[0], fraction=0.046)

    masked_flow = magnitude.copy()
    masked_flow[~boundary] = 0
    im = axes[1].imshow(masked_flow, cmap='hot')
    axes[1].set_title(f'Flow at boundary only\nMean: {magnitude[boundary].mean():.4f}')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    plt.tight_layout()
    plt.savefig(output_path / 'boundary_flow_diagnostic.png', dpi=200, bbox_inches='tight')
    print(f"Boundary flow diagnostic saved to: {output_path / 'boundary_flow_diagnostic.png'}")
    plt.close()

if __name__ == '__main__':
    print("OPTICAL FLOW DIAGNOSTIC")
    print("="*50)

    test_image_loading()
    test_optical_flow()
    test_mask_and_boundary()
    test_boundary_flow()

    print("\n" + "="*50)
    print("Diagnostic complete. Check output images.")