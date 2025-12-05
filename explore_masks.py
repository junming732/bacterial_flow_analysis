#!/usr/bin/env python3

from pathlib import Path
import cv2
import numpy as np

data_path = Path('/home/junming/nobackup_junming')

print("Exploring mask directory structure...")
print("="*50)

mask_base = data_path / 'REF_masks101_110' / 'Pos101'
print(f"Base mask directory: {mask_base}")
print(f"Contents:")

for item in sorted(mask_base.iterdir()):
    print(f"  {item.name}")
    if item.is_dir():
        files = list(item.glob('*'))
        print(f"    Contains {len(files)} files")
        if len(files) > 0:
            print(f"    First few files:")
            for f in sorted(files)[:5]:
                print(f"      {f.name}")
                if f.suffix in ['.tif', '.tiff', '.png']:
                    img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        print(f"        Shape: {img.shape}, unique values: {np.unique(img)[:10]}, non-zero: {np.sum(img > 0)}")

print("\n" + "="*50)
print("Checking PreprocessedPhase directory...")
preprocessed = mask_base / 'PreprocessedPhase'
if preprocessed.exists():
    files = list(preprocessed.glob('*'))
    print(f"Files in PreprocessedPhase: {len(files)}")
    for f in sorted(files)[:10]:
        print(f"  {f.name}")
        if f.suffix in ['.tif', '.tiff', '.png']:
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                print(f"    Shape: {img.shape}, unique values: {np.unique(img)[:10]}, non-zero: {np.sum(img > 0)}")

print("\n" + "="*50)
print("Checking pickle files...")
pickle_files = list(mask_base.glob('*/*.pickle'))
print(f"Found {len(pickle_files)} pickle files:")
for p in pickle_files:
    print(f"  {p}")