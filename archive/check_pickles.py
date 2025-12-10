#!/usr/bin/env python3

from pathlib import Path
import pickle
import numpy as np

data_path = Path('/home/junming/nobackup_junming')
mask_dir = data_path / 'REF_masks101_110' / 'Pos101' / 'PreprocessedPhaseMasks'

print("Checking pickle files...")
print("="*50)

growth_areas_file = mask_dir / 'growth_areas.pickle'
if growth_areas_file.exists():
    print(f"\nLoading {growth_areas_file.name}...")
    with open(growth_areas_file, 'rb') as f:
        growth_areas = pickle.load(f)

    print(f"Type: {type(growth_areas)}")
    if isinstance(growth_areas, dict):
        print(f"Keys: {list(growth_areas.keys())}")
        for key, value in list(growth_areas.items())[:3]:
            print(f"  {key}: {type(value)}, shape/len: {np.array(value).shape if hasattr(value, '__len__') else 'scalar'}")
    elif isinstance(growth_areas, (list, np.ndarray)):
        print(f"Length: {len(growth_areas)}")
        print(f"First few values: {growth_areas[:5]}")
    else:
        print(f"Content: {growth_areas}")

growth_rate_file = mask_dir / 'growth_rate.pickle'
if growth_rate_file.exists():
    print(f"\nLoading {growth_rate_file.name}...")
    with open(growth_rate_file, 'rb') as f:
        growth_rate = pickle.load(f)

    print(f"Type: {type(growth_rate)}")
    if isinstance(growth_rate, dict):
        print(f"Keys: {list(growth_rate.keys())}")
        for key, value in list(growth_rate.items())[:3]:
            print(f"  {key}: {type(value)}, shape/len: {np.array(value).shape if hasattr(value, '__len__') else 'scalar'}")
    elif isinstance(growth_rate, (list, np.ndarray)):
        print(f"Length: {len(growth_rate)}")
        print(f"First few values: {growth_rate[:5]}")
    else:
        print(f"Content: {growth_rate}")

print("\n" + "="*50)
print("Since masks are empty, we should analyze flow on entire colony region")
print("or use intensity thresholding to find colony boundaries")