#!/usr/bin/env python3

import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

data_path = Path('/home/junming/nobackup_junming')
output_path = Path('/home/junming/private/rearch_methodology')

def load_ground_truth(condition, position):
    """Load ground truth growth data from pickle files."""
    if condition == 'REF':
        mask_dir = data_path / 'REF_masks101_110' / f'Pos{position}' / 'PreprocessedPhaseMasks'
    else:
        mask_dir = data_path / 'RIF10_masks201_210' / f'Pos{position}' / 'PreprocessedPhaseMasks'

    growth_rate_file = mask_dir / 'growth_rate.pickle'
    growth_areas_file = mask_dir / 'growth_areas.pickle'

    growth_rates = None
    growth_areas = None

    if growth_rate_file.exists():
        with open(growth_rate_file, 'rb') as f:
            growth_rates = pickle.load(f)

    if growth_areas_file.exists():
        with open(growth_areas_file, 'rb') as f:
            growth_areas = pickle.load(f)

    return growth_rates, growth_areas

print("Loading ground truth data...")
print("="*60)

all_data = []

for condition, positions in [('REF', range(101, 111)), ('RIF10', range(201, 211))]:
    for pos in positions:
        growth_rates, growth_areas = load_ground_truth(condition, pos)

        if growth_rates is not None and growth_areas is not None:
            print(f"{condition} Pos{pos}:")
            print(f"  Growth rates: {len(growth_rates)} values")
            print(f"  Mean growth rate: {np.mean(growth_rates):.4f}")
            print(f"  Growth areas: {len(growth_areas)} values")
            print(f"  Mean area: {np.mean(growth_areas):.1f}")

            for t, (rate, area) in enumerate(zip(growth_rates[:30], growth_areas[:30])):
                all_data.append({
                    'condition': condition,
                    'position': pos,
                    'time': t,
                    'growth_rate': rate,
                    'area': area
                })

df_truth = pd.DataFrame(all_data)

print("\n" + "="*60)
print("Summary by condition:")
summary = df_truth.groupby('condition').agg({
    'growth_rate': ['mean', 'std', 'count'],
    'area': ['mean', 'std']
})
print(summary)

# Load optical flow data
df_flow = pd.read_csv(output_path / 'flow_analysis_direct.csv')

# Merge
df_merged = pd.merge(df_flow, df_truth, on=['condition', 'position', 'time'], how='inner')

print(f"\nMerged {len(df_merged)} matching datapoints")

# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Growth rate by condition
ax = axes[0, 0]
for condition in ['REF', 'RIF10']:
    data = df_truth[df_truth['condition'] == condition]
    grouped = data.groupby('time')['growth_rate'].agg(['mean', 'std'])
    ax.plot(grouped.index, grouped['mean'], label=condition, linewidth=2)
    ax.fill_between(grouped.index,
                     grouped['mean'] - grouped['std'],
                     grouped['mean'] + grouped['std'],
                     alpha=0.3)
ax.set_xlabel('Time point')
ax.set_ylabel('Ground Truth Growth Rate')
ax.set_title('Ground Truth: Growth Rate Over Time')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Optical flow by condition
ax = axes[0, 1]
for condition in ['REF', 'RIF10']:
    data = df_flow[df_flow['condition'] == condition]
    grouped = data.groupby('time')['mean_flow'].agg(['mean', 'std'])
    ax.plot(grouped.index, grouped['mean'], label=condition, linewidth=2)
    ax.fill_between(grouped.index,
                     grouped['mean'] - grouped['std'],
                     grouped['mean'] + grouped['std'],
                     alpha=0.3)
ax.set_xlabel('Time point')
ax.set_ylabel('Optical Flow (pixels/frame)')
ax.set_title('Optical Flow: Mean Speed')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Correlation - optical flow vs growth rate
ax = axes[1, 0]
for condition, color in [('REF', 'blue'), ('RIF10', 'orange')]:
    data = df_merged[df_merged['condition'] == condition]
    ax.scatter(data['growth_rate'], data['mean_flow'], alpha=0.5, s=20, label=condition, color=color)
ax.set_xlabel('Ground Truth Growth Rate')
ax.set_ylabel('Optical Flow')
ax.set_title('Correlation: Optical Flow vs Growth Rate')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Area over time
ax = axes[1, 1]
for condition in ['REF', 'RIF10']:
    data = df_truth[df_truth['condition'] == condition]
    grouped = data.groupby('time')['area'].agg(['mean', 'std'])
    ax.plot(grouped.index, grouped['mean'], label=condition, linewidth=2)
    ax.fill_between(grouped.index,
                     grouped['mean'] - grouped['std'],
                     grouped['mean'] + grouped['std'],
                     alpha=0.3)
ax.set_xlabel('Time point')
ax.set_ylabel('Colony Area (pixels)')
ax.set_title('Ground Truth: Colony Area Over Time')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_path / 'ground_truth_comparison.png', dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {output_path / 'ground_truth_comparison.png'}")

# Calculate correlations
print("\nCorrelations between optical flow and growth rate:")
for condition in ['REF', 'RIF10']:
    data = df_merged[df_merged['condition'] == condition]
    corr = np.corrcoef(data['growth_rate'], data['mean_flow'])[0, 1]
    print(f"  {condition}: r = {corr:.3f}")

plt.close()

print("\nDone!")