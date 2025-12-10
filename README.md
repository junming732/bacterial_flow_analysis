# Research Methodology: Bacterial Growth Optical Flow Analysis

## 1. Overview
This project analyzes the **Net Optical Flow** of bacterial colony growth. It compares untreated bacterial samples (REF) against those treated with Rifampicin (RIF) to detect growth anomalies and points of divergence.

The repository contains two methods:
1.  **`analysis_comparison.py`**: A quantitative script that plots the net flow direction over time and highlights statistical divergence.
2.  **`vector_viz.py`**: A qualitative visualization script that generates heatmap videos/images with vector arrows to show flow dynamics.

## 2. Setup Instructions

### Dependencies
Install the required Python libraries:
```bash
pip install -r requirements.txt
```
### Data Preparation
To reproduce the results, the raw data has to be put in specific folders.
1. Create a folder named **data**.
2. Inside **data**, place the specific raw data folders:

* `REF_raw_data101_110`
* `RIF10_raw_data201_210`

### Directory Structure:

```text
.
├── analysis_comparison.py     # Quantitative analysis script
├── vector_viz.py              # Qualitative visualization script
├── requirements.txt           # Dependencies
└── data/
    ├── REF_raw_data101_110/   # Untreated images
    └── RIF10_raw_data201_210/ # Treated images
```

## 3. How to Run
### Method 1: Quantitative Analysis
Run the comparison script to generate time-series plots:

```bash
python analysis_comparison.py
```
Output: Generates results_comparison.png, showing the net flow difference between REF and RIF and marking the exact frame where behavior diverges.

### Method 2: Qualitative Visualization
Run the visualization script to generate vector fields:

```bash
python vector_viz.py
```
Output: Creates a results_viz/ folder containing heatmaps (e.g., viz_RIF_frame030.png) showing the direction and magnitude of bacterial movement at specific timepoints.