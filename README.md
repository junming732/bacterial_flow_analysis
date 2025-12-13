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

## 4. Expected Results & Interpretation

The results from this method are subtle. The antibiotic effect does not cause the bacteria to immediately disappear; rather, it biases their movement direction.

### A. Quantitative Analysis (`results_comparison.png`)
This image contains 5 panels. Here is how to read them:

1.  **Horizontal Flow (Top Panel):**
    * **What it shows:** The raw movement trajectories of Untreated (REF, Blue) vs. Treated (RIF, Red).
    * **Interpretation:** You may see the lines overlapping significantly. The signals of separation here are **not always clear** to the naked eye because both colonies are growing.

2.  **Vertical Flow (Second Panel):**
    * **What it shows:** Up/Down movement.
    * **Interpretation:** Generally serves as a control. Both conditions typically oscillate around 0 (no significant vertical bias), confirming that the chamber setup is level.

3.  **Flow Magnitude (Third Panel):**
    * **What it shows:** The overall "intensity" of movement, regardless of direction.
    * **Interpretation:** Treated bacteria often show higher magnitude spikes due to stress responses, but this signal can be noisy.

4.  **Difference Plot (Fourth Panel - **Most Important**):**
    * **What it shows:** The mathematical difference: `RIF (Treated) - REF (Untreated)`.
    * **Interpretation:** This is where the signal becomes visible. You will likely see **positive values**, indicating that the RIF-treated bacteria have a persistent **rightward inclination** compared to the control. Even if the signal is weak, a consistent positive trend indicates the "declining" or dying phase where biomass drifts or expands asymmetrically.

5.  **Statistical Summary (Bottom Right):**
    * Provides the global averages. If `RIF10 Avg Horizontal` is higher than REF, it confirms the rightward directional bias quantitatively.

### B. Qualitative Visualization (`results_viz/`)
* **Files:** Images like `viz_RIF_frame030.png`.
* **Interpretation:** Do not look for a difference in the "size" or "area" of the colony, as both grow initially.
* **What to look for:** Look at the **arrows (Vectors)**.
    * **REF (Untreated):** Arrows point outward in all directions (healthy symmetric expansion).
    * **RIF (Treated):** You may see a tendency for arrows to align **rightward** (yellow/red arrows). This directional bias is the visual signature of the antibiotic treatment taking effect.