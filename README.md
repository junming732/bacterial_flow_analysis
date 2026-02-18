# Research Methodology: Bacterial Growth Optical Flow Analysis

## 1. Project Overview
This repository hosts a high-throughput computer vision pipeline designed to quantify bacterial phenotypic responses to antibiotic stress. By replacing computationally expensive deep learning segmentation (O(N)) with dense optical flow estimation (O(1)), this system provides a lightweight, real-time kinetic proxy for bacterial mortality.

The pipeline utilizes **Farneback’s algorithm** to compute dense vector fields, applying **statistical outlier rejection** to isolate biological motility from stochastic sensor noise. It successfully identifies the "Point of Divergence" between Rifampicin-treated (RIF) and Control (REF), offering a computationally efficient, zero-latency alternative to traditional biomass area integration methods.

### Key Engineering Features
* **Computer Vision:** Dense optical flow estimation (Farneback) with spatial smoothing kernels to track coherent colony expansion.
* **Signal Processing:** Double-threshold filtering (Bottom 50% / Top 5%) to eliminate Brownian motion noise and camera artifacts.
* **Event Detection:** Automated temporal gating to identify pharmacological injection events ("fluid dynamic perturbations") and exclude them from growth metrics.

## 2. Directory Structure
The codebase is organized into a modular package structure to separate business logic from execution scripts.

```text
.
├── data/                          # Raw experimental image stacks
│   ├── REF_raw_data101_110/       # Control Group (Untreated)
│   └── RIF10_raw_data201_210/     # Experimental Group (Treated)
├── results/                       # Generated plots and time-series data
├── src/                           # Source Code
│   ├── optical_flow_pipeline.py   # Main differential analysis engine
│   ├── injection_event_detector.py# Temporal perturbation monitor
│   └── vector_field_visualizer.py # Qualitative heatmap generator
├── requirements.txt               # Dependencies
└── README.md
```

## 3. Setup & Installation
### Dependencies
Install the required Python libraries:

```bash
pip install -r requirements.txt
```
Output: Generates results_comparison.png, showing the net flow difference between REF and RIF and marking the exact frame where behavior diverges.

### Data Preparation
Ensure raw data is placed in the `data/` directory as shown in the structure above.

## 4. Usage Instructions

### Method 1: Quantitative Differential Analysis

Executes the core pipeline to generate kinetic time-series comparisons and statistical divergence plots.

```bash
python src/optical_flow_pipeline.py
```
**Output**: Generates `results/results_comparison.png`, visualizing the net flow difference and statistically identifying the onset of necrotic cessation.

### Method 2: Injection Event Detection

Analyzes temporal vector magnitude spikes to pinpoint the exact frame of compound administration.

```bash
python python src/injection_event_detector.py
```
**Output**: Generates a temporal profile identifying the "Injection Window" (fluid perturbation artifact) to ensure downstream data integrity.

### Method 3: Vector Field Visualization

Generates optical vector filed to visually verify flow dynamics and directional bias.

```bash
python python src/vector_field_visualizer.py
```
**Output**: Generates `results/results_comparison.png`, visualizing the net flow difference and statistically identifying the onset of necrotic cessation.

## 5. Interpretation of Results

The results from this method are subtle. The antibiotic effect does not cause the bacteria to immediately disappear; rather, it biases their movement direction.

### A. Quantitative Analysis (`results/results_comparison.png`)
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

### B. Qualitative Visualization (`results/`)
* **Files:** Images like `viz_RIF_frame030.png`.
* **Interpretation:** Do not look for a difference in the "size" or "area" of the colony, as both grow initially.
* **What to look for:** Look at the **arrows (Vectors)**.
    * **REF (Untreated):** Arrows point outward in all directions (healthy symmetric expansion).
    * **RIF (Treated):** You may see a tendency for arrows to align **rightward** (yellow/red arrows). This directional bias is the visual signature of the antibiotic treatment taking effect.