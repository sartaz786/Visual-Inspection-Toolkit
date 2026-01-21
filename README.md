# Visual Inspection Toolkit 👁️

A computer vision repository featuring two core modules for industrial inspection: **Multi-Modal Alignment** and **Structural Change Detection**.

## 🚀 Modules

### 1. Thermal-RGB Alignment
* **File:** `modules/thermal_alignment.py`
* **Description:** Aligns thermal images (low-res) with RGB images (high-res) using SIFT feature matching and RANSAC-based affine transformation.
* **Key Tech:** OpenCV, SIFT, CLAHE, RANSAC.

### 2. Change Detection
* **File:** `modules/change_detection.py`
* **Description:** Identifies structural changes (e.g., missing components) between "Before" and "After" images.
* **Key Tech:** Morphological Operations, Adaptive Thresholding, Contour Analysis.

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR-USERNAME/Visual-Inspection-Toolkit.git](https://github.com/YOUR-USERNAME/Visual-Inspection-Toolkit.git)
