# Migration Report: Monolithic TensorFlow to Modular PyTorch

This document summarizes the comprehensive changes made to the "Smart Disease Predictor" codebase to improve maintainability, performance, and robustness.

## 1. Architectural Changes

### Monolithic to Modular
- **Original**: Inference and preprocessing were tightly coupled within `app.py`. Training logic was scattered across Jupyter Notebooks.
- **New**: 
  - **`src/utils/preprocess.py`**: Centralized all data preparation, feature engineering, and model architecture definitions.
  - **`src/training/`**: Dedicated scripts for each disease, allowing independent training and versioning.
  - **`config/train_config.yaml`**: Decoupled hyperparameters and paths from code.

### Framework Shift (Image Models)
- **Original**: TensorFlow/Keras (`.h5` models) for Malaria and Pneumonia.
- **New**: **PyTorch (`.pth` models)**.
  - Enabled **CUDA acceleration** for GPU-trained models.
  - Switched to Transfer Learning with `MobileNetV2` (Malaria) and `ResNet18` (Pneumonia).
  - Integrated `tqdm` for interactive terminal progress bars.

## 2. Model Logic Fixes

### Kidney Disease Predictor
- **Issue**: Form sent numeric codes (0/1) while the model was trained on text labels ("normal"/"abnormal"). Target labels were also inverted.
- **Fix**: Implemented explicit categorical mapping in `preprocess_kidney` and retrained the model. Remapped `ckd` to `1` (Positive) for UI consistency.

### Liver Disease Predictor
- **Issue**: Inconsistent feature count between the dataset and the HTML form.
- **Fix**: Standardized the 10-feature set (ILPD format) and updated the template to match.

### Breast Cancer Predictor
- **Issue**: Mismatch between the model's 30-feature expectation and the dataset's 20-feature availability.
- **Fix**: Synchronized the model to use Mean and SE features only, matching the dataset and form inputs.

## 3. Application Robustness

### Enhanced Input Handling
- Added "graceful degradation" in `app.py` for empty or non-numeric form inputs, defaulting to `0.0` or raw strings instead of crashing.

### Automated Pipelines
- **Data Acquisition**: Created `data/download_datasets.py` to automate the collection of all 5 CSV datasets.
- **Orchestration**: Created `src/training/train_all.py` to allow one-command training of the entire suite.

## 4. Dependencies
- Removed `tensorflow` and `keras`.
- Added `torch`, `torchvision`, `tqdm`, `PyYAML`, and `requests`.

---
