# Alzheimer's Disease Detection Pipeline

A research-oriented multimodal Alzheimer's disease detection pipeline for ADNI 3D MRI NIfTI data and XML clinical metadata.

The target classes are:
- CN: Cognitively Normal
- MCI: Mild Cognitive Impairment
- AD: Alzheimer's Disease

The pipeline supports MRI-only, clinical-only, and multimodal experiments.

## Important research status

This repository contains an experimental research pipeline. It is not a clinical diagnostic system.

The following must be clearly distinguished:

- Dataset-level evaluation is separate from single-case demonstration.
- A single MRI/XML demonstration is not a performance result.
- Metrics must be interpreted in the context of the small current dataset.
- No clinical conclusions should be made from model predictions.
- Counterfactual explanations are model-based examples, not causal explanations.
- SHAP values describe model behavior, not biological causality.

## Pipeline

```text
Raw 3D NIfTI
    |
    v
Dataset audit and MRI/XML mapping
    |
    v
Subject-level splitting
    |
    v
Processing-version grouping and selection
    |
    v
MRI preprocessing
    |
    v
Segmentation
    |
    v
ROI masks
    |
    v
Radiomics features
    |
    +----------------------+
    |                      |
    v                      v
Clinical XML features   Radiomics features
    |                      |
    +----------+-----------+
               |
               v
Multimodal feature alignment
               |
               v
Train-only imputation and scaling
               |
               v
Baseline / GWO / HHO feature selection
               |
               v
Logistic Regression and TabPFN
               |
               v
SHAP and counterfactual explanations
               |
               v
Uncertainty and calibration
               |
               v
Performance evaluation
               |
               v
Research dashboard and HTML demonstration
