# End-to-End ML Pipeline for Clinical & Biomedical Tabular Data  
Nested Cross-Validation • Imbalance Handling • Classical ML • TabTransformer

This repository contains a research-grade, fully modular pipeline for **predictive modeling on structured biomedical or clinical tabular datasets**.

It provides two complementary modeling tracks:

---

## 1. Classical Machine Learning Pipeline  
(based on scikit-learn + imbalanced-learn)

Includes:
- Nested cross-validation (outer CV for evaluation, inner CV for tuning)
- Model zoo (Logistic Regression, XGBoost, Random Forest, LightGBM, etc.)
- Imbalance-aware methods (SMOTE / ADASYN / Class-weighting)
- Threshold tuning and calibration
- SHAP-based feature importance extraction
- Exportable fold-wise metrics, confusion matrices, ROC/PR curves

---

## 2. TabTransformer Deep Learning Pipeline  
(using `pytorch-tabular`)

Includes:
- TabTransformer architecture with attention-based feature embeddings  
- End-to-end representation learning  
- Integrated training/evaluation loop  
- Support for cost-sensitive learning  
- Unified `runs/` directory shared with the classical ML pipeline

---

## Data Privacy & Ethical Use Notice (Important)

> **The original version of this pipeline was developed with sensitive biomedical data (patient-level clinical measurements).  
> Due to GDPR, institutional ethics, and data-sharing restrictions, no original datasets can be included in this repository.**
>
> All examples, configuration files, and loaders included here are **generic placeholders**, so users may connect their own datasets safely.

This repository provides **code only** — users must supply their own compliant dataset.

---

## Features at a Glance

### ✔ Reproducible ML Engineering
- Deterministic runs (random seeds)
- Config-driven experiment definitions
- Automatic export of results into structured folders

### ✔ Flexible Model Selection
- Classical models and ensemble methods  
- Deep learning via TabTransformer  
- Easy extension to other tabular DL architectures

### ✔ Analysis-Ready Exports
- SHAP values  
- Per-fold predictions  
- Fold-wise confusion matrices  
- ROC, PR curves, threshold vs metric plots  
- Aggregated metrics tables  

### ✔ Imbalance-Robust Training
- SMOTE, ADASYN, RandomOverSampler, class-weights  
- Proper separation of CV folds to prevent leakage  

---

## Requirements

- Python 3.9+  
- scikit-learn  
- imbalanced-learn  
- lightgbm / xgboost  
- pytorch-tabular  
- PyTorch >= 2.0  
- pandas, numpy, matplotlib, seaborn  

---

## Basic Usage

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
