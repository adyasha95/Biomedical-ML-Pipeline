# PredictionTendinopathy

A research-grade, **end‑to‑end nested cross‑validation pipeline** for **small, imbalanced binary classification problems**, built around a real‑world tendinopathy injury prediction task.

The repository currently contains **two complementary pipelines**:

1. **Classical ML + Imbalance Pipeline**  
   A scikit‑learn / imbalanced‑learn based model zoo with SMOTE/ADASYN and related resampling strategies, guarded thresholding, calibration, and SHAP/feature‑importance exports.

2. **TabTransformer Pipeline**  
   A deep‑learning extension using `pytorch-tabular`’s **TabTransformer** model, wired into the **same philosophy**: nested CV, guarded threshold selection, calibration, permutation baselines, and interpretability exports, but with **end‑to‑end representation learning** on the raw tabular features.

Both pipelines target the **same Excel dataset** and outcome, and write their outputs into a shared `runs/` directory for downstream analysis and plotting.

---

## 1. Project Overview

**Goal:**  
Provide a **reproducible template** for handling **small‑N, highly imbalanced medical / injury datasets** with:

- Robust **nested cross‑validation** (outer CV for performance estimation, inner CV for model/threshold tuning)
- **Guarded threshold selection** policies that trade off precision, sensitivity, and minimum predicted‑positive rate
- Careful **imbalance handling** (SMOTE/ADASYN with guards, SMOTEENN, undersampling, cost‑sensitive weighting, internally balanced ensembles)
- **Platt calibration** (or safe identity calibration) with reliability curve exports
- **Interpretability** via logistic coefficients, SHAP summaries, and feature importances
- Resume‑safe **per‑fold checkpoints** and per‑model/per‑imbalance summary tables

You can either:

- **Use it “as is”** with the tendinopathy dataset, or  
- **Adapt it** to any other binary classification problem by changing the config paths and target name.

---

## 2. Dataset & Outcome Definition

By default, both pipelines expect:

- An **Excel file** (XLSX) containing:
  - One binary outcome column (e.g., `"Tendinopathy Injury"`)
  - Multiple clinical / biomechanical / questionnaire features

In the TabTransformer script, this is controlled via:

```python
class Config:
    DATA_PATH: str = "/content/drive/MyDrive/tendinopathy/Final Dastaset Oct25.xlsx"
    TARGET_NAME: str = "Tendinopathy Injury"
```

To run this repo on your own machine:

1. Place your dataset, e.g. `data/Final_Dataset_Oct25.xlsx`, inside the repository.
2. Update `Config.DATA_PATH` in **both scripts** to point to your local path, e.g.:

   ```python
   DATA_PATH: str = "data/Final_Dataset_Oct25.xlsx"
   ```

3. Make sure `TARGET_NAME` matches the exact column name of your binary label.

The scripts automatically:

- Drop rows where the target is missing
- Detect and drop:
  - ID‑like columns (e.g. patient IDs, codes)
  - Date‑like columns (e.g. timestamps, injury dates)
  - Outcome‑leaky columns (e.g. explicit injury/diagnosis fields)
  - Very high missingness (> `MISSING_THRESHOLD`)
  - Near‑zero variance or constant features

Resulting curated feature lists and removal logs are exported as:

- `curation_artifacts/final_features.csv`
- `curation_artifacts/must_remove.csv`
- `curation_artifacts/missing_rate_overall.csv`

---

## 3. Environment, Dependencies & Reproducibility

### 3.1. Recommended Python Version

- Python **3.9+** (works with modern PyTorch, scikit‑learn, and pytorch‑tabular stacks)

### 3.2. Installation

Clone and create a virtual environment:

```bash
git clone https://github.com/<your-username>/PredictionTendinopathy.git
cd PredictionTendinopathy

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

A typical `requirements.txt` (you can adapt as needed) would include:

```text
numpy
pandas
scikit-learn
imbalanced-learn
openpyxl
tqdm
shap

torch
pytorch-tabular
omegaconf
pytorch-lightning  # often required by pytorch-tabular

xgboost
lightgbm
catboost
pytorch-tabnet
```

> **Note:**  
> - `xgboost`, `lightgbm`, `catboost`, and `pytorch-tabnet` are **optional**.  
> - The code checks their availability and only runs them if the imports succeed.  
> - The TabTransformer pipeline will **only** run if `pytorch-tabular` is installed and importable.

### 3.3. Reproducibility

Key reproducibility knobs (identical in spirit across both pipelines):

- Global random seed (`Config.RANDOM_STATE`)
- Fixed CV splits via `StratifiedKFold` / `RepeatedStratifiedKFold`
- Explicit thresholds grid for policy search (`Config.THR_GRID`)
- Consistent calibration and permutation baseline parameters

Set once in the `Config` dataclass at the top of each script.

---

## 4. Repository Structure

A minimal GitHub‑friendly layout for this project:

```text
PredictionTendinopathy/
├── predinjury_main.py      # Classical ML + imbalance pipeline (nested CV + SHAP)
├── predinjury_tabtransformer.py # TabTransformer-only nested CV sweep (this script)
├── README.md                      # This documentation
├── requirements.txt               # Python dependencies
├── data/
│   └── Final_Dataset_Oct25.xlsx   # Example dataset (not committed, local only)
└── runs/
    └── ...                        # Per-model, per-fold JSON/CSV/checkpoint outputs
```

You can, of course, rename the two main scripts, e.g.:

- `predinjury_main.py` → `trial_runner.py`  
- `predinjury_tabtransformer.py` → `trial_runner_tabtransformer.py`

Just make sure the README and any example commands use the same names.

---

## 5. How to Run the Pipelines

### 5.1. Classical ML + Imbalance Pipeline

**Script:** `classical_trial_runner.py`  
*(Name may differ in your repo; adjust accordingly.)*

This pipeline:

- Builds a **CombinedPreprocessor**:
  - Residualizes numeric features against covariates (e.g. Age, Sex)
  - Imputes missing values (median / most frequent)
  - Standardizes numeric features
  - One‑hot encodes categoricals
  - Ordinal‑encodes explicitly ordinal variables (if defined)
- Sweeps a **model zoo**:
  - Logistic Regression (L1/L2/Elastic Net)
  - SVM (RBF)
  - Random Forest
  - Balanced Random Forest
  - EasyEnsemble
  - XGBoost (if installed)
  - LightGBM (if installed)
  - CatBoost (if installed)
  - TabNet (if installed)
- Applies **imbalance handling**:
  - none
  - cost‑sensitive weighting
  - SMOTE / ADASYN with guards (`SafeSampler`)
  - SMOTEENN
  - random undersampling
  - internal balancing for BalancedRF / EasyEnsemble
- Runs **nested CV**, calibration, SHAP / feature‑importance exports, and summary aggregation.

Run:

```bash
python predinjury_main.py
```

Outputs will go into:

- `runs/` (per‑model/per‑imbalance per‑fold metrics)
- `runs/summary_by_model.csv` (global ranking table)
- `runs/fold_metrics_all.csv` (all folds, all models, all imbalance modes)

### 5.2. TabTransformer Pipeline

**Script:** `tabtransformer_trial_runner.py`  
(the large script you see in this repo, centered on `PTTabTransformer`)

The TabTransformer pipeline:

- Uses *the same curated features* as the classical pipeline
- Bypasses the scikit‑learn preprocessor for the model itself (TabTransformer consumes the raw DataFrame)
- Restricts imbalance modes to:
  - `"none"`
  - `"cost_sensitive"` (implemented as a **class‑weighted CrossEntropyLoss** inside the model)
- Integrates directly into the **same nested CV + threshold + calibration** framework
- Provides the **same style of fold‑wise metrics and summary tables**, but only for TabTransformer combos.

Run:

```bash
python predinjury_tabtransformer.py
```

What it does:

1. Loads and curates the data (ID/date/leakage masks, missingness filters, preferred feature set).
2. Runs a small **TabTransformer smoke test**:
   - Quick fit on a tiny subset
   - Ensures `predict_proba` and calibration logic behave as expected
3. **Cleans previous TabTransformer artifacts** via `nuke_tabtx_artifacts()`:
   - Deletes old `TabTransformer__*.json` checkpoints, CSVs, and per‑fold dirs
   - This ensures you rerun a clean TabTransformer experiment without interfering with classical runs
4. Runs **repeated stratified outer CV** and for each fold:
   - Inner CV `RandomizedSearchCV` over TabTransformer hyperparameters:
     - `max_epochs`, `batch_size`, `lr`, `dropout`, `n_layers`, `n_heads`, `embed_dim`, `attn_dropout`, `patience`
   - Learns a guarded threshold using the **same policy** as the classical pipeline
   - Fits a CalibratedClassifierCV (Platt) when sufficient positives/negatives are present
   - Computes permutation AUC baselines and calibration curves
   - Writes per‑fold model pickle, metrics, threshold info, and SHAP/FI artifacts (where applicable)
5. Aggregates **per‑model/per‑imbalance summary** rows and writes them to `summary_by_model.csv`.

---

## 6. Learning Algorithms & Imbalance Strategies

### 6.1. Classical Model Zoo

Depending on library availability, the classical pipeline can include:

- **Logistic Regression**
  - L2 (`LogReg_L2`)
  - Elastic Net (`LogReg_EN`)
  - L1 (`LogReg_L1`)
- **SVM (RBF)**
- **Random Forest**
- **Balanced Random Forest**
- **EasyEnsembleClassifier**
- **XGBoost** (`XGBClassifier`) – optional
- **LightGBM** (`LGBMClassifier`) – optional
- **CatBoost** (`CatBoostClassifier`) – optional
- **TabNet** (`TabNetClassifier`) – optional

Each model comes with a reasonable **hyperparameter search grid**, used by `RandomizedSearchCV` within the inner CV.

### 6.2. Imbalance Modes

For classical models:

- `"none"` – vanilla training on imbalanced data
- `"cost_sensitive"` – use `class_weight="balanced"` when supported
- `"smote"` / `"adasyn"` – via `SafeSampler` (with:
  - Minimum minority count to enable synthetic sampling
  - Guarded k‑neighbors range)
- `"smoteenn"` – combined over/under sampling
- `"undersample"` – `RandomUnderSampler`
- `"internal_balance"` – for models that **do their own** rebalancing (BalancedRF, EasyEnsemble)

For **TabTransformer**:

- No external resamplers (SMOTE/ADASYN/etc.) are used.
- Supported imbalance modes:
  - `"none"` – standard CrossEntropyLoss
  - `"cost_sensitive"` – CrossEntropyLoss with class weights proportional to inverse prevalence

This is necessary because TabTransformer learns **end‑to‑end on the raw DataFrame**, and the pipeline does not pass through the usual `CombinedPreprocessor` / imblearn samplers.

---

## 7. Nested CV, Thresholding, & Calibration

### 7.1. CV Strategy

- **Outer CV:** `RepeatedStratifiedKFold`
  - `OUTER_FOLDS` × `OUTER_REPEATS`
  - Provides a distribution of fold‑wise performance metrics
- **Inner CV:** `StratifiedKFold`
  - Used inside `RandomizedSearchCV`
  - Tuning both:
    - Model hyperparameters
    - *Implicitly* the final threshold policy via out‑of‑fold probabilities

### 7.2. Threshold Selection Policies

The code currently supports:

- `"fixed_precision_guarded"` (default)
  - Seek thresholds where:
    - PPV ≥ `FIXED_PRECISION`
    - Sensitivity ≥ `MIN_SENS_AT_THR`
  - Includes a **tolerant fallback** if constraints cannot be met exactly
  - Includes last‑resort fallbacks (max BAC under constraints, etc.)
- `"youden_specfloor"`
  - Maximize Youden’s J (Sensitivity + Specificity − 1)
  - **Subject to** a minimum specificity floor (`SPEC_FLOOR`)

Policies search over a **threshold grid** enriched by the empirical score distribution:

- Base grid: `Config.THR_GRID`
- Plus midpoints and local refinements based on predictions

All policy details and decisions are stored in a JSON blob (`ThrInfo`) per fold.

### 7.3. Calibration and Reliability Curves

For each outer fold:

1. A **best pipeline** is refit on the full training fold.
2. Calibration is chosen via `build_safe_calibrator`:
   - If enough positives/negatives:
     - Wrap the pipeline in `CalibratedClassifierCV` (sigmoid / Platt scaling)
   - Else:
     - Use an **IdentityCalibrator** that just passes through `predict_proba` or `decision_function`.
3. The calibrated probabilities on the test fold are:
   - Used for threshold metrics
   - Logged to compute **calibration curves** (`calibration_curve_data`):
     - Bin centers
     - Observed positive rates
     - Counts per bin

Calibration curves are serialized into per‑fold JSON for plotting.

---

## 8. Evaluation Metrics

For each **outer test fold**, the pipelines compute metrics at:

1. **Default threshold = 0.5**
2. **Tuned threshold** (`Thr`) learned from inner CV policy

Metrics include:

- Sensitivity / Recall
- Specificity
- Balanced Accuracy (BAC)
- ROC AUC
- PR AUC
- PPV (Precision)
- F1
- NPV
- Brier score

In addition, a **permutation AUC baseline** is computed:

- Shuffle labels `N_PERMUTATIONS` times
- Compute AUC distribution under the null
- Report:
  - Observed AUC (`Obs_ROC_AUC`)
  - Null interval (`Perm_AUC_lo`, `Perm_AUC_hi`)

---

## 9. Interpretability & Exports

Per **(model, imbalance, fold)**, the pipeline tries to export:

1. **Logistic Regression Coefficients**
   - CSV with feature names + coefficients
   - Stored under the fold’s artifact directory
2. **SHAP Values**
   - For tree ensembles: `shap.TreeExplainer`
   - For linear models: `shap.LinearExplainer`
   - Fallback: `shap.KernelExplainer` on a small background set
   - Writes `shap_mean_abs.csv` with mean absolute SHAP per feature
   - Saves top 15 features per fold for quick inspection
3. **Feature Importances**
   - If SHAP fails and a model exposes `feature_importances_`:
     - Exports `feature_importances.csv`
     - Used as a fallback interpretability channel

All artifact paths are stored in the per‑fold metrics row:

- `ModelPicklePath`
- `CoefTablePath`
- `SHAPSummaryPath`

This makes it easy to later load the best model(s), inspect coefficients/SHAP, and integrate into plots or reports.

---

## 10. TabTransformer‑Specific Notes & Constraints

- **Optional dependency:** The TabTransformer pipeline is only active if:
  - `pytorch-tabular` imports successfully  
  - The helper compatibility shims (`_safe_TabTransformerConfig`, `_safe_TrainerConfig`, `_safe_OptimizerConfig`, `_safe_DataConfig`) are able to instantiate configs for the installed version.
- **Device selection:**  
  - A small `_infer_device` helper tries to use `"cuda"` if available; otherwise falls back to `"cpu"`.
- **Probability outputs:**  
  - The wrapper looks for a **probability‑like column** in the `TabularModel.predict()` output:
    - Prefer columns ending with `"1_probability"` or `"_1_probability"`
    - Fallbacks include `"prediction_probability"`, `"prediction_proba"`, `"prediction_1"`, or (last resort) `"prediction"`
- **Imbalance handling:**  
  - External samplers are **not** used.
  - `"cost_sensitive"` mode converts class prevalence into a `weight` tensor passed to `CrossEntropyLoss`.

If `TABTRANSFORMER_AVAILABLE` is `False`, the script will **skip** TabTransformer model specs and log a diagnostic message.

---

## 11. Typical Workflow

1. **Prepare data**
   - Place Excel dataset in `data/`
   - Set `DATA_PATH` and `TARGET_NAME` in both scripts.

2. **Run classical pipeline**
   ```bash
   python classical_trial_runner.py
   ```
   - Inspect:
     - `runs/summary_by_model.csv`
     - `runs/fold_metrics_all.csv`
     - Per‑fold artifacts for best models

3. **Run TabTransformer pipeline**
   ```bash
   python tabtransformer_trial_runner.py
   ```
   - Confirm TabTransformer imports successfully.
   - Check:
     - `runs/summary_by_model.csv` (now including TabTransformer rows)
     - `runs/TabTransformer__*/fold*/` for model pickles and interpretability outputs.

4. **Downstream analysis**
   - Use your own analysis notebooks / R scripts to:
     - Plot AUC/BAC distributions
     - Plot calibration curves and permutation nulls
     - Visualize SHAP or feature importance rankings
     - Compare classical models vs TabTransformer on stability and robustness.

---

## 12. Possible Extensions

You can extend this repo by:

- Adding **new models** to the classical model zoo (e.g. GLMM, GAMs, other neural nets).
- Implementing **multi‑class** or **time‑to‑event** versions of the pipeline.
- Adding new **threshold policies**, e.g. cost curves, fixed sensitivity, or operating points defined by clinical constraints.
- Integrating **reporting utilities** (e.g. automatic PDF/HTML reports summarizing results and SHAP plots).

---

## 13. Citation / Acknowledgment

If you use this codebase in a paper or presentation, you can acknowledge it informally as:

> “We used a custom nested cross‑validation and imbalance‑handling pipeline with classical models and TabTransformer, based on the open‑source *PredictionTendinopathy* repository.”

(Replace with your own canonical citation once the associated manuscript is published.)

---

If you encounter issues running the code (especially with `pytorch-tabular` versions), check the diagnostic prints at the top of the scripts and verify your library versions match the expected APIs.
