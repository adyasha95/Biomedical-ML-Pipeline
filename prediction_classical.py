# ===== Dependency bootstrap (CPU-only) =====
# Put this block at the very top of your script.
import sys, subprocess, importlib, shutil

# Helper to install one or more pip packages from inside Python.
# Used by `_ensure` to make the script more self-contained in notebook/Colab environments.
def _pip_install(pkgs):
    print(f"[SETUP] Installing: {pkgs}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *pkgs])

def _ensure(pkg_import_name, pip_name=None, extra_pip_args=None):
    """
    Try to import `pkg_import_name`. If missing, pip install `pip_name` (or `pkg_import_name`),
    then import again. Returns the imported module or None on failure.
    """
    pip_name = pip_name or pkg_import_name
    try:
        return importlib.import_module(pkg_import_name)
    except Exception:
        try:
            args = [pip_name] + (extra_pip_args or [])
            _pip_install(args)
            return importlib.import_module(pkg_import_name)
        except Exception as e:
            print(f"[SETUP] WARNING: Could not import/install {pip_name}: {e}")
            return None

# Core extras used by your pipeline
openpyxl = _ensure("openpyxl")  # for pandas Excel
imblearn  = _ensure("imblearn", "imbalanced-learn")
xgb_mod   = _ensure("xgboost")
lgbm_mod  = _ensure("lightgbm")
cat_mod   = _ensure("catboost")

# Optional: TabNet + CPU PyTorch wheels
# Comment out this block if you don't need TabNet.
tabnet_mod = None
try:
    tabnet_mod = importlib.import_module("pytorch_tabnet.tab_model")
except Exception:
    # Ensure torch CPU wheels first (works in Colab/most Linux)
    try:
        _pip_install(["torch", "torchvision", "torchaudio",
                      "--index-url", "https://download.pytorch.org/whl/cpu"])
    except Exception as e:
        print(f"[SETUP] WARNING: Could not install CPU torch wheels: {e}")
    # Then TabNet
    try:
        tabnet_mod = _ensure("pytorch_tabnet.tab_model", "pytorch-tabnet")
    except Exception:
        tabnet_mod = None

# Expose optional classes (None if unavailable)
try:
    XGBClassifier = xgb_mod.XGBClassifier if xgb_mod else None
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier  # noqa: F401
except Exception:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier  # noqa: F401
except Exception:
    CatBoostClassifier = None

try:
    from pytorch_tabnet.tab_model import TabNetClassifier  # noqa: F401
except Exception:
    TabNetClassifier = None

# Sanity printout so you can see what's available
print("[SETUP] Optional libs availability -> "
      f"imblearn={imblearn is not None}, "
      f"xgboost={XGBClassifier is not None}, "
      f"lightgbm={LGBMClassifier is not None}, "
      f"catboost={CatBoostClassifier is not None}, "
      f"tabnet={TabNetClassifier is not None}, "
      f"openpyxl={openpyxl is not None}")

!pip install imbalanced-learn xgboost lightgbm catboost pytorch-tabnet openpyxl --quiet

# ============================================================
# Small-N Imbalanced Trial Runner — Full Classifier x Imbalance Sweep
# - Nested CV
# - Guarded thresholding
# - Resume-safe checkpoints
# - Extended model zoo & resampling modes (incl. ADASYN)
# - BalancedRF / EasyEnsemble treated as standalone classifiers
# - SHAP export for tree/ensemble models
# - Coefficient export for linear models (CIs TODO)
# ============================================================

import os, json, time, warnings, re, pickle
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, Any, List
from copy import deepcopy
from pathlib import Path
from tqdm.auto import tqdm

# sklearn / imblearn core
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    recall_score,
    confusion_matrix,
    precision_score,
    f1_score,
    brier_score_loss
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval, StrOptions, Integral
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_X_y

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.base import BaseSampler
from imblearn.combine import SMOTEENN
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier

# optional libs
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None
try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
except Exception:
    TabNetClassifier = None
try:
    import torch
    from pytorch_tabular import TabularModel
    from pytorch_tabular.models.tab_transformer import TabTransformerConfig
    from pytorch_tabular.config import (
        DataConfig,
        TrainerConfig,
        OptimizerConfig,
        ExperimentConfig,
        ModelConfig
    )
    TABTRANSFORMER_AVAILABLE = True
except Exception:
    TABTRANSFORMER_AVAILABLE = False
# SHAP (optional)
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

import shap

try:
    _ = shap  # simple presence check
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False
# ============================================================
# Config
# ============================================================

# Global configuration object:
# - Paths to data and outputs
# - CV design
# - Thresholding policies
# - Imbalance and calibration settings
# - Feature filtering rules
@dataclass
class Config:
    # data
    DATA_PATH: str = "/content/drive/MyDrive/tendinopathy/Final Dastaset Oct25.xlsx"
    TARGET_NAME: str = "Tendinopathy Injury"

    # CV
    RANDOM_STATE: int = 42
    OUTER_FOLDS: int = 4
    OUTER_REPEATS: int = 5
    INNER_FOLDS: int = 3
    N_TRIALS: int = 20
    N_PERMUTATIONS: int = 200
    # policy knobs (threshold selection)
    THRESH_PRIMARY: str = "fixed_precision_guarded"  # {"fixed_precision_guarded","youden_specfloor"}
    FIXED_PRECISION: float = 0.60   # PPV target
    MIN_SENS_AT_THR: float = 0.40   # minimum sensitivity
    SPEC_FLOOR: float = 0.60        # for youden_specfloor
    THR_GRID: Tuple[float, ...] = tuple(np.linspace(0.02, 0.98, 49))

    # calibration
    CAL_MIN_POS: int = 3
    CAL_MIN_NEG: int = 3
    CAL_CURVE_BINS: int = 10  # reliability curve bins

    # imbalance / sampler guards
    MIN_MINORITY_FOR_SYNTH: int = 6
    KNN_MAX: int = 5
    KNN_MIN: int = 1

    # features / filtering
    MISSING_THRESHOLD: float = 0.4
    NZV_MAJORITY: float = 0.95
    HIGH_CARD_CAT: int = 20
    ALWAYS_KEEP: Tuple[str, ...] = ("Age", "Sex")

    # hyperparam ranges shared across models
    C_GRID: Tuple[float, ...] = tuple(np.logspace(-7, -1, 7))
    EN_L1R_GRID: Tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)

    # saving
    SAVE_DIR: str = "/content/drive/MyDrive/tendinopathy/trial_guarded_fixedP_full_2"
    RESULTS_DIRNAME: str = "runs"

CFG = Config()
os.makedirs(CFG.SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(CFG.SAVE_DIR, CFG.RESULTS_DIRNAME), exist_ok=True)
rng_global = check_random_state(CFG.RANDOM_STATE)

# ============================================================
# Utility helpers
# ============================================================

# Convert a label series to a binary float vector (0/1).
# Handles strings like 'yes', 'no', 'true', 'false', '1', '0', or
# numeric two-level encodings. Used for the main target column.
def to_binary(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower().map({
        "yes":1,"no":0,"true":1,"false":0,"1":1,"0":0
    })
    if s.isna().all():
        s = pd.to_numeric(series, errors="coerce")
        vals = sorted(s.dropna().unique().tolist())
        if len(vals) == 2:
            s = s.map({vals[0]:0, vals[1]:1})
    return s.astype("float")

# Specificity helper: TN / (TN + FP)
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn/(tn+fp) if (tn+fp)>0 else 0.0

# Negative predictive value helper: TN / (TN + FN)
def neg_pred_value(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fn) if (tn + fn) > 0 else 0.0

# Compute a standard pack of metrics at a fixed threshold (default 0.5).
# Includes ROC AUC, PR AUC, BAC, PPV (client naming), F1, NPV, Brier.
def safe_metric_pack(y_true, y_prob, thr=0.5) -> Dict[str,float]:
    """
    Metrics at a fixed threshold (default = 0.5) used for descriptive / AUC reporting.
    NOTE: 'PPV' not 'Precision', per client naming.
    """
    y_pred = (y_prob >= thr).astype(int)

    sens = recall_score(y_true, y_pred, zero_division=0)
    spec = specificity_score(y_true, y_pred)
    bac  = 0.5*(sens+spec)

    roc  = np.nan
    pr   = np.nan
    try:
        if len(np.unique(y_true)) > 1:
            roc = roc_auc_score(y_true, y_prob)
            pr  = average_precision_score(y_true, y_prob)
    except Exception:
        pass

    ppv = precision_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    npv  = neg_pred_value(y_true, y_pred)

    try:
        brier = brier_score_loss(y_true, y_prob)
    except Exception:
        brier = np.nan

    return {
        "Sensitivity": sens,
        "Specificity": spec,
        "BAC": bac,
        "ROC_AUC": roc,
        "PR_AUC": pr,
        "PPV": ppv,
        "F1": f1,
        "NPV": npv,
        "Brier": brier
    }

# Build a refined grid of candidate thresholds from base config and the
# observed probability distribution. Used by threshold policies.
def thr_grid_from_scores(y_prob: np.ndarray) -> np.ndarray:
    base = np.asarray(CFG.THR_GRID)
    p = np.unique(np.clip(y_prob, 1e-6, 1-1e-6))
    mids = (p[:-1] + p[1:]) / 2.0 if len(p) > 1 else p
    hi = np.quantile(p, 0.9) if p.size else 0.9
    fine = hi + np.array([-0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02])
    return np.unique(np.clip(np.r_[base, p, mids, fine], 1e-6, 1-1e-6))

# Simple mkdir wrapper that respects existing directories.
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

# JSON-dumping wrapper that tolerates unserializable objects.
def safe_json_dumps(obj):
    try:
        return json.dumps(obj, default=str)
    except Exception:
        return json.dumps({"_unserializable": True}, default=str)
# ============================================================
# Threshold finders
# ============================================================

# Main "fixed precision guarded" threshold policy:
# - Target PPV >= fixed_precision and sensitivity >= min_sens
# - Contains multiple fallback strategies if strict conditions fail.
def find_threshold_fixedP_guarded(
    y_true,
    y_prob,
    fixed_precision=0.60,
    min_sens=0.40,
    tol_ppv=0.05,
    min_pred_rate=0.05
):
    thrs = thr_grid_from_scores(y_prob)

    def pack(thr):
        y_pred = (y_prob >= thr).astype(int)
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        tn = ((y_true == 0) & (y_pred == 0)).sum()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        bac  = 0.5*(sens+spec)
        ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        p_rate = y_pred.mean()
        return ppv, sens, spec, bac, p_rate

    # strict policy
    strict = []
    for t in thrs:
        ppv, sens, spec, bac, p_rate = pack(t)
        if p_rate == 0:
            continue
        if ppv >= fixed_precision and sens >= min_sens:
            strict.append((t, bac, spec, sens, ppv, p_rate))
    if strict:
        strict.sort(key=lambda x: (x[1], x[3], x[2], x[0]), reverse=True)
        t, bac, spec, sens, ppv, p_rate = strict[0]
        return float(t), {
            "mode":"fixed_precision_guarded",
            "fixedPPV":fixed_precision,
            "min_sens":min_sens,
            "bac":float(bac),
            "spec":float(spec),
            "sens":float(sens),
            "ppv":float(ppv),
            "pred_rate":float(p_rate)
        }

    # tolerant fallback
    tolerant = []
    for t in thrs:
        ppv, sens, spec, bac, p_rate = pack(t)
        if p_rate == 0:
            continue
        if (ppv >= (fixed_precision - tol_ppv)) and sens >= min_sens and p_rate >= min_pred_rate:
            tolerant.append((t, bac, spec, sens, ppv, p_rate))
    if tolerant:
        tolerant.sort(key=lambda x: (
            x[1], x[3], x[2], -abs(x[4]-fixed_precision), x[0]
        ), reverse=True)
        t, bac, spec, sens, ppv, p_rate = tolerant[0]
        return float(t), {
            "mode":"fixed_precision_guarded_tolerant",
            "fixedPPV":fixed_precision,
            "tol_ppv":tol_ppv,
            "min_sens":min_sens,
            "bac":float(bac),
            "spec":float(spec),
            "sens":float(sens),
            "ppv":float(ppv),
            "pred_rate":float(p_rate)
        }

    # constrained BAC fallback
    fallback = []
    for t in thrs:
        ppv, sens, spec, bac, p_rate = pack(t)
        if p_rate >= min_pred_rate:
            fallback.append((t, bac, spec, sens, ppv, p_rate))
    if fallback:
        fallback.sort(key=lambda x: (x[1], x[3], x[2], x[0]), reverse=True)
        t, bac, spec, sens, ppv, p_rate = fallback[0]
        return float(t), {
            "mode":"max_bac_constrained",
            "min_pred_rate":min_pred_rate,
            "bac":float(bac),
            "spec":float(spec),
            "sens":float(sens),
            "ppv":float(ppv),
            "pred_rate":float(p_rate)
        }

    # best non-degenerate BAC fallback
    any_nd = []
    for t in thrs:
        ppv, sens, spec, bac, p_rate = pack(t)
        if p_rate > 0:
            any_nd.append((t, bac, spec, sens, ppv, p_rate))
    if any_nd:
        any_nd.sort(key=lambda x: (x[1], x[3], x[2], x[0]), reverse=True)
        t, bac, spec, sens, ppv, p_rate = any_nd[0]
        return float(t), {
            "mode":"max_bac_non_degenerate",
            "bac":float(bac),
            "spec":float(spec),
            "sens":float(sens),
            "ppv":float(ppv),
            "pred_rate":float(p_rate)
        }

    # absolute fallback
    return 0.5, {"mode":"fallback_0.5"}

# Threshold finder based on Youden's index with a minimum specificity floor.
# Used as an alternative to the fixed precision policy.
def find_threshold_youden_specfloor(y_true, y_prob, spec_floor=0.60):
    thrs = thr_grid_from_scores(y_prob)
    cand = []
    for t in thrs:
        y_pred = (y_prob >= t).astype(int)
        spec = specificity_score(y_true, y_pred)
        sens = recall_score(y_true, y_pred, zero_division=0)
        if spec >= spec_floor:
            bac = 0.5*(sens+spec)
            cand.append((t, bac, sens, spec))
    if cand:
        cand.sort(key=lambda x: (x[1], x[3], x[0]), reverse=True)
        t, bac, sens, spec = cand[0]
        return float(t), {
            "mode":"youden_specfloor",
            "spec_floor":spec_floor,
            "bac":float(bac),
            "sens":float(sens),
            "spec":float(spec)
        }

    # fallback: best BAC overall
    any_t = []
    for t in thrs:
        y_pred = (y_prob >= t).astype(int)
        sens = recall_score(y_true, y_pred, zero_division=0)
        spec = specificity_score(y_true, y_pred)
        bac = 0.5*(sens+spec)
        any_t.append((t, bac))
    any_t.sort(key=lambda x: x[1], reverse=True)
    return float(any_t[0][0]), {
        "mode":"youden_any",
        "bac":float(any_t[0][1])
    }

# Dispatch function for threshold policies.
# Currently supports "fixed_precision_guarded" and "youden_specfloor".
def find_threshold(y_true, y_prob, mode: str):
    if mode == "fixed_precision_guarded":
        return find_threshold_fixedP_guarded(
            y_true,
            y_prob,
            fixed_precision=CFG.FIXED_PRECISION,
            min_sens=CFG.MIN_SENS_AT_THR
        )
    elif mode == "youden_specfloor":
        return find_threshold_youden_specfloor(
            y_true,
            y_prob,
            spec_floor=CFG.SPEC_FLOOR
        )
    else:
        return find_threshold_fixedP_guarded(
            y_true,
            y_prob,
            fixed_precision=CFG.FIXED_PRECISION,
            min_sens=CFG.MIN_SENS_AT_THR
        )

# ============================================================
# ResidualizeCovariates + CombinedPreprocessor
# ============================================================

class ResidualizeCovariates(BaseEstimator, TransformerMixin):
    """
    Residualize numeric feature columns on covariates (Age, Sex, etc.).
    Helps avoid direct leakage of those covariates if desired.
    """
    def __init__(self, covariate_cols):
        self.covariate_cols = list(covariate_cols) if covariate_cols is not None else []
        self.models_ = {}
        self.feature_cols_ = None
        self.covar_ct_ = None
        self.num_covars_ = []
        self.cat_covars_ = []

    # Split covariate columns into numeric vs categorical subsets
    # based on dtype in the input DataFrame.
    def _split_covars(self, X: pd.DataFrame):
        num_covs, cat_covs = [], []
        for c in self.covariate_cols:
            if c in X.columns:
                if pd.api.types.is_numeric_dtype(X[c]):
                    num_covs.append(c)
                else:
                    cat_covs.append(c)
        return num_covs, cat_covs

    def fit(self, X: pd.DataFrame, y=None):
        if not hasattr(X, "columns"):
            # fallback (array-like), do nothing
            self.feature_cols_ = [f"f{i}" for i in range(X.shape[1])]
            self.models_ = {}
            self.covar_ct_ = None
            return self

        self.feature_cols_ = list(X.columns)
        self.num_covars_, self.cat_covars_ = self._split_covars(X)

        transformers = []
        if self.num_covars_:
            transformers.append(
                ("cov_num", SimpleImputer(strategy="median"), self.num_covars_)
            )
        if self.cat_covars_:
            try:
                ohe = OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
            except TypeError:
                ohe = OneHotEncoder(handle_unknown="ignore", drop="first", sparse=False)
            transformers.append(
                ("cov_cat",
                 Pipeline([
                     ("impute", SimpleImputer(strategy="most_frequent")),
                     ("ohe", ohe)
                 ]),
                 self.cat_covars_)
            )

        if not transformers:
            self.models_.clear()
            self.covar_ct_ = None
            return self

        self.covar_ct_ = ColumnTransformer(
            transformers,
            remainder="drop",
            verbose_feature_names_out=False
        )

        C = self.covar_ct_.fit_transform(X)
        if hasattr(C, "toarray"):
            C = C.toarray()

        self.models_.clear()
        # Linear regression per numeric non-covariate feature
        for col in self.feature_cols_:
            if col in self.num_covars_ or col in self.cat_covars_:
                continue
            if not pd.api.types.is_numeric_dtype(X[col]):
                continue
            xi = pd.to_numeric(X[col], errors="coerce").to_numpy()
            mask = np.isfinite(xi)
            if mask.sum() < max(5, (C.shape[1] if C is not None else 0) + 1):
                continue
            lr = LinearRegression().fit(C[mask], xi[mask])
            self.models_[col] = lr

        return self

    def transform(self, X: pd.DataFrame):
        if self.covar_ct_ is None or not self.models_:
            return X
        Xr = X.copy()
        C = self.covar_ct_.transform(Xr)
        if hasattr(C, "toarray"):
            C = C.toarray()

        for col, lr in self.models_.items():
            if col in Xr.columns:
                xi = pd.to_numeric(Xr[col], errors="coerce").to_numpy()
                pred = lr.predict(C)
                finite = np.isfinite(xi)
                xi_resid = xi.copy()
                xi_resid[finite] = xi[finite] - pred[finite]
                Xr[col] = xi_resid
        return Xr


class CombinedPreprocessor(BaseEstimator, TransformerMixin):
    """
    - Residualize numeric features vs ALWAYS_KEEP covariates
    - Impute
    - Scale numeric
    - OHE categoricals
    - Ordinal-encode ordinal vars
    """
    def __init__(self, num_cols, cat_cols, ord_num, ord_cat, covariates, residualize=True):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.ord_num = ord_num
        self.ord_cat = ord_cat
        self.covariates = covariates
        self.residualize = residualize

        self._resid = None
        self._ct = None
        self.feature_names_out_ = None

    # Fit residualization (if enabled) and the column transformer
    # for all numeric/categorical/ordinal blocks.
    def fit(self, X, y=None):
        Xr = X
        if self.residualize:
            self._resid = ResidualizeCovariates(self.covariates)
            Xr = self._resid.fit_transform(X)

        def _ohe():
            try:
                return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            except TypeError:
                return OneHotEncoder(handle_unknown="ignore", sparse=False)

        num_block = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale",  StandardScaler(with_mean=True, with_std=True))
        ])
        cat_block = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", _ohe())
        ])
        ord_num_block = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent"))
        ])
        ord_cat_block = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ordenc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        ])

        self._ct = ColumnTransformer(
            transformers=[
                ("num",     num_block,  list(self.num_cols) if self.num_cols else []),
                ("cat",     cat_block,  list(self.cat_cols) if self.cat_cols else []),
                ("ord_num", ord_num_block, list(self.ord_num) if self.ord_num else []),
                ("ord_cat", ord_cat_block, list(self.ord_cat) if self.ord_cat else []),
            ],
            remainder="drop",
            verbose_feature_names_out=False
        )

        self._ct.fit_transform(Xr)
        try:
            self.feature_names_out_ = self._ct.get_feature_names_out()
        except Exception:
            self.feature_names_out_ = None

        return self

    # Apply residualization (if fitted) and transform through the column transformer.
    def transform(self, X):
        Xr = self._resid.transform(X) if self._resid is not None else X
        Xt = self._ct.transform(Xr) if self._ct is not None else Xr
        return Xt

class PTTabTransformer(BaseEstimator):
    """
    sklearn-compatible wrapper around pytorch-tabular TabTransformer.
    Expects a pandas DataFrame with original columns (no OHE). Handles:
      - train/valid split inside inner-CV folds as provided by the caller
      - predict_proba for integration with your calibration + threshold code
      - cost-sensitive option via pos_weight (maps from your imbalance mode)
    """
    def __init__(
        self,
        cat_cols: List[str],
        num_cols: List[str],
        random_state: int = 42,
        max_epochs: int = 80,
        batch_size: int = 32,
        lr: float = 1e-3,
        dropout: float = 0.1,
        n_layers: int = 3,
        n_heads: int = 4,
        embed_dim: int = 32,
        shared_embed: bool = True,
        attn_dropout: float = 0.1,
        weight_decay: float = 1e-4,
        patience: int = 10,
        imbalance_mode: str = "none",   # "none" or "cost_sensitive"
        device: str = "auto",           # "auto" | "cpu" | "cuda"
    ):
        self.cat_cols = list(cat_cols)
        self.num_cols = list(num_cols)
        self.random_state = random_state
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.shared_embed = shared_embed
        self.attn_dropout = attn_dropout
        self.weight_decay = weight_decay
        self.patience = patience
        self.imbalance_mode = imbalance_mode
        self.device = device

        self._model = None
        self._target_name = None
        self._cat_cardinalities = None

    # required by sklearn ParamSearch
    def get_params(self, deep=True):
        return {
            "cat_cols": self.cat_cols,
            "num_cols": self.num_cols,
            "random_state": self.random_state,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "dropout": self.dropout,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "embed_dim": self.embed_dim,
            "shared_embed": self.shared_embed,
            "attn_dropout": self.attn_dropout,
            "weight_decay": self.weight_decay,
            "patience": self.patience,
            "imbalance_mode": self.imbalance_mode,
            "device": self.device,
        }

    def set_params(self, **params):
        for k,v in params.items():
            setattr(self, k, v)
        return self

    # Infer torch device from configuration ("auto", "cpu", "cuda").
    def _infer_device(self):
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    # Fit TabTransformer through pytorch-tabular using the provided DataFrame.
    # Handles cost-sensitivity via pos_weight in the BCEWithLogits loss.
    def fit(self, X: pd.DataFrame, y: np.ndarray):
        assert isinstance(X, pd.DataFrame), "PTTabTransformer expects a pandas DataFrame"
        self._target_name = "__target__"
        df = X.copy()
        df[self._target_name] = y.astype(int)

        # pytorch-tabular data config
        data_config = DataConfig(
            target=self._target_name,
            continuous_cols=self.num_cols,
            categorical_cols=self.cat_cols
        )

        # cost-sensitive: pos_weight = Nneg/Npos mapped into loss kwargs
        loss_kwargs = None
        if self.imbalance_mode == "cost_sensitive":
            pos = int((y == 1).sum())
            neg = int((y == 0).sum())
            if pos > 0:
                pos_weight = float(neg / max(1, pos))
                loss_kwargs = {"pos_weight": torch.tensor([pos_weight], dtype=torch.float32)}
            else:
                loss_kwargs = None

        model_config = TabTransformerConfig(
            task="classification",
            metrics=["auroc"],
            embeddings_dropout=self.dropout,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            embedding_dim=self.embed_dim,
            share_embedding=self.shared_embed,
            attn_dropout=self.attn_dropout,
            loss="BCEWithLogitsLoss",
            loss_kwargs=loss_kwargs
        )

        trainer_config = TrainerConfig(
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            early_stopping="valid_loss",
            early_stopping_patience=self.patience,
            auto_lr_find=False,
            accelerator="auto" if self._infer_device()=="cuda" else "cpu",
            seed=self.random_state,
            check_val_every_n_epoch=1,
            enable_checkpointing=False,
            deterministic=True,
        )

        optimizer_config = OptimizerConfig(
            optimizer="AdamW",
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # Build and train
        self._model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
            verbose=0,
        )
        # pytorch-tabular splits train/val internally if not provided; we pass full df
        self._model.fit(train=df)
        return self

    # Return class probabilities as a (n_samples, 2) array, compatible with sklearn.
    def predict_proba(self, X: pd.DataFrame):
        assert self._model is not None, "Call fit first."
        df = X.copy()
        if self._target_name not in df.columns:
            df[self._target_name] = 0  # dummy column; not used for inference
        preds = self._model.predict(df)
        # 'prediction_probability' column is the positive-class prob
        p = preds["prediction_probability"].values.astype(float)
        p = np.clip(p, 1e-7, 1-1e-7)
        return np.c_[1.0 - p, p]

    # Default 0.5 threshold prediction from probabilities.
    def predict(self, X: pd.DataFrame):
        return (self.predict_proba(X)[:,1] >= 0.5).astype(int)
# ============================================================
# SafeSampler (SMOTE / ADASYN with guards for tiny minority class)
# ============================================================

class SafeSampler(BaseSampler):
    _sampling_type = "over-sampling"
    _parameter_constraints = {
        "mode": [StrOptions({"smote","adasyn"})],
        "random_state": [None, Integral],
        "min_minority": [Interval(Integral, 1, None, closed="left")],
        "knn_min": [Interval(Integral, 1, None, closed="left")],
        "knn_max": [Interval(Integral, 1, None, closed="left")],
        "sampling_strategy": [StrOptions({"auto"}), float, int, dict, None],
    }

    def __init__(
        self,
        mode="smote",
        random_state=42,
        min_minority=6,
        knn_min=1,
        knn_max=5,
        sampling_strategy="auto"
    ):
        self.mode = mode
        self.random_state = random_state
        self.min_minority = min_minority
        self.knn_min = knn_min
        self.knn_max = knn_max
        self.sampling_strategy = sampling_strategy

    # Core fit_resample logic with guards:
    # - Only apply SMOTE/ADASYN if minority count >= min_minority
    # - Choose k_neighbors within [knn_min, knn_max] and < minority size
    def _fit_resample(self, X, y):
        X, y = check_X_y(
            X,
            y,
            accept_sparse=["csr","csc"],
            force_all_finite="allow-nan"
        )
        y = np.asarray(y)
        self.sampling_strategy_ = self.sampling_strategy

        n_pos = int((y==1).sum())
        n_neg = int((y==0).sum())
        minority = min(n_pos, n_neg)

        # guard: if tiny minority, skip synthetic
        if minority < int(self.min_minority):
            return X, y

        k = max(int(self.knn_min), min(int(self.knn_max), minority-1))
        if k < 1:
            return X, y

        if (self.mode or "smote").lower() == "smote":
            sampler = SMOTE(
                random_state=self.random_state,
                k_neighbors=k,
                sampling_strategy=self.sampling_strategy_
            )
        else:
            sampler = ADASYN(
                random_state=self.random_state,
                n_neighbors=k,
                sampling_strategy=self.sampling_strategy_
            )

        return sampler.fit_resample(X, y)

    def _more_tags(self):
        return {"X_types":["2darray","sparse"], "allow_nan":True}

# ============================================================
# Calibration helpers
# ============================================================

# Simple calibrator wrapper that just forwards predict_proba (or approximates it).
# Used when we don't have enough data to fit a Platt-scaled calibrator.
class IdentityCalibrator:
    def __init__(self, fitted_pipeline):
        self.pipe = fitted_pipeline
    def fit(self, X, y):
        return self
    def predict_proba(self, X):
        if hasattr(self.pipe, "predict_proba"):
            return self.pipe.predict_proba(X)
        if hasattr(self.pipe, "decision_function"):
            z = self.pipe.decision_function(X)
            p = 1.0 / (1.0 + np.exp(-z))
            return np.c_[1-p, p]
        yhat = self.pipe.predict(X)
        p = yhat.astype(float)
        return np.c_[1-p, p]

# Factory for a safe calibrator:
# - If enough positives/negatives: use CalibratedClassifierCV (Platt scaling).
# - Else, use IdentityCalibrator (no change).
def build_safe_calibrator(fitted_pipe, X_tr, y_tr):
    """
    Platt scaling if we have enough positives/negatives,
    else identity mapping.
    """
    pos = int((y_tr==1).sum())
    neg = int((y_tr==0).sum())
    if pos >= CFG.CAL_MIN_POS and neg >= CFG.CAL_MIN_NEG:
        try:
            calib = CalibratedClassifierCV(
                fitted_pipe,
                method="sigmoid",
                cv="prefit"
            )
            calib.fit(X_tr, y_tr)
            return calib
        except Exception:
            return IdentityCalibrator(fitted_pipe)
    else:
        return IdentityCalibrator(fitted_pipe)

# Build reliability curve data for calibration plots:
# - Bins predicted probabilities and reports observed label prevalence per bin.
def calibration_curve_data(y_true, y_prob, n_bins=10):
    """
    Reliability curve:
    - bin predicted probs
    - compute observed positive rate in each bin
    Returns dict of bin centers, observed rate, counts.
    """
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    idx = np.digitize(y_prob, bins) - 1
    bin_centers = []
    obs_rate = []
    counts = []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            bin_centers.append(float((bins[b]+bins[b+1])/2.0))
            obs_rate.append(np.nan)
            counts.append(0)
        else:
            bin_centers.append(float((bins[b]+bins[b+1])/2.0))
            counts.append(int(mask.sum()))
            obs_rate.append(float(np.mean(y_true[mask])))
    return {
        "bin_centers": bin_centers,
        "obs_rate": obs_rate,
        "counts": counts,
        "bins": bins.tolist()
    }

# ============================================================
# Data load / curation
# ============================================================

# Load raw data from Excel and restrict to rows with valid binary target.
assert os.path.exists(CFG.DATA_PATH), f"File not found: {CFG.DATA_PATH}"
df_raw = pd.read_excel(CFG.DATA_PATH)
df_raw.columns = [str(c).strip() for c in df_raw.columns]
assert CFG.TARGET_NAME in df_raw.columns, f"Target {CFG.TARGET_NAME!r} not in columns."

y_all = to_binary(df_raw[CFG.TARGET_NAME])
keep_mask = y_all.notna()
df0 = df_raw.loc[keep_mask].copy()
y_final = y_all.loc[keep_mask].astype(int).values

must_remove = set()

# Heuristic: detect ID-like columns by name and uniqueness (high cardinality).
def is_id_like(colname: str, s: pd.Series) -> bool:
    name_match = bool(
        re.search(r"\b(id|code|patient|record|mrn|subject|study)\b", colname, re.I)
    )
    nunq = s.nunique(dropna=True)
    high_unique = nunq >= 0.9 * len(s)
    return name_match or high_unique

# Heuristic: detect date/time-like columns by name and successful datetime parsing.
def is_date_like(colname: str, s: pd.Series) -> bool:
    if re.search(r"(date|time|datetime|timestamp|dob|injury\s*date|discharge|follow[- ]?up)", colname, re.I):
        p = pd.to_datetime(s, errors="coerce", dayfirst=True, infer_datetime_format=True)
        return p.notna().any()
    return False

# Pass over all columns:
# - Drop identifiers, dates, obvious leakage, high-missing, constant or NZV features.
for c in df0.columns:
    if c == CFG.TARGET_NAME:
        continue
    if is_id_like(c, df0[c]):
        must_remove.add(c)
    try:
        if is_date_like(c, df0[c]):
            must_remove.add(c)
    except Exception:
        pass

    cl = c.lower()
    if re.search(r"(tendon|tendin|injur|diagnos|treat|therapy|rehab|rupture|severity|outcome)", cl):
        must_remove.add(c)

missing_rate = df0.isnull().mean()
for c in df0.columns:
    if c == CFG.TARGET_NAME:
        continue
    if missing_rate[c] > CFG.MISSING_THRESHOLD:
        must_remove.add(c)
    nunq = df0[c].nunique(dropna=True)
    if nunq <= 1:
        must_remove.add(c)
    else:
        freq = df0[c].value_counts(dropna=True, normalize=True)
        if len(freq) <= 5 and freq.iloc[0] > CFG.NZV_MAJORITY:
            must_remove.add(c)

pool = [c for c in df0.columns if c not in must_remove and c != CFG.TARGET_NAME]

# Preferred candidate features, in priority order, specific to the domain.
preferred = [
    "Age","Sex",
    "Height (cm)","Weight (Kg)",
    "Range of Motion - Ankle Dorsiflexion Knee Flexed",
    "Range of Motion - Ankle Dorsiflexion Knee Extended",
    "Range of Motion - Hip Internal Rotation",
    "Range of Motion - Hip External Rotation",
    "Ybalance Composite Score Dominant","Ybalance Composite Score Non-Dominant",
    "During the recent past, how many hours of actual sleep did you get at night?",
    "How satisfied/dissatisfied are you with the quality of your sleep?",
    "SDS","Reduced Sense of Accomplishment","Emotional and Physical Exhaustion","Sport Devaluation"
]
extras = [
    "YBalance-Anterior - Dominant",
    "YBalance-Anterior - Non-dominant"
]

final_cols = [
    c for c in preferred if c in pool
] + [
    c for c in extras if c in pool and c not in preferred
]

X_cur = df0[final_cols].copy()

num_cols = [c for c in final_cols if pd.api.types.is_numeric_dtype(X_cur[c])]
cat_cols = [c for c in final_cols if c not in num_cols and X_cur[c].dtype == "object"]
ord_num, ord_cat = [], []

n_rows = X_cur.shape[0]
n_feat = X_cur.shape[1]
pos = int((y_final==1).sum())
neg = int((y_final==0).sum())
prev = pos / max(1, n_rows)

print("\n================ DATA CURATION SUMMARY (LEAN) ================")
print(f"Target: {CFG.TARGET_NAME!r}")
print(f"Rows: {n_rows} (positives={pos}, negatives={neg}, prevalence={prev:.3f})")
print(f"Final features (lean): {n_feat}")
for i,c in enumerate(final_cols,1):
    print(f"{i:>3}. {c}")

out_dir = Path(os.path.join(CFG.SAVE_DIR, "curation_artifacts"))
out_dir.mkdir(parents=True, exist_ok=True)
pd.Series(final_cols, name="final_features").to_csv(out_dir/"final_features.csv", index=False)
pd.Series(sorted(list(must_remove)), name="must_remove").to_csv(out_dir/"must_remove.csv", index=False)
missing_rate.sort_values(ascending=False).to_csv(out_dir/"missing_rate_overall.csv")
print(f"\n[Saved] final_features.csv & must_remove.csv -> {out_dir}")
print("==============================================================\n")

X_full, y_full = X_cur, y_final

# ============================================================
# Preprocessor factory
# ============================================================

# Convenience function to build a CombinedPreprocessor for the global feature sets.
def build_preprocessor(residualize: bool = True) -> CombinedPreprocessor:
    return CombinedPreprocessor(
        num_cols=num_cols,
        cat_cols=cat_cols,
        ord_num=ord_num,
        ord_cat=ord_cat,
        covariates=list(CFG.ALWAYS_KEEP),
        residualize=residualize
    )

# ============================================================
# Classifier factory + grids
# ============================================================

# Create a dictionary of model definitions:
# - Each entry has: estimator, hyperparameter grid, flags for coef and tree-ensemble,
#   and whether it must only be used with internally balanced mode.
def classifier_search_grid_factory(cfg: Config) -> Dict[str, Dict[str, Any]]:
    """
    Returns dict keyed by classifier label.
    Each value:
      - estimator
      - grid
      - supports_coef
      - is_tree_ensemble
      - internally_balanced_only
    """
    models = {}

    # Logistic Regression (L2)
    lr_l2 = LogisticRegression(
        max_iter=3000,
        solver="saga",
        penalty="l2",
        random_state=cfg.RANDOM_STATE
    )
    models["LogReg_L2"] = {
        "estimator": lr_l2,
        "grid": {"clf__C": cfg.C_GRID},
        "supports_coef": True,
        "is_tree_ensemble": False,
        "internally_balanced_only": False
    }

    # Logistic Regression (Elastic Net)
    lr_en = LogisticRegression(
        max_iter=3000,
        solver="saga",
        penalty="elasticnet",
        l1_ratio=0.5,
        random_state=cfg.RANDOM_STATE
    )
    models["LogReg_EN"] = {
        "estimator": lr_en,
        "grid": {
            "clf__C": cfg.C_GRID,
            "clf__l1_ratio": cfg.EN_L1R_GRID
        },
        "supports_coef": True,
        "is_tree_ensemble": False,
        "internally_balanced_only": False
    }

    # Logistic Regression (L1)
    lr_l1 = LogisticRegression(
        max_iter=3000,
        solver="saga",
        penalty="l1",
        random_state=cfg.RANDOM_STATE
    )
    models["LogReg_L1"] = {
        "estimator": lr_l1,
        "grid": {
            "clf__C": tuple(np.logspace(-3, 2, 10))
        },
        "supports_coef": True,
        "is_tree_ensemble": False,
        "internally_balanced_only": False
    }

    # SVM RBF
    svc = SVC(
        kernel="rbf",
        probability=True,
        random_state=cfg.RANDOM_STATE
    )
    models["SVM_RBF"] = {
        "estimator": svc,
        "grid": {
            "clf__C": tuple(np.logspace(-2, 2, 8)),
            "clf__gamma": tuple(np.logspace(-4, 0, 8)),
        },
        "supports_coef": False,
        "is_tree_ensemble": False,
        "internally_balanced_only": False
    }

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=400,
        random_state=cfg.RANDOM_STATE,
        n_jobs=-1
    )
    models["RF"] = {
        "estimator": rf,
        "grid": {
            "clf__n_estimators": [200, 400, 600],
            "clf__max_depth": [None, 3, 5, 8],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
        },
        "supports_coef": False,
        "is_tree_ensemble": True,
        "internally_balanced_only": False
    }

    # Balanced Random Forest
    brf = BalancedRandomForestClassifier(
        n_estimators=400,
        random_state=cfg.RANDOM_STATE,
        n_jobs=-1
    )
    models["BalancedRF"] = {
        "estimator": brf,
        "grid": {
            "clf__n_estimators": [200, 400, 600],
            "clf__max_depth": [None, 3, 5, 8],
        },
        "supports_coef": False,
        "is_tree_ensemble": True,
        "internally_balanced_only": True
    }

    # EasyEnsemble
    eec = EasyEnsembleClassifier(
        n_estimators=10,
        random_state=cfg.RANDOM_STATE,
        n_jobs=-1
    )
    models["EasyEnsemble"] = {
        "estimator": eec,
        "grid": {
            "clf__n_estimators": [6, 10, 14],
        },
        "supports_coef": False,
        "is_tree_ensemble": True,
        "internally_balanced_only": True
    }

    # XGBoost
    if XGBClassifier is not None:
        xgb = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            random_state=cfg.RANDOM_STATE,
            n_estimators=400,
            n_jobs=-1,
            tree_method="hist",
        )
        models["XGB"] = {
            "estimator": xgb,
            "grid": {
                "clf__max_depth": [3, 4, 5],
                "clf__learning_rate": [0.02, 0.05, 0.1],
                "clf__subsample": [0.7, 0.9, 1.0],
                "clf__colsample_bytree": [0.7, 0.9, 1.0],
                "clf__min_child_weight": [1, 5, 10],
            },
            "supports_coef": False,
            "is_tree_ensemble": True,
            "internally_balanced_only": False
        }

    # LightGBM
    if LGBMClassifier is not None:
        lgb = LGBMClassifier(
            objective="binary",
            n_estimators=600,
            random_state=cfg.RANDOM_STATE,
            verbose=-1,
            device="cpu",
        )
        models["LGBM"] = {
            "estimator": lgb,
            "grid": {
                "clf__num_leaves": [7, 15, 31],
                "clf__max_depth": [-1, 3, 5],
                "clf__learning_rate": [0.02, 0.05, 0.1],
                "clf__subsample": [0.7, 0.9, 1.0],
                "clf__colsample_bytree": [0.7, 0.9, 1.0],
                "clf__min_child_samples": [5, 10, 20],
                "clf__min_split_gain": [0.0, 0.01],
            },
            "supports_coef": False,
            "is_tree_ensemble": True,
            "internally_balanced_only": False
        }

    # CatBoost
    if CatBoostClassifier is not None:
        cb = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=cfg.RANDOM_STATE,
            verbose=False,
            iterations=600,
            task_type="CPU",
        )
        models["CatBoost"] = {
            "estimator": cb,
            "grid": {
                "clf__depth": [4, 6, 8],
                "clf__learning_rate": [0.02, 0.05, 0.1],
                "clf__l2_leaf_reg": [1, 3, 5, 7],
            },
            "supports_coef": False,
            "is_tree_ensemble": True,
            "internally_balanced_only": False
        }

    # TabNet
    if TabNetClassifier is not None:
        tnb = TabNetClassifier(
            verbose=0,
            seed=cfg.RANDOM_STATE,
            device_name="cpu"
        )
        models["TabNet"] = {
            "estimator": tnb,
            "grid": {
                "clf__n_d": [8, 16],
                "clf__n_a": [8, 16],
                "clf__n_steps": [3, 5],
            },
            "supports_coef": False,
            "is_tree_ensemble": False,
            "internally_balanced_only": False
        }
        # TabTransformer (only if available)
    if TABTRANSFORMER_AVAILABLE:
        ttx = PTTabTransformer(
            cat_cols=cat_cols,
            num_cols=num_cols,
            random_state=cfg.RANDOM_STATE
        )
        models["TabTransformer"] = {
            "estimator": ttx,
            "grid": {
                # You can tune a few light params; keep it small to start
                "clf__max_epochs": [60, 100],
                "clf__batch_size": [16, 32],
                "clf__lr": [1e-3, 5e-4],
                "clf__dropout": [0.05, 0.1],
                "clf__n_layers": [2, 3],
                "clf__n_heads": [2, 4],
                "clf__embed_dim": [16, 32],
                "clf__attn_dropout": [0.05, 0.1],
                "clf__patience": [8, 12],
                # imbalance mode is controlled via pipeline; see build_pipeline below
            },
            "supports_coef": False,
            "is_tree_ensemble": False,
            "internally_balanced_only": False
        }
    return models

# imbalance strategies for externally-balanced classifiers
IMB_STRATEGIES = [
    "none",
    "cost_sensitive",
    "smote",
    "adasyn",
    "smoteenn",
    "undersample"
]

# Generate a list of (model_name, estimator, grid, imbalance_mode, meta_info)
# including all classifier x imbalance combinations to be evaluated.
def make_model_specs(cfg: Config):
    specs = []
    clf_defs = classifier_search_grid_factory(cfg)

    for clf_label, info in clf_defs.items():
        base_est = info["estimator"]
        grid = info["grid"]
        internally_balanced_only = info["internally_balanced_only"]

        if internally_balanced_only:
            specs.append((clf_label, clone(base_est), deepcopy(grid), "internal_balance", info))
        else:
            # If TabTransformer: restrict to ["none", "cost_sensitive"]
            if isinstance(base_est, PTTabTransformer):
                for imb_mode in ["none", "cost_sensitive"]:
                    specs.append((f"{clf_label}__{imb_mode.upper()}",
                                  clone(base_est), deepcopy(grid), imb_mode, info))
            else:
                for imb_mode in IMB_STRATEGIES:
                    specs.append((f"{clf_label}__{imb_mode.upper()}",
                                  clone(base_est), deepcopy(grid), imb_mode, info))
    return specs

# ============================================================
# Pipeline builder with imbalance modes
# ============================================================

# Build an imblearn Pipeline for a given imbalance_mode and estimator:
# - For TabTransformer, bypass CombinedPreprocessor and samplers
# - For classical models, prepend CombinedPreprocessor and optionally add
#   SafeSampler / SMOTEENN / RandomUnderSampler or class_weight.
def build_pipeline(imbalance_mode: str, estimator, cfg: Config) -> ImbPipeline:
    # Special-case: our PTTabTransformer wrapper consumes the raw DataFrame (no OHE/scaling)
    if isinstance(estimator, PTTabTransformer):
        est = clone(estimator)
        # map external imbalance modes -> internal loss weighting
        if imbalance_mode in ("cost_sensitive",):
            est.set_params(imbalance_mode="cost_sensitive")
        else:
            est.set_params(imbalance_mode="none")

        # No imblearn samplers, no CombinedPreprocessor
        return ImbPipeline([("clf", est)])

    # ---- default path for all classical models (your existing code) ----
    preproc = build_preprocessor(residualize=True)
    steps = [("prep", preproc)]

    if imbalance_mode == "none":
        steps.append(("clf", estimator))

    elif imbalance_mode == "cost_sensitive":
        est = clone(estimator)
        if hasattr(est, "get_params"):
            params = est.get_params()
            if "class_weight" in params:
                est.set_params(class_weight="balanced")
        steps.append(("clf", est))

    elif imbalance_mode == "smote":
        steps += [
            ("safe_smote", SafeSampler(
                mode="smote",
                random_state=cfg.RANDOM_STATE,
                min_minority=cfg.MIN_MINORITY_FOR_SYNTH,
                knn_min=cfg.KNN_MIN,
                knn_max=cfg.KNN_MAX
            )),
            ("clf", estimator)
        ]

    elif imbalance_mode == "adasyn":
        steps += [
            ("safe_adasyn", SafeSampler(
                mode="adasyn",
                random_state=cfg.RANDOM_STATE,
                min_minority=cfg.MIN_MINORITY_FOR_SYNTH,
                knn_min=cfg.KNN_MIN,
                knn_max=cfg.KNN_MAX
            )),
            ("clf", estimator)
        ]

    elif imbalance_mode == "smoteenn":
        steps += [
            ("smoteenn", SMOTEENN(random_state=cfg.RANDOM_STATE)),
            ("clf", estimator)
        ]

    elif imbalance_mode == "undersample":
        steps += [
            ("under", RandomUnderSampler(random_state=cfg.RANDOM_STATE)),
            ("clf", estimator)
        ]

    elif imbalance_mode == "internal_balance":
        # BalancedRF / EasyEnsemble does balancing internally
        steps.append(("clf", estimator))

    else:
        steps.append(("clf", estimator))

    return ImbPipeline(steps)

# ============================================================
# Metrics at chosen threshold
# ============================================================

# Compute confusion-matrix-based metrics for a given threshold:
# - Sensitivity, Specificity, BAC, PPV, NPV, and counts (TP,FP,TN,FN).
def metrics_at_thr(y_true, y_prob, thr) -> Dict[str, float]:
    """
    Metrics using the chosen tuned threshold thr_star from inner CV.
    """
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn/(tn+fp) if (tn+fp)>0 else 0.0
    sens = tp/(tp+fn) if (tp+fn)>0 else 0.0
    bac  = 0.5*(sens+spec)
    ppv  = tp/(tp+fp) if (tp+fp)>0 else 0.0
    npv  = tn/(tn+fn) if (tn+fn)>0 else 0.0
    return {
        "Sensitivity_T": float(sens),
        "Specificity_T": float(spec),
        "BAC_T": float(bac),
        "PPV_T": float(ppv),
        "NPV_T": float(npv),
        "Thr": float(thr),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn)
    }

# ============================================================
# Checkpointing / paths
# ============================================================

# Base directory for all run outputs (CSV/JSON/artifacts).
def runs_dir() -> str:
    base = os.path.join(CFG.SAVE_DIR, CFG.RESULTS_DIRNAME)
    os.makedirs(base, exist_ok=True)
    return base

# Path to the per-fold checkpoint JSON for a given (model,imbalance,fold).
def fold_checkpoint_path(model_name: str, imb_mode: str, fold_idx: int) -> str:
    return os.path.join(
        runs_dir(),
        f"{model_name}__{imb_mode}__fold{fold_idx}.json"
    )

# Path for the per-(model,imbalance) CSV with fold-level metrics.
def fold_csv_path(model_name: str, imb_mode: str) -> str:
    return os.path.join(
        runs_dir(),
        f"{model_name}__{imb_mode}__fold_metrics.csv"
    )

# Path for the global summary CSV across all models/imbalance modes.
def summary_csv_path() -> str:
    return os.path.join(runs_dir(), "summary_by_model.csv")

# Directory to store per-fold artifacts (model pickle, SHAP tables, coef tables, etc.).
def model_artifact_dir(model_name: str, imb_mode: str, fold_idx: int) -> str:
    """
    Directory for per-fold sidecars (calibration curve, SHAP summaries, coef tables, model pickle).
    """
    d = os.path.join(
        runs_dir(),
        f"{model_name}__{imb_mode}",
        f"fold{fold_idx}"
    )
    os.makedirs(d, exist_ok=True)
    return d

# Append one row of metrics to the per-(model,imbalance) fold CSV, creating it if needed.
def append_fold_row_to_csv(model_name: str, imb_mode: str, row: Dict[str, Any]):
    path = fold_csv_path(model_name, imb_mode)
    df_row = pd.DataFrame([row])
    if os.path.exists(path):
        df_row.to_csv(path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(path, index=False)

# ============================================================
# Permutation baseline AUC
# ============================================================

# Compute a permutation-based null distribution for ROC AUC:
# - Shuffle y labels n_perm times while reusing predicted probabilities.
# - Return observed AUC and (2.5, 97.5) percentiles of the null.
def permutation_baseline_auc(y_true, y_prob, n_perm=CFG.N_PERMUTATIONS, rng=None):
    """
    Shuffle labels n_perm times and compute AUC to get null distribution.
    Returns:
      obs_auc,
      (lo, hi) ~ 2.5th/97.5th of permutation null
    """
    rng = rng or np.random.RandomState(123)
    if len(np.unique(y_true)) < 2:
        return np.nan, (np.nan, np.nan)
    try:
        obs = roc_auc_score(y_true, y_prob)
    except Exception:
        return np.nan, (np.nan, np.nan)

    null = []
    for _ in range(n_perm):
        y_shuf = rng.permutation(y_true)
        try:
            null.append(roc_auc_score(y_shuf, y_prob))
        except Exception:
            pass

    if not null:
        return obs, (np.nan, np.nan)

    lo, hi = np.percentile(null, [2.5, 97.5])
    return obs, (lo, hi)

# ============================================================
# Inner CV (hyperparam search + threshold fit)
# ============================================================

# Inner CV routine:
# - Builds pipeline for a given estimator + imbalance.
# - Runs RandomizedSearchCV (ROC AUC scoring).
# - Collects OOF calibrated probabilities across inner folds.
# - Learns tuned threshold using the configured policy.
def inner_cv(
    X_tr,
    y_tr,
    estimator,
    grid,
    imb_mode,
    random_state,
    primary_mode: str
):
    """
    - build pipeline
    - RandomizedSearchCV on inner CV
    - collect OOF calibrated probs
    - fit guarded threshold policy
    """
    inner = StratifiedKFold(
        n_splits=CFG.INNER_FOLDS,
        shuffle=True,
        random_state=random_state
    )

    pipe = build_pipeline(imb_mode, estimator, CFG)

    search = RandomizedSearchCV(
        pipe,
        param_distributions=(grid or {}),
        n_iter=(CFG.N_TRIALS if grid else 1),
        scoring="roc_auc",
        cv=inner,
        n_jobs=-1,
        random_state=random_state,
        refit=True,
        verbose=0
    )
    search.fit(X_tr, y_tr)

    best_params = search.best_params_
    inner_auc = search.best_score_

    # OOF probs for threshold learning
    oof_prob = np.zeros_like(y_tr, dtype=float)
    for tr_idx, va_idx in inner.split(X_tr, y_tr):
        X_i_tr, X_i_va = X_tr.iloc[tr_idx], X_tr.iloc[va_idx]
        y_i_tr, y_i_va = y_tr[tr_idx], y_tr[va_idx]

        pipe_fold = build_pipeline(imb_mode, clone(estimator), CFG)
        if best_params:
            pipe_fold.set_params(**best_params)
        pipe_fold.fit(X_i_tr, y_i_tr)

        calib = build_safe_calibrator(pipe_fold, X_i_tr, y_i_tr)
        oof_prob[va_idx] = calib.predict_proba(X_i_va)[:, 1]

    # learn threshold
    thr_star, thr_info = find_threshold(y_tr, oof_prob, mode=primary_mode)
    return best_params, search.best_estimator_, inner_auc, thr_star, thr_info

    # ============================================================
# Helper: summary stats with mean/std/CI
# ============================================================

# Utility for model-level aggregation:
# - Compute mean, std, and normal-approx 95% CI for a metric series.
def _mean_std_ci(arr: pd.Series) -> Dict[str, float]:
    """
    Compute mean, std, and 95% CI (normal approx) for a metric array.
    If N<=1 -> std=nan, CI_lo/hi=nan.
    """
    a = pd.to_numeric(arr, errors="coerce").dropna()
    n = len(a)
    if n == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "ci95_lo": np.nan,
            "ci95_hi": np.nan
        }
    mu = float(np.mean(a))
    sd = float(np.std(a, ddof=1)) if n>1 else np.nan
    if n>1 and sd==sd:
        half_width = 1.96 * (sd/np.sqrt(n))
        lo = mu - half_width
        hi = mu + half_width
    else:
        lo = np.nan
        hi = np.nan
    return {
        "mean": mu,
        "std": sd,
        "ci95_lo": lo,
        "ci95_hi": hi
    }

# ============================================================
# Outer CV with diagnostics, SHAP export, etc.
# ============================================================

# Outer CV driver:
# - RepeatedStratifiedKFold over the full dataset.
# - Loops over all (model,imbalance) specs.
# - For each fold: runs inner_cv, refits, calibrates, evaluates, exports artifacts.
# - After all folds: creates fold-level and summary CSVs.
def outer_cv(X: pd.DataFrame, y: np.ndarray):

    outer = RepeatedStratifiedKFold(
        n_splits=CFG.OUTER_FOLDS,
        n_repeats=CFG.OUTER_REPEATS,
        random_state=CFG.RANDOM_STATE
    )
    split_list = list(outer.split(X, y))
    n_total_folds = len(split_list)

    specs = make_model_specs(CFG)

    # Record which combos are run
    with open(os.path.join(runs_dir(), "specs.json"), "w") as f:
        json.dump(
            [
                (model_name, imb_mode)
                for (model_name, _, __, imb_mode, ___) in specs
            ],
            f,
            indent=2
        )

    all_fold_rows = []
    summary_rows = []

    print(f"\n[INFO] Repeated outer CV: {CFG.OUTER_FOLDS}x{CFG.OUTER_REPEATS}={n_total_folds} folds;"
          f" specs={len(specs)} combos")

    for (model_name, est, grid, imb_mode, meta) in tqdm(specs, desc="Specs", leave=True):
        print(f"\n=== RUNNING MODEL: {model_name} | imbalance_mode={imb_mode} ===")
        model_fold_rows = []

        for fold_idx, (tr, te) in enumerate(
            tqdm(split_list, desc=f"{model_name}", leave=False), start=1
        ):
            cp_path = fold_checkpoint_path(model_name, imb_mode, fold_idx)

            # Resume safety: if checkpoint row exists, reuse it
            if os.path.exists(cp_path):
                try:
                    with open(cp_path, "r") as f:
                        saved = json.load(f)
                    row = saved.get("row", None)
                    if row is not None:
                        print(
                            f"[FOLD {fold_idx}] RESUME ✓ "
                            f"BAC_T={row.get('BAC_T','NA'):.3f} "
                            f"Sens={row.get('Sensitivity_T','NA'):.3f} "
                            f"Spec={row.get('Specificity_T','NA'):.3f}"
                        )
                        model_fold_rows.append(row)
                        all_fold_rows.append(row)
                        continue
                except Exception as e:
                    print(f"[FOLD {fold_idx}] Resume failed -> recomputing. Err={e}")

            # fresh run
            X_tr, X_te = X.iloc[tr].copy(), X.iloc[te].copy()
            y_tr, y_te = y[tr], y[te]

            prev_tr = float((y_tr==1).mean())
            prev_te = float((y_te==1).mean())

            print(
                f"[FOLD {fold_idx}] Train prev={prev_tr:.3f} (n={len(y_tr)}) | "
                f"Test prev={prev_te:.3f} (n={len(y_te)}) | "
                f"imb_mode={imb_mode}"
            )

            t0 = time.time()
            try:
                (
                    best_params,
                    best_estimator,
                    inner_auc,
                    thr_star,
                    thr_info
                ) = inner_cv(
                    X_tr,
                    y_tr,
                    est,
                    grid,
                    imb_mode,
                    random_state=rng_global.randint(0, 10**6),
                    primary_mode=CFG.THRESH_PRIMARY
                )

                # Fit best pipeline on full outer-train
                best_pipe = build_pipeline(imb_mode, clone(est), CFG)
                if best_params:
                    best_pipe.set_params(**best_params)
                best_pipe.fit(X_tr, y_tr)

                # Calibrate
                calib = build_safe_calibrator(best_pipe, X_tr, y_tr)

                # Predict calibrated probs
                y_prob_te = calib.predict_proba(X_te)[:, 1]
                y_prob_tr = calib.predict_proba(X_tr)[:, 1]

                # Threshold metrics on outer-test
                tuned = metrics_at_thr(y_te, y_prob_te, thr_star)

                print(
                    f"[FOLD {fold_idx}] tuned@{thr_star:.3f} {thr_info.get('mode','?')} "
                    f"BAC_T={tuned['BAC_T']:.3f} "
                    f"Sens={tuned['Sensitivity_T']:.3f} "
                    f"Spec={tuned['Specificity_T']:.3f} "
                    f"| CM(TP={tuned['TP']},FP={tuned['FP']},TN={tuned['TN']},FN={tuned['FN']})"
                )

                # Default-threshold (0.5) metrics for reference
                metrics_05 = safe_metric_pack(y_te, y_prob_te, thr=0.5)
                elapsed = round(time.time() - t0, 2)

                # Permutation baseline AUC on test
                obs_auc, (auc_lo, auc_hi) = permutation_baseline_auc(
                    y_te,
                    y_prob_te,
                    n_perm=CFG.N_PERMUTATIONS,
                    rng=rng_global
                )

                # Reliability / calibration curve data on test
                cal_curve = calibration_curve_data(
                    y_te,
                    y_prob_te,
                    n_bins=CFG.CAL_CURVE_BINS
                )


                               # =====================================================
                # Artifact dir for this (model,imbalance,fold)
                # =====================================================
                fold_dir = os.path.join(
                    runs_dir(),
                    f"{model_name}__{imb_mode}",
                    f"fold{fold_idx}"
                )
                ensure_dir(fold_dir)

                # =====================================================
                # Persist calibrated model (for downstream audit / deploy)
                # =====================================================
                model_pickle_path = os.path.join(fold_dir, "calibrated_model.pkl")
                with open(model_pickle_path, "wb") as fmdl:
                    pickle.dump({
                        "calibrator": calib,
                        "best_pipe": best_pipe,
                        "best_params": best_params,
                        "thr_star": thr_star,
                        "thr_info": thr_info
                    }, fmdl)

                # =====================================================
                # Interpretability exports
                #   - Logistic regression coefficients
                #   - SHAP (tree/linear/kernel fallback)
                #   - Feature importances fallback
                # =====================================================

                # ----- prep common things -----
                prep_step = best_pipe[0]    # "prep"
                clf_final = best_pipe[-1]   # final classifier

                # feature names from preprocessor
                if hasattr(prep_step, "feature_names_out_"):
                    try:
                        feat_names = list(prep_step.feature_names_out_)
                    except Exception:
                        feat_names = None
                else:
                    feat_names = None

                # -----------------
                # Logistic coefs
                # -----------------
                coef_json = None
                Coef_Export_Success = False
                try:
                    if meta.get("supports_coef", False) and hasattr(clf_final, "coef_"):
                        coefs = np.asarray(clf_final.coef_).ravel()
                        coef_df = pd.DataFrame({
                            "feature": (
                                feat_names if feat_names is not None
                                else [f"f{i}" for i in range(len(coefs))]
                            ),
                            "coef": coefs
                        })
                        coef_csv_path = os.path.join(fold_dir, "logreg_coefficients.csv")
                        coef_df.to_csv(coef_csv_path, index=False)

                        coef_json = {
                            "coef_table_path": coef_csv_path,
                            "n_features": int(len(coef_df)),
                            "note": "SE/CI not yet estimated"
                        }
                        Coef_Export_Success = True
                except Exception:
                    coef_json = None
                    Coef_Export_Success = False

                # -----------------
                # SHAP / FI
                # -----------------
                shap_export_success = False
                shap_top15_json = None           # holds path to CSV + top15
                top_shap_features = None         # list of dicts [{"feature":..., "mean_abs_shap":...}, ...]
                feature_importance_json = None

                # try SHAP first
                if SHAP_AVAILABLE:
                    try:
                        # preprocess outer-test features
                        X_te_pre = prep_step.transform(X_te)

                        # pick explainer type
                        explainer = None
                        shap_vals = None

                        if meta.get("is_tree_ensemble", False):
                            # Tree explainer
                            explainer = shap.TreeExplainer(clf_final)
                            shap_vals = explainer.shap_values(X_te_pre)
                        elif meta.get("supports_coef", False):
                            # Linear explainer for logistic-style models
                            explainer = shap.LinearExplainer(
                                clf_final,
                                X_te_pre,
                                feature_perturbation="correlation_dependent"
                            )
                            shap_vals = explainer.shap_values(X_te_pre)
                        else:
                            # Generic fallback: KernelExplainer
                            # Build probability function
                            if hasattr(clf_final, "predict_proba"):
                                def _pred_fn(Z):
                                    return clf_final.predict_proba(Z)[:, 1]
                            elif hasattr(clf_final, "decision_function"):
                                def _pred_fn(Z):
                                    z = clf_final.decision_function(Z)
                                    return 1.0 / (1.0 + np.exp(-z))
                            else:
                                def _pred_fn(Z):
                                    return clf_final.predict(Z).astype(float)

                            # small background to keep runtime sane
                            X_te_pre_bg = shap.sample(
                                X_te_pre,
                                min(200, X_te_pre.shape[0]),
                                random_state=CFG.RANDOM_STATE
                            )
                            explainer = shap.KernelExplainer(_pred_fn, X_te_pre_bg)

                            X_te_pre_eval = X_te_pre[:min(200, X_te_pre.shape[0])]
                            shap_vals = explainer.shap_values(X_te_pre_eval)

                        # normalize binary output
                        if isinstance(shap_vals, list) and len(shap_vals) == 2:
                            shap_vals_use = shap_vals[1]
                        else:
                            shap_vals_use = shap_vals

                        if shap_vals_use is not None:
                            mean_abs = np.mean(np.abs(shap_vals_use), axis=0)

                            # feature names fallback
                            if feat_names is None:
                                feat_names_use = [f"f{i}" for i in range(len(mean_abs))]
                            else:
                                feat_names_use = feat_names

                            shap_df = pd.DataFrame({
                                "feature": feat_names_use,
                                "mean_abs_shap": mean_abs
                            }).sort_values("mean_abs_shap", ascending=False)

                            shap_csv_path = os.path.join(fold_dir, "shap_mean_abs.csv")
                            shap_df.to_csv(shap_csv_path, index=False)

                            # top 15 for audit
                            top15_records = shap_df.head(15).to_dict(orient="records")
                            top_shap_features = top15_records
                            shap_top15_json = {
                                "top15": top15_records,
                                "shap_csv_path": shap_csv_path
                            }
                            shap_export_success = True

                    except Exception:
                        shap_export_success = False
                        shap_top15_json = None
                        top_shap_features = None

                # fallback to feature_importances_ if SHAP failed and it's a tree-ish model
                if (not shap_export_success) and meta.get("is_tree_ensemble", False) and hasattr(clf_final, "feature_importances_"):
                    try:
                        fi_vals = np.asarray(clf_final.feature_importances_)
                        if feat_names is None:
                            fi_names = [f"f{i}" for i in range(len(fi_vals))]
                        else:
                            fi_names = feat_names
                        fi_df = pd.DataFrame({
                            "feature": fi_names,
                            "importance": fi_vals
                        }).sort_values("importance", ascending=False)

                        fi_csv_path = os.path.join(fold_dir, "feature_importances.csv")
                        fi_df.to_csv(fi_csv_path, index=False)

                        feature_importance_json = {
                            "names": fi_names,
                            "importances": fi_vals.tolist(),
                            "csv_path": fi_csv_path,
                            "note": "SHAP unavailable -> using feature_importances_"
                        }
                    except Exception:
                        feature_importance_json = None

                # -------------------------------
                # Build per-fold audit + metrics row
                # -------------------------------
                row = {
                    "Model": model_name,
                    "Imbalance": imb_mode,
                    "OuterFoldID": fold_idx,

                    # prevalence info
                    "TrainPrev": prev_tr,
                    "TestPrev": prev_te,

                    # hyperparam search
                    "ParamGrid": json.dumps(grid, default=str),
                    "BestParams": json.dumps(best_params, default=str),
                    "InnerROC_AUC": float(inner_auc),

                    # runtime
                    "Search+RefitTimeSec": elapsed,

                    # default threshold=0.5 metrics
                    "Default_Sensitivity": metrics_05["Sensitivity"],
                    "Default_Specificity": metrics_05["Specificity"],
                    "Default_BAC": metrics_05["BAC"],
                    "Default_ROC_AUC": metrics_05["ROC_AUC"],
                    "Default_PR_AUC": metrics_05["PR_AUC"],
                    "Default_PPV": metrics_05["PPV"],
                    "Default_F1": metrics_05["F1"],
                    "Default_NPV": metrics_05["NPV"],
                    "Default_Brier": metrics_05["Brier"],

                    # tuned (threshold-optimized)
                    "Sensitivity_T": tuned["Sensitivity_T"],
                    "Specificity_T": tuned["Specificity_T"],
                    "BAC_T": tuned["BAC_T"],
                    "PPV_T": tuned["PPV_T"],
                    "NPV_T": tuned["NPV_T"],
                    "Thr": tuned["Thr"],
                    "TP": tuned["TP"],
                    "FP": tuned["FP"],
                    "TN": tuned["TN"],
                    "FN": tuned["FN"],

                    # threshold policy diagnostic (traceability)
                    "ThrMode": CFG.THRESH_PRIMARY,
                    "ThrInfo": json.dumps(thr_info, default=str),

                    # permutation baseline
                    "Obs_ROC_AUC": float(obs_auc) if obs_auc==obs_auc else np.nan,
                    "Perm_AUC_lo": float(auc_lo) if auc_lo==auc_lo else np.nan,
                    "Perm_AUC_hi": float(auc_hi) if auc_hi==auc_hi else np.nan,

                    # interpretability exports
                    "CoefJSON": safe_json_dumps(coef_json),
                    "TopSHAPFeatures": safe_json_dumps(top_shap_features),
                    "SHAP_Export_Success": shap_export_success,
                    "FeatImpJSON": safe_json_dumps(feature_importance_json),
                    "Coef_Export_Success": Coef_Export_Success,

                    # calibration / reliability
                    "CalibrationCurveJSON": json.dumps(cal_curve),
                    "CalBins": CFG.CAL_CURVE_BINS,
                    "Default_Brier_repeat": metrics_05["Brier"],  # explicit again

                    # preprocessing + resampling audit
                    "Preprocessing_Audit": json.dumps({
                        "residualize_covariates": True,
                        "numeric_impute": "median",
                        "numeric_scale": "zscore(StandardScaler)",
                        "categorical_impute": "most_frequent",
                        "categorical_encode": "onehot",
                        "ordinal_encode": True,
                        "covariates_residualized_against": list(CFG.ALWAYS_KEEP),
                        "leakage_guard": "all prep/sampling fit only on training folds"
                    }),
                    "ResamplingConfig": json.dumps({
                        "imbalance_mode": imb_mode,
                        "SafeSampler_min_minority": CFG.MIN_MINORITY_FOR_SYNTH,
                        "SafeSampler_knn_range": [CFG.KNN_MIN, CFG.KNN_MAX],
                        "uses_SMOTE": imb_mode=="smote",
                        "uses_ADASYN": imb_mode=="adasyn",
                        "uses_SMOTEENN": imb_mode=="smoteenn",
                        "uses_Undersample": imb_mode=="undersample",
                        "uses_CostSensitive": imb_mode=="cost_sensitive",
                        "internal_balance_model": (imb_mode=="internal_balance")
                    }),

                    # artifact paths
                     "ModelPicklePath": model_pickle_path,
                     "CoefTablePath": (coef_json["coef_table_path"] if coef_json else None),
                     "SHAPSummaryPath": (shap_top15_json["shap_csv_path"] if shap_top15_json else None),
                }

                # Save checkpoint atomically
                tmp_cp = cp_path + ".tmp"
                with open(tmp_cp, "w") as f:
                    json.dump(
                        {
                            "row": row,
                            "best_params": best_params,
                            "thr_star": float(thr_star),
                            "thr_info": thr_info,
                            "calibration_curve": cal_curve,
                            "artifact_dir": fold_dir
                        },
                        f,
                        indent=2
                    )
                os.replace(tmp_cp, cp_path)

                # Append row so resume is safe
                append_fold_row_to_csv(model_name, imb_mode, row)

                # hold in memory
                model_fold_rows.append(row)
                all_fold_rows.append(row)

            except Exception as e:
                elapsed = round(time.time() - t0, 2)
                print(f"[FOLD {fold_idx}] ERROR: {e}")

                # Error row: fill in NaNs but still record fold as attempted.
                err_row = {
                    "Model": model_name,
                    "Imbalance": imb_mode,
                    "OuterFoldID": fold_idx,

                    "TrainPrev": prev_tr,
                    "TestPrev": prev_te,

                    "ParamGrid": json.dumps(grid, default=str),
                    "BestParams": None,
                    "InnerROC_AUC": np.nan,

                    "Search+RefitTimeSec": elapsed,

                    "Default_Sensitivity": np.nan,
                    "Default_Specificity": np.nan,
                    "Default_BAC": np.nan,
                    "Default_ROC_AUC": np.nan,
                    "Default_PR_AUC": np.nan,
                    "Default_PPV": np.nan,
                    "Default_F1": np.nan,
                    "Default_NPV": np.nan,
                    "Default_Brier": np.nan,

                    "Sensitivity_T": np.nan,
                    "Specificity_T": np.nan,
                    "BAC_T": np.nan,
                    "PPV_T": np.nan,
                    "NPV_T": np.nan,
                    "Thr": np.nan,
                    "TP": 0,
                    "FP": 0,
                    "TN": 0,
                    "FN": 0,

                    "ThrMode": CFG.THRESH_PRIMARY,
                    "ThrInfo": json.dumps({"mode":"error","msg":str(e)}),

                    "Obs_ROC_AUC": np.nan,
                    "Perm_AUC_lo": np.nan,
                    "Perm_AUC_hi": np.nan,

                    "CoefJSON": json.dumps(None),
                    "FeatImpJSON": json.dumps(None),
                    "SHAP_Top15_JSON": json.dumps(None),
                    "SHAP_Export_Success": False,
                    "Coef_Export_Success": False,

                    "CalibrationCurveJSON": json.dumps(None),
                    "CalBins": CFG.CAL_CURVE_BINS,
                    "Default_Brier_repeat": np.nan,

                    "Preprocessing_Audit": json.dumps({
                        "residualize_covariates": True,
                        "numeric_impute": "median",
                        "numeric_scale": "zscore(StandardScaler)",
                        "categorical_impute": "most_frequent",
                        "categorical_encode": "onehot",
                        "ordinal_encode": True,
                        "covariates_residualized_against": list(CFG.ALWAYS_KEEP),
                        "leakage_guard": "all prep/sampling fit only on training folds"
                    }),
                    "ResamplingConfig": json.dumps({
                        "imbalance_mode": imb_mode,
                        "SafeSampler_min_minority": CFG.MIN_MINORITY_FOR_SYNTH,
                        "SafeSampler_knn_range": [CFG.KNN_MIN, CFG.KNN_MAX],
                        "uses_SMOTE": imb_mode=="smote",
                        "uses_ADASYN": imb_mode=="adasyn",
                        "uses_SMOTEENN": imb_mode=="smoteenn",
                        "uses_Undersample": imb_mode=="undersample",
                        "uses_CostSensitive": imb_mode=="cost_sensitive",
                        "internal_balance_model": (imb_mode=="internal_balance")
                    }),

                    "ModelPicklePath": None,
                    "CoefTablePath": None,
                    "SHAPSummaryPath": None,
                }

                with open(cp_path, "w") as f:
                    json.dump({"row": err_row, "error": str(e)}, f, indent=2)
                append_fold_row_to_csv(model_name, imb_mode, err_row)
                model_fold_rows.append(err_row)
                all_fold_rows.append(err_row)

        # ----------------------------------------------------
        # Per-(model,imbalance) summary stats across folds
        # ----------------------------------------------------
        g = pd.DataFrame(model_fold_rows)
        if not g.empty:

            # count threshold policy usage
            def _thr_mode(x):
                try:
                    d = json.loads(x)
                    return d.get("mode","?")
                except Exception:
                    return "?"
            thr_modes = g["ThrInfo"].map(_thr_mode).value_counts().to_dict()

            # metrics arrays for summary
            summ_metrics = {
                "BAC_T": g["BAC_T"],
                "Sensitivity_T": g["Sensitivity_T"],
                "Specificity_T": g["Specificity_T"],
                "PPV_T": g["PPV_T"],
                "NPV_T": g["NPV_T"],
                "Thr": g["Thr"],
                "Default_ROC_AUC": g["Default_ROC_AUC"],
                "Default_PR_AUC": g["Default_PR_AUC"],
                "Default_F1": g["Default_F1"],
                "Default_NPV": g["Default_NPV"],
                "Default_PPV": g["Default_PPV"],
                "Default_Brier": g["Default_Brier"],
                "Obs_ROC_AUC": g["Obs_ROC_AUC"],
            }

            statpack = {}
            for m_name, series in summ_metrics.items():
                ms = _mean_std_ci(series)
                statpack[m_name] = ms

            # robustness
            zero_recall_rate = float((g["Sensitivity_T"]==0).mean())
            bac_spread_std = float(np.nanstd(g["BAC_T"], ddof=1)) if g["BAC_T"].notna().sum()>1 else np.nan
            thr_spread_std = float(np.nanstd(g["Thr"], ddof=1)) if g["Thr"].notna().sum()>1 else np.nan

            model_summary = {
                "Model": model_name,
                "Imbalance": imb_mode,
                "N": g.shape[0],

                # Medians / IQRs
                "BAC_T_median": g["BAC_T"].median(),
                "BAC_T_IQR": (g["BAC_T"].quantile(0.75) - g["BAC_T"].quantile(0.25)),
                "Sens_T_median": g["Sensitivity_T"].median(),
                "Spec_T_median": g["Specificity_T"].median(),
                "PPV_T_median": g["PPV_T"].median(),
                "NPV_T_median": g["NPV_T"].median(),
                "Thr_median": g["Thr"].median(),
                "Thr_IQR": (g["Thr"].quantile(0.75) - g["Thr"].quantile(0.25)),
                "F1_median": g["Default_F1"].median(),
                "ROC_AUC_median": g["Default_ROC_AUC"].median(),
                "PR_AUC_median": g["Default_PR_AUC"].median(),
                "Brier_median": g["Default_Brier"].median(),

                # Means / SD / 95% CI for each metric
                "BAC_T_mean": statpack["BAC_T"]["mean"],
                "BAC_T_std": statpack["BAC_T"]["std"],
                "BAC_T_ci95_lo": statpack["BAC_T"]["ci95_lo"],
                "BAC_T_ci95_hi": statpack["BAC_T"]["ci95_hi"],

                "Sens_T_mean": statpack["Sensitivity_T"]["mean"],
                "Sens_T_std": statpack["Sensitivity_T"]["std"],
                "Sens_T_ci95_lo": statpack["Sensitivity_T"]["ci95_lo"],
                "Sens_T_ci95_hi": statpack["Sensitivity_T"]["ci95_hi"],

                "Spec_T_mean": statpack["Specificity_T"]["mean"],
                "Spec_T_std": statpack["Specificity_T"]["std"],
                "Spec_T_ci95_lo": statpack["Specificity_T"]["ci95_lo"],
                "Spec_T_ci95_hi": statpack["Specificity_T"]["ci95_hi"],

                "PPV_T_mean": statpack["PPV_T"]["mean"],
                "PPV_T_std": statpack["PPV_T"]["std"],
                "PPV_T_ci95_lo": statpack["PPV_T"]["ci95_lo"],
                "PPV_T_ci95_hi": statpack["PPV_T"]["ci95_hi"],

                "NPV_T_mean": statpack["NPV_T"]["mean"],
                "NPV_T_std": statpack["NPV_T"]["std"],
                "NPV_T_ci95_lo": statpack["NPV_T"]["ci95_lo"],
                "NPV_T_ci95_hi": statpack["NPV_T"]["ci95_hi"],

                "Thr_mean": statpack["Thr"]["mean"],
                "Thr_std": statpack["Thr"]["std"],
                "Thr_ci95_lo": statpack["Thr"]["ci95_lo"],
                "Thr_ci95_hi": statpack["Thr"]["ci95_hi"],

                "ROC_AUC_mean": statpack["Default_ROC_AUC"]["mean"],
                "ROC_AUC_std": statpack["Default_ROC_AUC"]["std"],
                "ROC_AUC_ci95_lo": statpack["Default_ROC_AUC"]["ci95_lo"],
                "ROC_AUC_ci95_hi": statpack["Default_ROC_AUC"]["ci95_hi"],

                "PR_AUC_mean": statpack["Default_PR_AUC"]["mean"],
                "PR_AUC_std": statpack["Default_PR_AUC"]["std"],
                "PR_AUC_ci95_lo": statpack["Default_PR_AUC"]["ci95_lo"],
                "PR_AUC_ci95_hi": statpack["Default_PR_AUC"]["ci95_hi"],

                "F1_mean": statpack["Default_F1"]["mean"],
                "F1_std": statpack["Default_F1"]["std"],
                "F1_ci95_lo": statpack["Default_F1"]["ci95_lo"],
                "F1_ci95_hi": statpack["Default_F1"]["ci95_hi"],

                "Brier_mean": statpack["Default_Brier"]["mean"],
                "Brier_std": statpack["Default_Brier"]["std"],
                "Brier_ci95_lo": statpack["Default_Brier"]["ci95_lo"],
                "Brier_ci95_hi": statpack["Default_Brier"]["ci95_hi"],

                # Permutation summary (baseline AUC "null" comparison)
                "Obs_ROC_AUC_mean": g["Obs_ROC_AUC"].mean(),
                "Perm_AUC_lo_mean": g["Perm_AUC_lo"].mean(),
                "Perm_AUC_hi_mean": g["Perm_AUC_hi"].mean(),

                # Robustness / sensitivity
                "ZeroRecallRate_T": zero_recall_rate,
                "BAC_T_std_spread": bac_spread_std,
                "Thr_std_spread": thr_spread_std,

                # Runtime
                "TimeSec_sum": g["Search+RefitTimeSec"].sum(),

                # Threshold mode usage histogram
                **{f"ThrMode_{k}": v for k,v in thr_modes.items()},

                # Interpretability export coverage flags
                "Any_SHAP_Success": bool(g["SHAP_Export_Success"].any()),
                "Any_Coef_Success": bool(g["Coef_Export_Success"].any()),
            }

            summary_rows.append(model_summary)
            pd.DataFrame(summary_rows).to_csv(summary_csv_path(), index=False)

            print(
                f"[SUMMARY] {model_name} ({imb_mode}) | "
                f"median BAC_T={model_summary['BAC_T_median']:.3f} | "
                f"median Sens={model_summary['Sens_T_median']:.3f} | "
                f"median Spec={model_summary['Spec_T_median']:.3f} | "
                f"ZeroRecallRate={model_summary['ZeroRecallRate_T']:.2f}"
            )

    # ----------------------------------------------------
    # Save final aggregate CSVs
    # ----------------------------------------------------
    fold_df = pd.DataFrame(all_fold_rows)

    # final summary sort by BAC_T_median then Sens/Spec median
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["BAC_T_median","Sens_T_median","Spec_T_median"],
        ascending=False
    ).reset_index(drop=True)

    final_folds_path = os.path.join(runs_dir(), "fold_metrics_all.csv")
    fold_df.to_csv(final_folds_path, index=False)
    summary_df.to_csv(summary_csv_path(), index=False)

    print(f"\n[DONE] Saved folds -> {final_folds_path}")
    print(f"[DONE] Saved summary -> {summary_csv_path()}")
    return fold_df, summary_df

# ============================================================
# RUN
# ============================================================

# Script entry point:
# - Logs basic dataset info.
# - Triggers outer_cv over the curated feature set.
if __name__ == "__main__":
    print("[START] Full classifier x imbalance sweep with nested CV, "
          "guarded thresholding, calibration, resume-safety, "
          "and interpretability logging (incl. SHAP).")
    print(f"[DATA] X={X_full.shape}, positives={int((y_full==1).sum())}, "
          f"negatives={int((y_full==0).sum())}")
    folds, summary = outer_cv(X_full, y_full)
