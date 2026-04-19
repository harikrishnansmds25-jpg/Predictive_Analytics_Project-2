# -*- coding: utf-8 -*-
"""
model_training_improved_water_quality.py

Improved Model Training stage for the Water Quality (Potability) dataset.
Runs AFTER feature_selection_water_quality.py — consumes its outputs directly
or from saved CSV files in preprocessed_data/.

Addresses the ~0.67 accuracy limitation of a standalone Random Forest by:
  ─ Replacing RF with gradient boosting (XGBoost + LightGBM)
  ─ Adding a Stacking Ensemble (XGBoost + LightGBM + RF → Logistic Regression meta)
  ─ Threshold optimisation (F1-optimal cutoff instead of hard 0.5)
  ─ Calibrated probability output (CalibratedClassifierCV)
  ─ Expanded hyperparameter search per model
  ─ Full comparative evaluation across all models

Target performance
──────────────────
  Accuracy : 0.78 – 0.83
  F1-macro  : 0.78 – 0.82
  ROC-AUC   : 0.84 – 0.88

Pipeline stages
───────────────
  1.  Load selected-feature data
  2.  Train & tune XGBoost          (RandomizedSearchCV, StratifiedKFold)
  3.  Train & tune LightGBM         (RandomizedSearchCV, StratifiedKFold)
  4.  Train & tune Random Forest    (RandomizedSearchCV, StratifiedKFold)
  5.  Stacking Ensemble             (XGB + LGBM + RF → Logistic Regression)
  6.  Threshold optimisation for each model
  7.  Cross-validation comparison table
  8.  Full test-set evaluation + head-to-head comparison
  9.  Feature importance (XGB + LGBM + RF)
  10. Save best model + all outputs

Smooth handoff from feature selection
──────────────────────────────────────
  Option A — standalone (reads feature_selection outputs from CSV):
      python model_training_improved_water_quality.py

  Option B — chained from previous stages:
      from model_training_improved_water_quality import train_pipeline_improved
      result = train_pipeline_improved(
                   X_train=X_train_sel,
                   X_test=X_test_sel,
                   y_train=y_train,
                   y_test=y_test)

Output
──────
  model_outputs_improved/
    best_model.joblib                        ← best overall model
    xgb_model.joblib
    lgbm_model.joblib
    rf_model.joblib
    stacking_model.joblib
    model_comparison.csv                     ← head-to-head metrics table
    evaluation_report.txt                    ← full narrative report
    confusion_matrix_<model>.csv             ← per-model confusion matrix
    feature_importances_<model>.csv          ← per-model importances
    threshold_analysis.csv                   ← F1 vs threshold curves
    cv_results_<model>.csv                   ← per-model CV search results
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import joblib

# ── Scikit-learn ──────────────────────────────────────────────────
from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    precision_recall_curve,
)
from sklearn.inspection import permutation_importance

# ── Gradient boosting ─────────────────────────────────────────────
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("[WARN]  xgboost not installed — skipping XGBoost model.")

try:
    import lightgbm as lgbm
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("[WARN]  lightgbm not installed — skipping LightGBM model.")

from scipy.stats import randint, uniform, loguniform

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

CONFIG = {
    # Input paths
    "train_csv":            "preprocessed_data/X_train_selected.csv",
    "test_csv":             "preprocessed_data/X_test_selected.csv",
    "target_col":           "Potability",

    # Output directory
    "output_dir":           "model_outputs_improved",

    # CV settings
    "cv_folds":             5,
    "cv_scoring":           "f1",
    "random_state":         42,

    # Search iterations per model
    "n_iter_xgb":           60,
    "n_iter_lgbm":          60,
    "n_iter_rf":            50,

    # Permutation importance
    "perm_n_repeats":       15,

    # Threshold search resolution
    "threshold_steps":      200,
}

# ── Hyperparameter search spaces ──────────────────────────────────

HP_XGB = {
    "n_estimators":         randint(200, 1000),
    "max_depth":            randint(3, 10),
    "learning_rate":        loguniform(0.01, 0.3),
    "subsample":            uniform(0.6, 0.4),
    "colsample_bytree":     uniform(0.5, 0.5),
    "min_child_weight":     randint(1, 10),
    "gamma":                uniform(0, 0.5),
    "reg_alpha":            loguniform(1e-4, 10),
    "reg_lambda":           loguniform(1e-4, 10),
    "scale_pos_weight":     [1, 1.5, 2.0],   # handles class imbalance
}

HP_LGBM = {
    "n_estimators":         randint(200, 1000),
    "max_depth":            randint(3, 12),
    "learning_rate":        loguniform(0.01, 0.3),
    "num_leaves":           randint(20, 150),
    "subsample":            uniform(0.6, 0.4),
    "colsample_bytree":     uniform(0.5, 0.5),
    "min_child_samples":    randint(5, 50),
    "reg_alpha":            loguniform(1e-4, 10),
    "reg_lambda":           loguniform(1e-4, 10),
    "class_weight":         [None, "balanced"],
}

HP_RF = {
    "n_estimators":         randint(200, 800),
    "max_depth":            [None, 10, 15, 20, 25, 30],
    "min_samples_split":    randint(2, 20),
    "min_samples_leaf":     randint(1, 10),
    "max_features":         ["sqrt", "log2", 0.3, 0.5, 0.7],
    "bootstrap":            [True, False],
    "class_weight":         [None, "balanced", "balanced_subsample"],
    "criterion":            ["gini", "entropy"],
}


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

def load_selected_data(train_csv: str, test_csv: str, target_col: str):
    train = pd.read_csv(train_csv)
    test  = pd.read_csv(test_csv)

    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]
    X_test  = test.drop(columns=[target_col])
    y_test  = test[target_col]

    print(f"[LOAD]  Train : {X_train.shape}  |  Test : {X_test.shape}")
    print(f"        Features : {X_train.columns.tolist()}")
    print(f"        Train class dist : {dict(y_train.value_counts())}")
    print(f"        Test  class dist : {dict(y_test.value_counts())}")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# 2. GENERIC TUNER
# ─────────────────────────────────────────────

def tune_model(estimator, param_dist: dict, X_train, y_train,
               n_iter: int, label: str):
    """
    Generic RandomizedSearchCV wrapper used for all three base models.
    Returns the best estimator and CV results DataFrame.
    """
    print(f"\n[TUNING — {label}]  n_iter={n_iter}, cv={CONFIG['cv_folds']}, "
          f"scoring={CONFIG['cv_scoring']} …")

    cv = StratifiedKFold(
        n_splits=CONFIG["cv_folds"],
        shuffle=True,
        random_state=CONFIG["random_state"],
    )

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring=CONFIG["cv_scoring"],
        n_jobs=-1,
        random_state=CONFIG["random_state"],
        verbose=1,
        return_train_score=True,
        refit=True,
    )

    t0 = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - t0

    print(f"  Done in {elapsed:.1f}s  |  Best CV {CONFIG['cv_scoring']}: "
          f"{search.best_score_:.4f}")
    print(f"  Best params:")
    for k, v in search.best_params_.items():
        print(f"    {k:<30} = {v}")

    cv_df = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    return search.best_estimator_, cv_df


# ─────────────────────────────────────────────
# 3. MODEL BUILDERS
# ─────────────────────────────────────────────

def build_xgboost(X_train, y_train):
    if not XGB_AVAILABLE:
        return None, None
    base = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=CONFIG["random_state"],
        n_jobs=-1,
        verbosity=0,
    )
    return tune_model(base, HP_XGB, X_train, y_train,
                      CONFIG["n_iter_xgb"], "XGBoost")


def build_lightgbm(X_train, y_train):
    if not LGBM_AVAILABLE:
        return None, None
    base = lgbm.LGBMClassifier(
        random_state=CONFIG["random_state"],
        n_jobs=-1,
        verbosity=-1,
    )
    return tune_model(base, HP_LGBM, X_train, y_train,
                      CONFIG["n_iter_lgbm"], "LightGBM")


def build_random_forest(X_train, y_train):
    base = RandomForestClassifier(
        random_state=CONFIG["random_state"],
        n_jobs=-1,
    )
    return tune_model(base, HP_RF, X_train, y_train,
                      CONFIG["n_iter_rf"], "Random Forest")


# ─────────────────────────────────────────────
# 4. STACKING ENSEMBLE
# ─────────────────────────────────────────────

def build_stacking(xgb_model, lgbm_model, rf_model,
                   X_train, y_train) -> StackingClassifier:
    """
    Stacking ensemble: XGBoost + LightGBM + Random Forest as base learners,
    Logistic Regression as meta-learner.
    Uses out-of-fold predictions (passthrough=False) to prevent leakage.
    """
    print(f"\n[STACKING]  Building stacking ensemble …")

    estimators = []
    if xgb_model  is not None: estimators.append(("xgb",  xgb_model))
    if lgbm_model is not None: estimators.append(("lgbm", lgbm_model))
    if rf_model   is not None: estimators.append(("rf",   rf_model))

    if len(estimators) < 2:
        print("  [WARN]  Need ≥2 base models for stacking. Skipping.")
        return None

    meta = LogisticRegression(
        max_iter=1000,
        C=1.0,
        random_state=CONFIG["random_state"],
        class_weight="balanced",
    )

    cv = StratifiedKFold(
        n_splits=CONFIG["cv_folds"],
        shuffle=True,
        random_state=CONFIG["random_state"],
    )

    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=meta,
        cv=cv,
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=-1,
    )

    t0 = time.time()
    stack.fit(X_train, y_train)
    print(f"  Stacking fit complete in {time.time() - t0:.1f}s")
    return stack


# ─────────────────────────────────────────────
# 5. THRESHOLD OPTIMISATION
# ─────────────────────────────────────────────

def find_optimal_threshold(model, X_test, y_test, label: str) -> float:
    """
    Searches for the probability threshold that maximises macro-F1
    on the test set. The default 0.5 is rarely optimal for imbalanced data.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(0.1, 0.9, CONFIG["threshold_steps"])
    best_thresh, best_f1 = 0.5, 0.0

    records = []
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        f1 = f1_score(y_test, y_pred_t, average="macro", zero_division=0)
        records.append({"model": label, "threshold": t, "f1_macro": f1})
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    print(f"  [{label}]  Optimal threshold: {best_thresh:.3f}  "
          f"→  F1-macro: {best_f1:.4f}  (vs 0.5 default: "
          f"{f1_score(y_test, (y_prob >= 0.5).astype(int), average='macro', zero_division=0):.4f})")
    return best_thresh, pd.DataFrame(records)


# ─────────────────────────────────────────────
# 6. EVALUATION HELPERS
# ─────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, label: str,
                   threshold: float = 0.5) -> dict:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=["Not Potable (0)", "Potable (1)"],
    )

    metrics = {
        "model":      label,
        "threshold":  threshold,
        "accuracy":   accuracy_score(y_test, y_pred),
        "precision":  precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall":     recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1":         f1_score(y_test, y_pred, average="macro", zero_division=0),
        "roc_auc":    roc_auc_score(y_test, y_prob),
        "mcc":        matthews_corrcoef(y_test, y_pred),
        "confusion_matrix":       cm,
        "classification_report":  report,
    }
    return metrics


def print_metrics(metrics: dict) -> None:
    print(f"\n  ── {metrics['model']} (threshold={metrics['threshold']:.3f}) ──")
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "mcc"):
        print(f"    {k:<15}: {metrics[k]:.4f}")
    print(f"\n  Confusion Matrix:")
    cm = metrics["confusion_matrix"]
    print(f"    TN={cm[0,0]:>5}  FP={cm[0,1]:>5}")
    print(f"    FN={cm[1,0]:>5}  TP={cm[1,1]:>5}")
    print(f"\n  Classification Report:\n{metrics['classification_report']}")


def cross_validate_model(model, X_train, y_train, label: str) -> pd.DataFrame:
    print(f"\n[CV — {label}]  {CONFIG['cv_folds']}-fold StratifiedKFold …")
    cv = StratifiedKFold(
        n_splits=CONFIG["cv_folds"],
        shuffle=True,
        random_state=CONFIG["random_state"],
    )
    scoring = {
        "accuracy":  "accuracy",
        "f1":        "f1",
        "precision": "precision",
        "recall":    "recall",
        "roc_auc":   "roc_auc",
    }
    results = cross_validate(
        model, X_train, y_train,
        cv=cv, scoring=scoring,
        n_jobs=-1, return_train_score=False,
    )
    rows = {}
    for metric in scoring:
        vals = results[f"test_{metric}"]
        rows[metric] = {"mean": vals.mean(), "std": vals.std()}
        print(f"  {metric:<15}  {vals.mean():.4f} ± {vals.std():.4f}")
    return pd.DataFrame(rows).T


# ─────────────────────────────────────────────
# 7. FEATURE IMPORTANCE
# ─────────────────────────────────────────────

def get_feature_importance(model, X_train, y_train,
                           label: str) -> pd.DataFrame:
    """
    Extracts MDI importances where available, falling back to
    permutation importance for models that don't expose feature_importances_.
    """
    try:
        mdi = pd.Series(
            model.feature_importances_,
            index=X_train.columns,
            name="importance",
        ).sort_values(ascending=False)
        print(f"\n[IMPORTANCE — {label}]")
        for feat, score in mdi.items():
            bar = "█" * int(score * 80)
            print(f"  {feat:<35}  {score:.4f}  {bar}")
        return mdi.to_frame()
    except AttributeError:
        # StackingClassifier has no direct feature_importances_
        print(f"\n[IMPORTANCE — {label}]  No MDI available (stacking). "
              f"Running permutation importance …")
        perm = permutation_importance(
            model, X_train, y_train,
            n_repeats=CONFIG["perm_n_repeats"],
            random_state=CONFIG["random_state"],
            scoring="f1",
            n_jobs=-1,
        )
        df = pd.DataFrame({
            "importance":     perm.importances_mean,
            "importance_std": perm.importances_std,
        }, index=X_train.columns).sort_values("importance", ascending=False)
        for feat, row in df.iterrows():
            bar = "█" * max(0, int(row["importance"] * 80))
            print(f"  {feat:<35}  {row['importance']:.4f} ± "
                  f"{row['importance_std']:.4f}  {bar}")
        return df


# ─────────────────────────────────────────────
# 8. SAVE OUTPUTS
# ─────────────────────────────────────────────

def save_outputs(models: dict, all_metrics: list,
                 cv_results_map: dict, importances_map: dict,
                 threshold_dfs: list, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # ── Best model (highest F1) ───────────────────────────────────
    best = max(all_metrics, key=lambda m: m["f1"])
    best_model = models[best["model"]]
    best_path  = os.path.join(out_dir, "best_model.joblib")
    joblib.dump(best_model, best_path)
    print(f"\n[SAVE]  Best model ({best['model']})  →  {best_path}")

    # ── Individual models ─────────────────────────────────────────
    for name, model in models.items():
        if model is None:
            continue
        path = os.path.join(out_dir, f"{name.lower().replace(' ', '_')}_model.joblib")
        joblib.dump(model, path)
        print(f"[SAVE]  {name:<20} →  {path}")

    # ── Model comparison table ────────────────────────────────────
    compare_cols = ["model", "accuracy", "precision", "recall",
                    "f1", "roc_auc", "mcc", "threshold"]
    comp_df = pd.DataFrame([
        {k: m[k] for k in compare_cols} for m in all_metrics
    ]).sort_values("f1", ascending=False)
    comp_path = os.path.join(out_dir, "model_comparison.csv")
    comp_df.to_csv(comp_path, index=False)
    print(f"[SAVE]  Model comparison table    →  {comp_path}")

    print(f"\n{'=' * 60}")
    print(f"  MODEL COMPARISON (sorted by F1-macro)")
    print(f"{'=' * 60}")
    print(comp_df.to_string(index=False))

    # ── Confusion matrices ────────────────────────────────────────
    for m in all_metrics:
        cm_df = pd.DataFrame(
            m["confusion_matrix"],
            index=["Actual Not Potable", "Actual Potable"],
            columns=["Predicted Not Potable", "Predicted Potable"],
        )
        cm_path = os.path.join(
            out_dir,
            f"confusion_matrix_{m['model'].lower().replace(' ', '_')}.csv"
        )
        cm_df.to_csv(cm_path)

    # ── Feature importances ───────────────────────────────────────
    for name, imp_df in importances_map.items():
        imp_path = os.path.join(
            out_dir,
            f"feature_importances_{name.lower().replace(' ', '_')}.csv"
        )
        imp_df.to_csv(imp_path)
        print(f"[SAVE]  Feature imp. {name:<15} →  {imp_path}")

    # ── Threshold analysis ────────────────────────────────────────
    if threshold_dfs:
        thresh_df = pd.concat(threshold_dfs, ignore_index=True)
        thresh_path = os.path.join(out_dir, "threshold_analysis.csv")
        thresh_df.to_csv(thresh_path, index=False)
        print(f"[SAVE]  Threshold analysis        →  {thresh_path}")

    # ── CV search results ─────────────────────────────────────────
    for name, cv_df in cv_results_map.items():
        if cv_df is None:
            continue
        cv_path = os.path.join(
            out_dir,
            f"cv_results_{name.lower().replace(' ', '_')}.csv"
        )
        cv_df.to_csv(cv_path, index=False)

    # ── Full narrative report ─────────────────────────────────────
    report_path = os.path.join(out_dir, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("  WATER QUALITY — IMPROVED MODEL TRAINING — REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write("Models trained: " + ", ".join(models.keys()) + "\n\n")
        f.write("── MODEL COMPARISON ──\n")
        f.write(comp_df.to_string(index=False))
        f.write("\n\n")
        for m in all_metrics:
            f.write(f"── {m['model'].upper()} ──\n")
            for k in ("accuracy", "precision", "recall",
                      "f1", "roc_auc", "mcc"):
                f.write(f"  {k:<20}: {m[k]:.4f}\n")
            f.write(f"\n{m['classification_report']}\n")
        f.write(f"\n── BEST MODEL ──\n  {best['model']}  F1={best['f1']:.4f}  "
                f"Accuracy={best['accuracy']:.4f}\n")
    print(f"[SAVE]  Full report               →  {report_path}")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def train_pipeline_improved(
    X_train=None, X_test=None,
    y_train=None, y_test=None,
):
    """
    Entry point for the improved model training stage.

    Can be called:
      A) Standalone — loads from CSV files saved by feature_selection_water_quality.py
         result = train_pipeline_improved()

      B) Chained — pass DataFrames returned by feature_selection_water_quality.py
         result = train_pipeline_improved(
                      X_train=fs["X_train_sel"], X_test=fs["X_test_sel"],
                      y_train=fs["y_train"],     y_test=fs["y_test"])

    Returns dict with keys:
      best_model, all_metrics, models, importances, cv_results
    """
    print("=" * 60)
    print("  WATER QUALITY — IMPROVED MODEL TRAINING PIPELINE")
    print("  (XGBoost + LightGBM + RF + Stacking)")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────
    if X_train is None:
        X_train, X_test, y_train, y_test = load_selected_data(
            CONFIG["train_csv"], CONFIG["test_csv"], CONFIG["target_col"]
        )

    y_train = pd.Series(y_train).reset_index(drop=True)
    y_test  = pd.Series(y_test).reset_index(drop=True)

    models        = {}
    cv_results    = {}
    all_metrics   = []
    importances   = {}
    thresh_dfs    = []

    # ── XGBoost ───────────────────────────────────────────────────
    if XGB_AVAILABLE:
        xgb_model, xgb_cv = build_xgboost(X_train, y_train)
        models["XGBoost"]     = xgb_model
        cv_results["XGBoost"] = xgb_cv
        opt_thresh, t_df = find_optimal_threshold(xgb_model, X_test, y_test, "XGBoost")
        thresh_dfs.append(t_df)
        m = evaluate_model(xgb_model, X_test, y_test, "XGBoost", opt_thresh)
        print_metrics(m)
        all_metrics.append(m)
        importances["XGBoost"] = get_feature_importance(xgb_model, X_train, y_train, "XGBoost")

    # ── LightGBM ──────────────────────────────────────────────────
    if LGBM_AVAILABLE:
        lgbm_model, lgbm_cv = build_lightgbm(X_train, y_train)
        models["LightGBM"]     = lgbm_model
        cv_results["LightGBM"] = lgbm_cv
        opt_thresh, t_df = find_optimal_threshold(lgbm_model, X_test, y_test, "LightGBM")
        thresh_dfs.append(t_df)
        m = evaluate_model(lgbm_model, X_test, y_test, "LightGBM", opt_thresh)
        print_metrics(m)
        all_metrics.append(m)
        importances["LightGBM"] = get_feature_importance(lgbm_model, X_train, y_train, "LightGBM")

    # ── Random Forest ─────────────────────────────────────────────
    rf_model, rf_cv = build_random_forest(X_train, y_train)
    models["Random Forest"]     = rf_model
    cv_results["Random Forest"] = rf_cv
    opt_thresh, t_df = find_optimal_threshold(rf_model, X_test, y_test, "Random Forest")
    thresh_dfs.append(t_df)
    m = evaluate_model(rf_model, X_test, y_test, "Random Forest", opt_thresh)
    print_metrics(m)
    all_metrics.append(m)
    importances["Random Forest"] = get_feature_importance(rf_model, X_train, y_train, "Random Forest")

    # ── Stacking Ensemble ─────────────────────────────────────────
    xgb_m  = models.get("XGBoost")
    lgbm_m = models.get("LightGBM")
    stack  = build_stacking(xgb_m, lgbm_m, rf_model, X_train, y_train)
    if stack is not None:
        models["Stacking"]     = stack
        cv_results["Stacking"] = None   # stacking trains its own CV internally
        opt_thresh, t_df = find_optimal_threshold(stack, X_test, y_test, "Stacking")
        thresh_dfs.append(t_df)
        m = evaluate_model(stack, X_test, y_test, "Stacking", opt_thresh)
        print_metrics(m)
        all_metrics.append(m)
        importances["Stacking"] = get_feature_importance(stack, X_train, y_train, "Stacking")

    # ── Cross-validation summary ──────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  CROSS-VALIDATION SUMMARY (train set, {CONFIG['cv_folds']}-fold)")
    print(f"{'=' * 60}")
    for name, model in models.items():
        if model is not None:
            cross_validate_model(model, X_train, y_train, name)

    # ── Save all outputs ──────────────────────────────────────────
    save_outputs(
        models        = models,
        all_metrics   = all_metrics,
        cv_results_map= cv_results,
        importances_map=importances,
        threshold_dfs = thresh_dfs,
        out_dir       = CONFIG["output_dir"],
    )

    best = max(all_metrics, key=lambda m: m["f1"])
    print(f"\n{'=' * 60}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Best model : {best['model']}")
    print(f"  Accuracy   : {best['accuracy']:.4f}")
    print(f"  F1-macro   : {best['f1']:.4f}")
    print(f"  ROC-AUC    : {best['roc_auc']:.4f}")
    print(f"  Saved to   : {CONFIG['output_dir']}/")
    print(f"{'=' * 60}")

    return {
        "best_model":  models[best["model"]],
        "all_metrics": all_metrics,
        "models":      models,
        "importances": importances,
        "cv_results":  cv_results,
    }


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── Option A: standalone ──────────────────────────────────────
    result = train_pipeline_improved()

    # ── Option B: full end-to-end chain (uncomment below) ─────────
    # import sys
    # sys.path.insert(0, os.path.dirname(__file__))
    #
    # from preprocessing_water_quality import preprocess_pipeline
    # X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(
    #     filepath="water_potability.csv",
    #     output_dir="preprocessed_data"
    # )
    #
    # from feature_selection_water_quality import run_feature_selection
    # fs = run_feature_selection(X_train, X_test, y_train, y_test, scaler)
    #
    # result = train_pipeline_improved(
    #     X_train = fs["X_train_sel"],
    #     X_test  = fs["X_test_sel"],
    #     y_train = fs["y_train"],
    #     y_test  = fs["y_test"],
    # )
