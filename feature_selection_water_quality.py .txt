# -*- coding: utf-8 -*-
"""
feature_selection_water_quality.py

Feature Selection stage for the Water Quality (Potability) dataset.
Runs AFTER preprocessing_water_quality.py — consumes its outputs directly
or from saved CSV files.

Methods applied (ensemble / consensus approach):
  1. Variance Threshold          — drop near-zero-variance features
  2. Pearson Correlation Filter  — detect inter-feature multicollinearity
  3. ANOVA F-test (SelectKBest)  — univariate statistical relevance
  4. Mutual Information          — non-linear statistical relevance
  5. Random Forest Importance    — model-based (MDI) importance
  6. Permutation Importance      — model-based (unbiased) importance
  7. RFE with Random Forest      — recursive feature elimination
  8. Consensus Ranking           — aggregates all methods; selects final set

Output
──────
  preprocessed_data/
    feature_scores_all_methods.csv   ← per-feature scores for every method
    feature_rankings_consensus.csv   ← final aggregated rank + selected flag
    selected_features.txt            ← plain list of selected feature names
    X_train_selected.csv             ← training set with selected features only
    X_test_selected.csv              ← test set  with selected features only

Smooth handoff to modelling
────────────────────────────
  from feature_selection_water_quality import run_feature_selection
  result = run_feature_selection()          # runs full pipeline internally
  # -- OR, if preprocessing already ran --
  result = run_feature_selection(
                X_train=X_train, X_test=X_test,
                y_train=y_train, y_test=y_test)

  result keys:
    X_train_sel, X_test_sel    → DataFrames ready for model.fit / model.predict
    selected_features          → list[str]
    feature_rankings           → pd.DataFrame (full audit trail)
    scaler                     → RobustScaler fitted in preprocessing
"""

import os
import warnings
import numpy as np
import pandas as pd

from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest, f_classif,
    mutual_info_classif,
    RFE,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

CONFIG = {
    # Paths — align with preprocessing_water_quality.py defaults
    "preprocessed_dir":     "preprocessed_data",
    "train_csv":            "preprocessed_data/train_preprocessed.csv",
    "test_csv":             "preprocessed_data/test_preprocessed.csv",
    "target_col":           "Potability",
    "output_dir":           "preprocessed_data",

    # Variance threshold (fraction of max variance)
    "variance_threshold":   0.01,

    # Correlation — drop one of a pair if |r| > this value
    "corr_threshold":       0.90,

    # How many top features to keep from univariate methods
    "k_best":               10,

    # Random Forest params (used for importance + RFE)
    "rf_n_estimators":      200,
    "rf_random_state":      42,
    "rf_n_jobs":            -1,

    # RFE: number of features to select
    "rfe_n_features":       8,

    # Final consensus: minimum number of methods that must rank
    # a feature in their top-k to be included in the selected set
    "consensus_top_k":      8,   # "top-k" threshold per method
    "consensus_min_votes":  3,   # must appear in top-k of at least N methods
}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _rank_series(series: pd.Series, ascending: bool = False) -> pd.Series:
    """Rank a score Series (higher rank = more important feature)."""
    return series.rank(ascending=ascending, method='min')


def _load_from_csv(train_csv: str, test_csv: str, target_col: str):
    train = pd.read_csv(train_csv)
    test  = pd.read_csv(test_csv)
    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]
    X_test  = test.drop(columns=[target_col])
    y_test  = test[target_col]
    print(f"[LOAD] Train: {X_train.shape}  |  Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# METHOD 1 — VARIANCE THRESHOLD
# ─────────────────────────────────────────────

def variance_filter(X_train: pd.DataFrame, threshold: float):
    """
    Removes features whose variance is below `threshold`.
    Near-zero-variance features carry almost no discriminative signal.
    """
    vt = VarianceThreshold(threshold=threshold)
    vt.fit(X_train)
    variances      = pd.Series(vt.variances_, index=X_train.columns, name="variance")
    kept_features  = X_train.columns[vt.get_support()].tolist()
    dropped        = [c for c in X_train.columns if c not in kept_features]

    print(f"\n[VARIANCE THRESHOLD]  threshold={threshold}")
    print(f"  Features dropped ({len(dropped)}): {dropped}")
    print(f"  Features kept   ({len(kept_features)}): {kept_features}")
    return variances, kept_features


# ─────────────────────────────────────────────
# METHOD 2 — CORRELATION FILTER
# ─────────────────────────────────────────────

def correlation_filter(X_train: pd.DataFrame, threshold: float):
    """
    Computes the Pearson correlation matrix.
    When two features are correlated above the threshold, the one with
    the lower mean absolute correlation (less globally correlated) is kept.
    Returns the correlation matrix and the list of features to drop.
    """
    corr_matrix = X_train.corr().abs()
    upper       = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = []
    for col in upper.columns:
        if any(upper[col] > threshold):
            to_drop.append(col)

    kept = [c for c in X_train.columns if c not in to_drop]

    print(f"\n[CORRELATION FILTER]  threshold={threshold}")
    print(f"  Highly correlated features to drop ({len(to_drop)}): {to_drop}")
    print(f"  Features retained ({len(kept)}): {kept}")
    return corr_matrix, to_drop


# ─────────────────────────────────────────────
# METHOD 3 — ANOVA F-TEST (SelectKBest)
# ─────────────────────────────────────────────

def anova_f_test(X_train: pd.DataFrame, y_train: pd.Series, k: int):
    """
    Univariate ANOVA F-statistic between each feature and the target.
    Measures linear dependency; complements mutual information.
    """
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X_train, y_train)

    scores = pd.Series(selector.scores_,   index=X_train.columns, name="anova_f_score")
    pvals  = pd.Series(selector.pvalues_,  index=X_train.columns, name="anova_p_value")
    top_k  = X_train.columns[selector.get_support()].tolist()

    print(f"\n[ANOVA F-TEST]  Top-{k} features:")
    for feat in top_k:
        print(f"  {feat:35s}  F={scores[feat]:.3f}  p={pvals[feat]:.4f}")

    return scores, pvals, top_k


# ─────────────────────────────────────────────
# METHOD 4 — MUTUAL INFORMATION
# ─────────────────────────────────────────────

def mutual_information(X_train: pd.DataFrame, y_train: pd.Series, k: int):
    """
    Non-parametric measure of dependency — captures non-linear relationships
    that ANOVA F-test misses (e.g., interaction effects in engineered features).
    """
    mi_scores = mutual_info_classif(
        X_train, y_train,
        random_state=CONFIG['rf_random_state']
    )
    scores = pd.Series(mi_scores, index=X_train.columns, name="mutual_info_score")
    top_k  = scores.nlargest(k).index.tolist()

    print(f"\n[MUTUAL INFORMATION]  Top-{k} features:")
    for feat in top_k:
        print(f"  {feat:35s}  MI={scores[feat]:.4f}")

    return scores, top_k


# ─────────────────────────────────────────────
# METHOD 5 — RANDOM FOREST MDI IMPORTANCE
# ─────────────────────────────────────────────

def rf_importance(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Mean Decrease in Impurity (MDI) — fast, model-based importance.
    Can overestimate high-cardinality / correlated features, so we pair
    it with permutation importance (Method 6) to cross-validate.
    """
    rf = RandomForestClassifier(
        n_estimators=CONFIG['rf_n_estimators'],
        random_state=CONFIG['rf_random_state'],
        n_jobs=CONFIG['rf_n_jobs'],
    )
    rf.fit(X_train, y_train)

    scores = pd.Series(rf.feature_importances_, index=X_train.columns,
                       name="rf_mdi_importance")
    top_k  = scores.nlargest(CONFIG['k_best']).index.tolist()

    print(f"\n[RF MDI IMPORTANCE]  Top-{CONFIG['k_best']} features:")
    for feat in top_k:
        print(f"  {feat:35s}  MDI={scores[feat]:.4f}")

    return scores, rf   # return fitted rf for permutation step


# ─────────────────────────────────────────────
# METHOD 6 — PERMUTATION IMPORTANCE
# ─────────────────────────────────────────────

def permutation_imp(X_train: pd.DataFrame, y_train: pd.Series,
                    rf_model: RandomForestClassifier):
    """
    Permutation importance randomly shuffles each feature column and
    measures the drop in model accuracy — unbiased to cardinality/correlation.
    Uses the same RF fitted in Method 5 (no extra training cost).
    """
    perm = permutation_importance(
        rf_model, X_train, y_train,
        n_repeats=10,
        random_state=CONFIG['rf_random_state'],
        n_jobs=CONFIG['rf_n_jobs'],
    )
    scores = pd.Series(perm.importances_mean, index=X_train.columns,
                       name="permutation_importance")
    top_k  = scores.nlargest(CONFIG['k_best']).index.tolist()

    print(f"\n[PERMUTATION IMPORTANCE]  Top-{CONFIG['k_best']} features:")
    for feat in top_k:
        print(f"  {feat:35s}  PI={scores[feat]:.4f}")

    return scores, top_k


# ─────────────────────────────────────────────
# METHOD 7 — RECURSIVE FEATURE ELIMINATION (RFE)
# ─────────────────────────────────────────────

def rfe_selection(X_train: pd.DataFrame, y_train: pd.Series, n_features: int):
    """
    RFE iteratively removes the least important features using a Random Forest
    estimator until `n_features` remain. Captures feature interactions that
    univariate methods miss.
    """
    rf_rfe = RandomForestClassifier(
        n_estimators=100,        # lighter RF — speed vs. accuracy trade-off
        random_state=CONFIG['rf_random_state'],
        n_jobs=CONFIG['rf_n_jobs'],
    )
    rfe = RFE(estimator=rf_rfe, n_features_to_select=n_features, step=1)
    rfe.fit(X_train, y_train)

    ranking  = pd.Series(rfe.ranking_,   index=X_train.columns, name="rfe_ranking")
    selected = X_train.columns[rfe.get_support()].tolist()

    print(f"\n[RFE]  Selected {n_features} features:")
    for feat in selected:
        print(f"  {feat}")

    return ranking, selected


# ─────────────────────────────────────────────
# METHOD 8 — CONSENSUS RANKING
# ─────────────────────────────────────────────

def consensus_ranking(
    all_features:       list,
    variance_scores:    pd.Series,
    anova_scores:       pd.Series,
    mi_scores:          pd.Series,
    rf_mdi_scores:      pd.Series,
    perm_scores:        pd.Series,
    rfe_ranking:        pd.Series,
    top_k:              int,
    min_votes:          int,
):
    """
    Aggregates all scoring methods into a single consensus table.

    Voting rule: a feature gets 1 vote per method in which it appears
    in the top-`top_k`. Features with >= `min_votes` are selected.

    RFE uses a rank-based vote: RFE ranking == 1 means selected by RFE.
    """
    scores_df = pd.DataFrame(index=all_features)
    scores_df["variance"]               = variance_scores
    scores_df["anova_f_score"]          = anova_scores
    scores_df["mutual_info_score"]      = mi_scores
    scores_df["rf_mdi_importance"]      = rf_mdi_scores
    scores_df["permutation_importance"] = perm_scores
    scores_df["rfe_ranking"]            = rfe_ranking   # lower = better for RFE

    # Normalised rank (1 = best) for each scoring method
    scores_df["rank_variance"]    = _rank_series(scores_df["variance"])
    scores_df["rank_anova"]       = _rank_series(scores_df["anova_f_score"])
    scores_df["rank_mi"]          = _rank_series(scores_df["mutual_info_score"])
    scores_df["rank_rf_mdi"]      = _rank_series(scores_df["rf_mdi_importance"])
    scores_df["rank_perm"]        = _rank_series(scores_df["permutation_importance"])
    scores_df["rank_rfe"]         = scores_df["rfe_ranking"]   # already a rank

    # Vote: 1 if feature is in top-k for that method, 0 otherwise
    scores_df["vote_variance"] = (scores_df["rank_variance"] <= top_k).astype(int)
    scores_df["vote_anova"]    = (scores_df["rank_anova"]    <= top_k).astype(int)
    scores_df["vote_mi"]       = (scores_df["rank_mi"]       <= top_k).astype(int)
    scores_df["vote_rf_mdi"]   = (scores_df["rank_rf_mdi"]   <= top_k).astype(int)
    scores_df["vote_perm"]     = (scores_df["rank_perm"]     <= top_k).astype(int)
    scores_df["vote_rfe"]      = (scores_df["rank_rfe"]      == 1).astype(int)

    vote_cols = [c for c in scores_df.columns if c.startswith("vote_")]
    scores_df["total_votes"] = scores_df[vote_cols].sum(axis=1)

    # Mean normalised rank across all methods (lower = better)
    rank_cols = [c for c in scores_df.columns if c.startswith("rank_")]
    scores_df["mean_rank"] = scores_df[rank_cols].mean(axis=1)

    scores_df["selected"] = (scores_df["total_votes"] >= min_votes)
    scores_df = scores_df.sort_values("mean_rank")

    selected_features = scores_df[scores_df["selected"]].index.tolist()

    print(f"\n{'=' * 60}")
    print(f"  CONSENSUS FEATURE SELECTION RESULTS")
    print(f"  (top_k={top_k}, min_votes={min_votes})")
    print(f"{'=' * 60}")
    print(f"\n  {'Feature':<35} {'Votes':>6}  {'Mean Rank':>10}  {'Selected':>9}")
    print(f"  {'-'*35} {'-'*6}  {'-'*10}  {'-'*9}")
    for feat, row in scores_df[["total_votes", "mean_rank", "selected"]].iterrows():
        flag = "✓" if row["selected"] else ""
        print(f"  {feat:<35} {int(row['total_votes']):>6}  {row['mean_rank']:>10.2f}  {flag:>9}")

    print(f"\n  → {len(selected_features)} features selected: {selected_features}")
    return scores_df, selected_features


# ─────────────────────────────────────────────
# SAVE OUTPUTS
# ─────────────────────────────────────────────

def save_outputs(
    scores_df:        pd.DataFrame,
    selected_features: list,
    X_train:          pd.DataFrame,
    X_test:           pd.DataFrame,
    y_train:          pd.Series,
    y_test:           pd.Series,
    out_dir:          str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Full scores table
    scores_path = os.path.join(out_dir, "feature_scores_all_methods.csv")
    scores_df.to_csv(scores_path)
    print(f"\n[SAVE]  All method scores   →  {scores_path}")

    # Consensus ranking (selected flag included)
    rank_path = os.path.join(out_dir, "feature_rankings_consensus.csv")
    scores_df[["total_votes", "mean_rank", "selected"]].to_csv(rank_path)
    print(f"[SAVE]  Consensus rankings   →  {rank_path}")

    # Plain list of selected feature names
    txt_path = os.path.join(out_dir, "selected_features.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(selected_features))
    print(f"[SAVE]  Selected feature list →  {txt_path}")

    # Train / Test with selected features (+ target column)
    X_train_sel = X_train[selected_features].copy()
    X_test_sel  = X_test[selected_features].copy()

    train_out = X_train_sel.copy()
    train_out["Potability"] = y_train.values
    test_out  = X_test_sel.copy()
    test_out["Potability"]  = y_test.values

    train_path = os.path.join(out_dir, "X_train_selected.csv")
    test_path  = os.path.join(out_dir, "X_test_selected.csv")
    train_out.to_csv(train_path, index=False)
    test_out.to_csv(test_path,  index=False)
    print(f"[SAVE]  X_train_selected      →  {train_path}  shape: {X_train_sel.shape}")
    print(f"[SAVE]  X_test_selected       →  {test_path}   shape: {X_test_sel.shape}")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_feature_selection(
    X_train=None, X_test=None,
    y_train=None, y_test=None,
    scaler=None,
):
    """
    Entry point for the feature selection stage.

    Can be called:
      A) Standalone — loads from CSV files saved by preprocessing_water_quality.py
         result = run_feature_selection()

      B) Chained — pass DataFrames returned by preprocessing_water_quality.py
         result = run_feature_selection(X_train, X_test, y_train, y_test, scaler)

    Returns
    ───────
    dict with keys:
      X_train_sel        : pd.DataFrame  — training features, selected subset
      X_test_sel         : pd.DataFrame  — test features, selected subset
      y_train            : pd.Series
      y_test             : pd.Series
      selected_features  : list[str]
      feature_rankings   : pd.DataFrame  — full audit trail
      scaler             : RobustScaler (passed through unchanged)
    """
    print("=" * 60)
    print("  WATER QUALITY — FEATURE SELECTION PIPELINE")
    print("=" * 60)

    # ── Load data if not passed in ────────────────────────────────
    if X_train is None:
        X_train, X_test, y_train, y_test = _load_from_csv(
            CONFIG['train_csv'],
            CONFIG['test_csv'],
            CONFIG['target_col'],
        )

    all_features = X_train.columns.tolist()
    print(f"\n[INFO]  Total features entering selection: {len(all_features)}")
    print(f"        Features: {all_features}")

    # ── Method 1: Variance ───────────────────────────────────────
    variance_scores, kept_variance = variance_filter(
        X_train, CONFIG['variance_threshold']
    )

    # ── Method 2: Correlation ────────────────────────────────────
    corr_matrix, corr_drop = correlation_filter(
        X_train, CONFIG['corr_threshold']
    )

    # ── Method 3: ANOVA F-test ───────────────────────────────────
    anova_scores, anova_pvals, anova_top = anova_f_test(
        X_train, y_train, CONFIG['k_best']
    )

    # ── Method 4: Mutual Information ─────────────────────────────
    mi_scores, mi_top = mutual_information(
        X_train, y_train, CONFIG['k_best']
    )

    # ── Method 5: RF MDI Importance ──────────────────────────────
    rf_mdi_scores, fitted_rf = rf_importance(X_train, y_train)

    # ── Method 6: Permutation Importance ─────────────────────────
    perm_scores, perm_top = permutation_imp(X_train, y_train, fitted_rf)

    # ── Method 7: RFE ────────────────────────────────────────────
    rfe_ranking, rfe_selected = rfe_selection(
        X_train, y_train, CONFIG['rfe_n_features']
    )

    # ── Method 8: Consensus ──────────────────────────────────────
    scores_df, selected_features = consensus_ranking(
        all_features    = all_features,
        variance_scores = variance_scores,
        anova_scores    = anova_scores,
        mi_scores       = mi_scores,
        rf_mdi_scores   = rf_mdi_scores,
        perm_scores     = perm_scores,
        rfe_ranking     = rfe_ranking,
        top_k           = CONFIG['consensus_top_k'],
        min_votes       = CONFIG['consensus_min_votes'],
    )

    # ── Apply selection to data ───────────────────────────────────
    X_train_sel = X_train[selected_features].copy()
    X_test_sel  = X_test[selected_features].copy()

    # ── Save ─────────────────────────────────────────────────────
    save_outputs(
        scores_df, selected_features,
        X_train, X_test,
        y_train, y_test,
        CONFIG['output_dir'],
    )

    print(f"\n[DONE]  Feature selection complete.")
    print(f"        {len(all_features)} → {len(selected_features)} features retained.")
    print(f"        Output dir: {CONFIG['output_dir']}/")

    return {
        "X_train_sel":       X_train_sel,
        "X_test_sel":        X_test_sel,
        "y_train":           y_train,
        "y_test":            y_test,
        "selected_features": selected_features,
        "feature_rankings":  scores_df,
        "scaler":            scaler,
    }


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # ── Option A: run standalone after preprocessing has saved CSVs ──
    result = run_feature_selection()

    # ── Option B: chain directly from preprocessing (uncomment below) ─
    # import sys, os
    # sys.path.insert(0, os.path.dirname(__file__))
    # from preprocessing_water_quality import preprocess_pipeline
    # X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(
    #     filepath="water_potability.csv",
    #     output_dir="preprocessed_data"
    # )
    # result = run_feature_selection(X_train, X_test, y_train, y_test, scaler)
