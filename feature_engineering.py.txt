# ============================================================
# feature_engineering.py
# Water Quality Classification — Full EDA + Feature Engineering
# All plots are embedded inline (base64) and displayed via
# IPython.display — no external image files required.
#
# Run in Jupyter / Colab:  %run feature_engineering.py
# Run as script:           python feature_engineering.py
# ============================================================

import warnings, os, io, base64
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings("ignore")

# ── Try IPython inline display; fall back to plt.show() ──────────
try:
    from IPython.display import display, Image as IPImage, HTML
    IN_NOTEBOOK = True
except ImportError:
    IN_NOTEBOOK = False

def _show(fig, title=""):
    """Render figure inline (notebook) or via plt.show() (script)."""
    if IN_NOTEBOOK:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        buf.seek(0)
        display(IPImage(data=buf.read()))
        buf.close()
    else:
        plt.show()
    plt.close(fig)

# ── Style ─────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
PALETTE  = {0: "#E74C3C", 1: "#2ECC71", "0": "#E74C3C", "1": "#2ECC71"}
BLUE     = "#2980B9"
ACCENT   = "#8E44AD"
ORANGE   = "#E67E22"

# ── Column groups ─────────────────────────────────────────────────
ORIGINAL_FEATS = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity",
]
ENGINEERED_FEATS = [
    "ph_deviation", "mineral_load", "chloramines_conductivity",
    "thm_per_carbon", "turbidity_solids_ratio",
]
ALL_FEATS = ORIGINAL_FEATS + ENGINEERED_FEATS
TARGET    = "Potability"

TRAIN_CSV = "train_preprocessed.csv"
TEST_CSV  = "test_preprocessed.csv"

# ═══════════════════════════════════════════════════════════════
# SECTION 0 — LOAD & OVERVIEW
# ═══════════════════════════════════════════════════════════════

def section_header(title):
    sep = "═" * 65
    print(f"\n{sep}\n  {title}\n{sep}")

def load_data():
    section_header("0. DATA LOADING & OVERVIEW")
    train = pd.read_csv(TRAIN_CSV)
    test  = pd.read_csv(TEST_CSV)
    print(f"  Train : {train.shape[0]:>5} rows × {train.shape[1]} cols")
    print(f"  Test  : {test.shape[0]:>5} rows × {test.shape[1]} cols")
    print(f"  Missing (train): {train.isnull().sum().sum()}")
    print(f"  Missing (test) : {test.isnull().sum().sum()}")
    print("\n  Train dtypes:")
    print(train.dtypes.to_string())
    return train, test

# ═══════════════════════════════════════════════════════════════
# SECTION 1 — TARGET DISTRIBUTION
# ═══════════════════════════════════════════════════════════════

def eda_target(train, test):
    section_header("1. TARGET DISTRIBUTION")
    for split, df in [("Train", train), ("Test", test)]:
        vc = df[TARGET].value_counts().sort_index()
        print(f"  [{split}]  Not Potable: {vc[0]}  |  Potable: {vc[1]}  |  Ratio: {vc[0]/vc[1]:.2f}:1")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Figure 1 — Target (Potability) Distribution", fontsize=14, fontweight="bold")
    for ax, df, title in zip(axes, [train, test], ["Train Set (after SMOTE)", "Test Set"]):
        counts = df[TARGET].value_counts().sort_index()
        bars = ax.bar(["Not Potable (0)", "Potable (1)"], counts.values,
                      color=[PALETTE[0], PALETTE[1]], edgecolor="white", linewidth=1.5, width=0.5)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+15,
                    f"{val}\n({val/len(df)*100:.1f}%)", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("Count")
        ax.set_ylim(0, counts.max() * 1.2)
    plt.tight_layout()
    _show(fig, "Target Distribution")

# ═══════════════════════════════════════════════════════════════
# SECTION 2 — FEATURE DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════

def eda_distributions(train):
    section_header("2. FEATURE DISTRIBUTIONS (Histogram + KDE by Class)")
    ncols = 3
    nrows = int(np.ceil(len(ORIGINAL_FEATS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.5))
    fig.suptitle("Figure 2 — Original Feature Distributions by Potability",
                 fontsize=14, fontweight="bold", y=1.01)
    for ax, feat in zip(axes.flat, ORIGINAL_FEATS):
        for label, grp in train.groupby(TARGET):
            grp[feat].plot.hist(ax=ax, bins=40, alpha=0.45, color=PALETTE[label],
                                density=True, label=f"{'Potable' if label else 'Not Potable'}")
            grp[feat].plot.kde(ax=ax, color=PALETTE[label], linewidth=2)
        ax.set_title(feat, fontsize=10, fontweight="bold")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
    for ax in axes.flat[len(ORIGINAL_FEATS):]:
        ax.set_visible(False)
    plt.tight_layout()
    _show(fig)

# ═══════════════════════════════════════════════════════════════
# SECTION 3 — BOX PLOTS
# ═══════════════════════════════════════════════════════════════

def eda_boxplots(train):
    section_header("3. BOX PLOTS by Class + T-Test Significance")
    ncols = 3
    nrows = int(np.ceil(len(ORIGINAL_FEATS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.5))
    fig.suptitle("Figure 3 — Box-plots: Features vs. Potability (* p<0.05, ** p<0.01, *** p<0.001)",
                 fontsize=13, fontweight="bold", y=1.01)
    for ax, feat in zip(axes.flat, ORIGINAL_FEATS):
        sns.boxplot(data=train, x=TARGET, y=feat, ax=ax, palette=PALETTE,
                    width=0.5, linewidth=1.2,
                    flierprops=dict(marker="o", markersize=2, alpha=0.4))
        g0 = train.loc[train[TARGET]==0, feat]
        g1 = train.loc[train[TARGET]==1, feat]
        _, p = stats.ttest_ind(g0, g1)
        sig = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else "ns"))
        ax.set_title(f"{feat}  (p={p:.3f} {sig})", fontsize=9, fontweight="bold")
        ax.set_xticklabels(["Not Potable", "Potable"])
        ax.set_xlabel("")
    for ax in axes.flat[len(ORIGINAL_FEATS):]:
        ax.set_visible(False)
    plt.tight_layout()
    _show(fig)

# ═══════════════════════════════════════════════════════════════
# SECTION 4 — CORRELATION HEATMAP
# ═══════════════════════════════════════════════════════════════

def eda_correlation(train):
    section_header("4. CORRELATION HEATMAP")
    corr = train[ALL_FEATS + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, annot=True, fmt=".2f",
                annot_kws={"size": 7}, linewidths=0.4, ax=ax, vmin=-1, vmax=1,
                cbar_kws={"shrink": 0.75})
    ax.set_title("Figure 4 — Correlation Heatmap (all features + target)",
                 fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    _show(fig)
    tc = corr[TARGET].drop(TARGET).abs().sort_values(ascending=False)
    print("  Top correlations with Potability:")
    for f, v in tc.items():
        print(f"    {f:<30} {v:.4f}")

# ═══════════════════════════════════════════════════════════════
# SECTION 5 — PAIR PLOT
# ═══════════════════════════════════════════════════════════════

def eda_pairplot(train):
    section_header("5. PAIR PLOT (core 5 parameters, 600-sample subset)")
    core = ["ph", "Hardness", "Chloramines", "Sulfate", "Conductivity", TARGET]
    sample = train[core].sample(min(600, len(train)), random_state=42).copy()
    sample[TARGET] = sample[TARGET].map({0: "Not Potable", 1: "Potable"})
    g = sns.pairplot(sample, hue=TARGET,
                     palette={"Not Potable": PALETTE[0], "Potable": PALETTE[1]},
                     diag_kind="kde", plot_kws=dict(alpha=0.4, s=15), corner=True)
    g.figure.suptitle("Figure 5 — Pair-plot: Core Water Parameters",
                       fontsize=13, fontweight="bold", y=1.01)
    if IN_NOTEBOOK:
        buf = io.BytesIO()
        g.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        buf.seek(0); display(IPImage(data=buf.read())); buf.close()
    else:
        plt.show()
    plt.close("all")

# ═══════════════════════════════════════════════════════════════
# SECTION 6 — VIOLIN PLOTS
# ═══════════════════════════════════════════════════════════════

def eda_violin(train):
    section_header("6. VIOLIN PLOTS — Feature Spread by Class")
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle("Figure 6 — Violin Plots: Feature Spread by Potability",
                 fontsize=14, fontweight="bold")
    for ax, feat in zip(axes.flat, ORIGINAL_FEATS):
        sns.violinplot(data=train, x=TARGET, y=feat, ax=ax,
                       palette=PALETTE, inner="box", linewidth=1)
        ax.set_title(feat, fontsize=10, fontweight="bold")
        ax.set_xticklabels(["Not Potable", "Potable"])
        ax.set_xlabel("")
    plt.tight_layout()
    _show(fig)

# ═══════════════════════════════════════════════════════════════
# SECTION 7 — CLASS MEAN COMPARISON
# ═══════════════════════════════════════════════════════════════

def eda_class_means(train):
    section_header("7. CLASS MEAN COMPARISON")
    means = train[ORIGINAL_FEATS + [TARGET]].groupby(TARGET)[ORIGINAL_FEATS].mean()
    diff  = (means.loc[1] - means.loc[0]).sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Figure 7 — Mean Feature Values by Class", fontsize=14, fontweight="bold")
    x = np.arange(len(ORIGINAL_FEATS)); w = 0.35
    ax = axes[0]
    ax.bar(x-w/2, means.loc[0].values, width=w, label="Not Potable",
           color=PALETTE[0], alpha=0.85, edgecolor="white")
    ax.bar(x+w/2, means.loc[1].values, width=w, label="Potable",
           color=PALETTE[1], alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(ORIGINAL_FEATS, rotation=30, ha="right", fontsize=9)
    ax.set_title("Mean Values per Class (RobustScaled)"); ax.legend()
    ax2 = axes[1]
    colors = [PALETTE[1] if v > 0 else PALETTE[0] for v in diff.values]
    ax2.barh(diff.index, diff.values, color=colors, edgecolor="white")
    ax2.axvline(0, color="black", linewidth=1)
    ax2.set_title("Δ Mean (Potable − Not Potable)")
    ax2.set_xlabel("Difference in scaled mean")
    plt.tight_layout()
    _show(fig)

# ═══════════════════════════════════════════════════════════════
# SECTION 8 — RANDOM FOREST FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════

def eda_feature_importance(train):
    section_header("8. RANDOM FOREST FEATURE IMPORTANCE")
    X, y = train[ALL_FEATS], train[TARGET]
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=ALL_FEATS).sort_values()
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [ACCENT if f in ENGINEERED_FEATS else BLUE for f in imp.index]
    bars = ax.barh(imp.index, imp.values, color=colors, edgecolor="white")
    ax.set_xlabel("Mean Decrease in Impurity (Gini importance)")
    ax.set_title("Figure 8 — Random Forest Feature Importance\n(purple = engineered features)",
                 fontsize=12, fontweight="bold")
    for bar, val in zip(bars, imp.values):
        ax.text(val+0.001, bar.get_y()+bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=8)
    legend_els = [mpatches.Patch(facecolor=BLUE, label="Original"),
                  mpatches.Patch(facecolor=ACCENT, label="Engineered")]
    ax.legend(handles=legend_els, loc="lower right")
    plt.tight_layout()
    _show(fig)
    print("  Top 5 features:")
    for f, v in imp.sort_values(ascending=False).head(5).items():
        print(f"    {f:<30} {v:.5f}")
    return imp

# ═══════════════════════════════════════════════════════════════
# SECTION 9 — MUTUAL INFORMATION
# ═══════════════════════════════════════════════════════════════

def eda_mutual_info(train):
    section_header("9. MUTUAL INFORMATION SCORES")
    X, y = train[ALL_FEATS], train[TARGET]
    mi = mutual_info_classif(X, y, random_state=42)
    mi_s = pd.Series(mi, index=ALL_FEATS).sort_values()
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [ACCENT if f in ENGINEERED_FEATS else BLUE for f in mi_s.index]
    ax.barh(mi_s.index, mi_s.values, color=colors, edgecolor="white")
    ax.set_xlabel("Mutual Information Score")
    ax.set_title("Figure 9 — Mutual Information: Features vs. Potability\n(purple = engineered)",
                 fontsize=12, fontweight="bold")
    legend_els = [mpatches.Patch(facecolor=BLUE, label="Original"),
                  mpatches.Patch(facecolor=ACCENT, label="Engineered")]
    ax.legend(handles=legend_els, loc="lower right")
    plt.tight_layout()
    _show(fig)

# ═══════════════════════════════════════════════════════════════
# SECTION 10 — ENGINEERED FEATURE DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════

def eda_engineered_dist(train):
    section_header("10. ENGINEERED FEATURE DISTRIBUTIONS")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Figure 10 — Engineered Feature Distributions by Potability",
                 fontsize=14, fontweight="bold")
    for ax, feat in zip(axes.flat, ENGINEERED_FEATS):
        for label, grp in train.groupby(TARGET):
            vals = grp[feat].clip(grp[feat].quantile(0.01), grp[feat].quantile(0.99))
            vals.plot.hist(ax=ax, bins=40, alpha=0.45, color=PALETTE[label],
                           density=True, label=f"{'Potable' if label else 'Not Potable'}")
            try: vals.plot.kde(ax=ax, color=PALETTE[label], linewidth=2)
            except: pass
        ax.set_title(feat, fontsize=10, fontweight="bold")
        ax.set_ylabel("Density"); ax.legend(fontsize=8)
    axes.flat[-1].set_visible(False)
    plt.tight_layout()
    _show(fig)

# ═══════════════════════════════════════════════════════════════
# SECTION 11 — ENGINEERED FEATURES vs POTABILITY
# ═══════════════════════════════════════════════════════════════

def eda_engineered_vs_target(train):
    section_header("11. ENGINEERED FEATURES vs. POTABILITY (Box plots)")
    fig, axes = plt.subplots(1, len(ENGINEERED_FEATS), figsize=(18, 5))
    fig.suptitle("Figure 11 — Engineered Features vs. Potability",
                 fontsize=13, fontweight="bold")
    for ax, feat in zip(axes, ENGINEERED_FEATS):
        lo, hi = train[feat].quantile(0.01), train[feat].quantile(0.99)
        tmp = train[[feat, TARGET]].copy()
        tmp[feat] = tmp[feat].clip(lo, hi)
        sns.boxplot(data=tmp, x=TARGET, y=feat, ax=ax, palette=PALETTE,
                    width=0.5, linewidth=1.2,
                    flierprops=dict(marker="o", markersize=2, alpha=0.3))
        ax.set_title(feat, fontsize=9, fontweight="bold")
        ax.set_xticklabels(["Not\nPotable", "Potable"], fontsize=8)
        ax.set_xlabel("")
    plt.tight_layout()
    _show(fig)

# ═══════════════════════════════════════════════════════════════
# SECTION 12 — TRAIN vs TEST DISTRIBUTION (KS TEST)
# ═══════════════════════════════════════════════════════════════

def eda_train_test_compare(train, test):
    section_header("12. TRAIN vs TEST DISTRIBUTION CHECK (KS-Test)")
    feats = ORIGINAL_FEATS[:6]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Figure 12 — Train vs Test Distribution Comparison",
                 fontsize=14, fontweight="bold")
    for ax, feat in zip(axes.flat, feats):
        train[feat].plot.kde(ax=ax, color=BLUE, linewidth=2, label="Train")
        test[feat].plot.kde(ax=ax, color=ORANGE, linewidth=2, label="Test", linestyle="--")
        ks, p = stats.ks_2samp(train[feat], test[feat])
        ax.set_title(f"{feat}\n(KS stat={ks:.3f}, p={p:.3f})", fontsize=9, fontweight="bold")
        ax.legend(fontsize=8)
    plt.tight_layout()
    _show(fig)

# ═══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING — NEW FEATURES
# ═══════════════════════════════════════════════════════════════

def apply_feature_engineering(df: pd.DataFrame, verbose=True) -> pd.DataFrame:
    """
    Applies all domain-based feature engineering.
    Works on any split — train or test.
    Input: DataFrame with the 9 original (scaled) water quality features.
    """
    df = df.copy()

    # ── Original 5 engineered features (already in preprocessed CSV) ──
    df["ph_deviation"]              = df["ph"].abs()
    df["mineral_load"]              = df["Hardness"] / (df["Solids"].replace(0, 1e-9))
    df["chloramines_conductivity"]  = df["Chloramines"] * df["Conductivity"]
    df["thm_per_carbon"]            = df["Trihalomethanes"] / (df["Organic_carbon"].replace(0, 1e-9))
    df["turbidity_solids_ratio"]    = df["Turbidity"] / (df["Solids"].replace(0, 1e-9))

    # ── 5 Additional domain-informed features ─────────────────────────
    # pH × Chloramines — disinfection effectiveness proxy
    df["ph_chloramines"]            = df["ph"] * df["Chloramines"]

    # Hardness / Conductivity — dissolved ion concentration proxy
    df["hardness_conductivity"]     = df["Hardness"] / (df["Conductivity"].replace(0, 1e-9))

    # Solids × Turbidity — physical contamination severity index
    df["solids_turbidity"]          = df["Solids"] * df["Turbidity"]

    # Sulfate / Conductivity — relative sulfate ion contribution
    df["sulfate_conductivity"]      = df["Sulfate"] / (df["Conductivity"].replace(0, 1e-9))

    # Quality stress index — combines pH deviation and turbidity
    df["quality_stress"]            = df["ph"].abs() + df["Turbidity"]

    if verbose:
        print(f"  Applied feature engineering → {df.shape[1]} total columns")
    return df

def feature_engineering_summary(train):
    section_header("13. FEATURE ENGINEERING SUMMARY")
    new_feats = ["ph_chloramines", "hardness_conductivity",
                 "solids_turbidity", "sulfate_conductivity", "quality_stress"]
    train_fe = apply_feature_engineering(train)
    print("\n  New engineered features — mean by class:")
    print(train_fe[new_feats + [TARGET]].groupby(TARGET)[new_feats].mean().round(4).to_string())
    print("\n  Statistical significance (Welch t-test):")
    for feat in new_feats:
        g0 = train_fe.loc[train_fe[TARGET]==0, feat]
        g1 = train_fe.loc[train_fe[TARGET]==1, feat]
        _, p = stats.ttest_ind(g0, g1)
        sig = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else "ns"))
        print(f"    {feat:<30}  p={p:.4f}  {sig}")
    return train_fe

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    train, test = load_data()
    eda_target(train, test)
    eda_distributions(train)
    eda_boxplots(train)
    eda_correlation(train)
    eda_pairplot(train)
    eda_violin(train)
    eda_class_means(train)
    eda_feature_importance(train)
    eda_mutual_info(train)
    eda_engineered_dist(train)
    eda_engineered_vs_target(train)
    eda_train_test_compare(train, test)
    train_fe = feature_engineering_summary(train)
    test_fe  = apply_feature_engineering(test, verbose=False)
    train_fe.to_csv("train_feature_engineered.csv", index=False)
    test_fe.to_csv("test_feature_engineered.csv",  index=False)
    print("\n  Saved → train_feature_engineered.csv")
    print("  Saved → test_feature_engineered.csv")
    print("\n[DONE] EDA + Feature Engineering complete.")

if __name__ == "__main__":
    main()
