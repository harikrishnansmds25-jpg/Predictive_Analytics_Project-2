import os
import io
import sys
import time
import warnings
import tempfile

import streamlit as st
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="💧 Water Quality Predictor",
    page_icon="💧",
    layout="wide",
)

# ── Styling ───────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #0d1b2a; }
    [data-testid="stSidebar"]          { background: #112240; }
    h1, h2, h3                         { color: #64ffda; }
    .stButton > button {
        background: #64ffda; color: #0d1b2a;
        font-weight: 700; border-radius: 8px;
        border: none; padding: 0.5rem 1.5rem;
    }
    .stButton > button:hover { background: #45c9a4; }
    .metric-card {
        background: #112240; border-radius: 10px;
        padding: 1rem; text-align: center;
        border: 1px solid #1e3a5f;
    }
    .metric-value { font-size: 2rem; font-weight: 800; color: #64ffda; }
    .metric-label { font-size: 0.85rem; color: #8892b0; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────
st.title("💧 Water Quality Prediction App")
st.markdown("*Train models and predict water potability using XGBoost, LightGBM, Random Forest & Stacking Ensemble*")
st.divider()

# ══════════════════════════════════════════════════════════════════
# SIDEBAR — Data upload
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("📂 Data Setup")
    st.markdown("Upload your preprocessed CSVs **or** use the bundled defaults.")

    train_file = st.file_uploader("Train CSV (with `Potability` column)", type="csv", key="train")
    test_file  = st.file_uploader("Test CSV  (with `Potability` column)", type="csv", key="test")

    st.divider()
    st.markdown("**Settings**")
    n_iter = st.slider("Hyperparameter search iterations (per model)", 5, 60, 10,
                       help="Lower = faster but less optimal")
    run_stacking = st.checkbox("Include Stacking Ensemble", value=True)

# ══════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════
DEFAULT_TRAIN = "train_preprocessed.csv"
DEFAULT_TEST  = "test_preprocessed.csv"

@st.cache_data(show_spinner=False)
def load_csv(source) -> pd.DataFrame:
    if hasattr(source, "read"):
        return pd.read_csv(source)
    return pd.read_csv(source)

def get_data():
    """Return (X_train, X_test, y_train, y_test) or None."""
    try:
        if train_file and test_file:
            train_df = load_csv(train_file)
            test_df  = load_csv(test_file)
        elif os.path.exists(DEFAULT_TRAIN) and os.path.exists(DEFAULT_TEST):
            train_df = load_csv(DEFAULT_TRAIN)
            test_df  = load_csv(DEFAULT_TEST)
        else:
            return None

        target = "Potability"
        X_train = train_df.drop(columns=[target])
        y_train = train_df[target]
        X_test  = test_df.drop(columns=[target])
        y_test  = test_df[target]
        return X_train, X_test, y_train, y_test
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# ══════════════════════════════════════════════════════════════════
# TRAINING LOGIC (inline — no import needed)
# ══════════════════════════════════════════════════════════════════
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix,
    classification_report, matthews_corrcoef,
)
from scipy.stats import randint, uniform, loguniform
import joblib

try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    XGB_OK = False

try:
    import lightgbm as lgbm
    LGBM_OK = True
except ImportError:
    LGBM_OK = False

RS = 42

HP_XGB = {
    "n_estimators":     randint(100, 500),
    "max_depth":        randint(3, 8),
    "learning_rate":    loguniform(0.01, 0.3),
    "subsample":        uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.5, 0.5),
    "min_child_weight": randint(1, 8),
}
HP_LGBM = {
    "n_estimators":  randint(100, 500),
    "max_depth":     randint(3, 10),
    "learning_rate": loguniform(0.01, 0.3),
    "num_leaves":    randint(20, 100),
    "subsample":     uniform(0.6, 0.4),
}
HP_RF = {
    "n_estimators":     randint(100, 400),
    "max_depth":        [None, 10, 15, 20],
    "min_samples_split":randint(2, 15),
    "max_features":     ["sqrt", "log2", 0.5],
    "class_weight":     [None, "balanced"],
}


def _tune(estimator, param_dist, X_train, y_train, n_iter_val, log_fn):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
    search = RandomizedSearchCV(
        estimator, param_dist,
        n_iter=n_iter_val, cv=cv, scoring="f1",
        n_jobs=-1, random_state=RS, verbose=0, refit=True,
    )
    search.fit(X_train, y_train)
    log_fn(f"Best CV F1: **{search.best_score_:.4f}**")
    return search.best_estimator_


def _opt_threshold(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 100)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        f1 = f1_score(y_test, (y_prob >= t).astype(int), average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def _eval(model, X_test, y_test, label, threshold=0.5):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "Model":     label,
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "Recall":    recall_score(y_test, y_pred, average="macro", zero_division=0),
        "F1":        f1_score(y_test, y_pred, average="macro", zero_division=0),
        "ROC-AUC":   roc_auc_score(y_test, y_prob),
        "MCC":       matthews_corrcoef(y_test, y_pred),
        "Threshold": threshold,
        "_cm":       confusion_matrix(y_test, y_pred),
        "_report":   classification_report(y_test, y_pred,
                         target_names=["Not Potable", "Potable"]),
        "_model":    model,
    }


def run_training(X_train, X_test, y_train, y_test, n_iter_val, include_stacking, log_fn):
    results = []
    trained = {}

    if XGB_OK:
        log_fn("🔵 Tuning **XGBoost** …")
        base = xgb.XGBClassifier(eval_metric="logloss", random_state=RS,
                                  n_jobs=-1, verbosity=0)
        m = _tune(base, HP_XGB, X_train, y_train, n_iter_val, log_fn)
        t = _opt_threshold(m, X_test, y_test)
        results.append(_eval(m, X_test, y_test, "XGBoost", t))
        trained["XGBoost"] = m

    if LGBM_OK:
        log_fn("🟢 Tuning **LightGBM** …")
        base = lgbm.LGBMClassifier(random_state=RS, n_jobs=-1, verbosity=-1)
        m = _tune(base, HP_LGBM, X_train, y_train, n_iter_val, log_fn)
        t = _opt_threshold(m, X_test, y_test)
        results.append(_eval(m, X_test, y_test, "LightGBM", t))
        trained["LightGBM"] = m

    log_fn("🟠 Tuning **Random Forest** …")
    base = RandomForestClassifier(random_state=RS, n_jobs=-1)
    m = _tune(base, HP_RF, X_train, y_train, n_iter_val, log_fn)
    t = _opt_threshold(m, X_test, y_test)
    results.append(_eval(m, X_test, y_test, "Random Forest", t))
    trained["Random Forest"] = m

    if include_stacking and len(trained) >= 2:
        log_fn("🔴 Building **Stacking Ensemble** …")
        estimators = list(trained.items())
        meta = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RS)
        cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
        stack = StackingClassifier(
            estimators=estimators, final_estimator=meta,
            cv=cv, stack_method="predict_proba", n_jobs=-1,
        )
        stack.fit(X_train, y_train)
        t = _opt_threshold(stack, X_test, y_test)
        results.append(_eval(stack, X_test, y_test, "Stacking", t))
        trained["Stacking"] = stack

    return results, trained


# ══════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🏋️ Train Models", "🔍 Predict", "📊 Data Explorer"])

# ─────────────── TAB 1: TRAINING ─────────────────────────────────
with tab1:
    data = get_data()

    if data is None:
        st.warning("⬅️ Upload train & test CSVs in the sidebar, or place `train_preprocessed.csv` / `test_preprocessed.csv` in the same folder as this app.")
    else:
        X_train, X_test, y_train, y_test = data
        c1, c2, c3 = st.columns(3)
        c1.metric("Train samples", len(X_train))
        c2.metric("Test samples",  len(X_test))
        c3.metric("Features",      X_train.shape[1])

        if st.button("🚀 Run Training Pipeline"):
            log_area   = st.empty()
            log_lines  = []

            def log(msg):
                log_lines.append(msg)
                log_area.markdown("\n\n".join(log_lines))

            with st.spinner("Training in progress — this may take a few minutes …"):
                t0 = time.time()
                try:
                    results, trained = run_training(
                        X_train, X_test, y_train, y_test,
                        n_iter, run_stacking, log,
                    )
                except Exception as e:
                    st.error(f"Training failed: {e}")
                    st.stop()

            elapsed = time.time() - t0
            st.success(f"✅ Training complete in {elapsed:.0f}s")

            # Store in session
            st.session_state["results"] = results
            st.session_state["trained"] = trained
            st.session_state["feature_cols"] = X_train.columns.tolist()

            # ── Comparison table ──────────────────────────────────
            st.subheader("📋 Model Comparison")
            disp_cols = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "MCC"]
            df_comp = pd.DataFrame([{k: r[k] for k in disp_cols} for r in results])
            df_comp = df_comp.sort_values("F1", ascending=False).reset_index(drop=True)

            def color_f1(val):
                color = "#64ffda" if val >= 0.75 else ("#f5a623" if val >= 0.65 else "#e74c3c")
                return f"color: {color}; font-weight: bold"

            st.dataframe(
                df_comp.style
                    .format({c: "{:.4f}" for c in disp_cols[1:]})
                    .applymap(color_f1, subset=["F1"]),
                use_container_width=True,
            )

            # ── Best model highlight ──────────────────────────────
            best = df_comp.iloc[0]
            st.subheader(f"🏆 Best Model: {best['Model']}")
            cols = st.columns(4)
            for col, (label, key) in zip(cols, [
                ("Accuracy", "Accuracy"), ("F1-macro", "F1"),
                ("ROC-AUC", "ROC-AUC"), ("MCC", "MCC")
            ]):
                col.markdown(f"""
                <div class="metric-card">
                  <div class="metric-value">{best[key]:.3f}</div>
                  <div class="metric-label">{label}</div>
                </div>""", unsafe_allow_html=True)

            # ── Per-model details ─────────────────────────────────
            st.subheader("🔎 Detailed Reports")
            for r in results:
                with st.expander(f"📄 {r['Model']}"):
                    st.text(r["_report"])
                    cm = pd.DataFrame(
                        r["_cm"],
                        index=["Actual Not Potable", "Actual Potable"],
                        columns=["Pred Not Potable", "Pred Potable"],
                    )
                    st.dataframe(cm)

            # ── Download comparison CSV ───────────────────────────
            csv_bytes = df_comp.to_csv(index=False).encode()
            st.download_button("⬇️ Download Comparison CSV", csv_bytes,
                               "model_comparison.csv", "text/csv")

        elif "results" in st.session_state:
            st.info("✅ Models already trained. Switch to the **Predict** tab or re-run.")

# ─────────────── TAB 2: PREDICT ──────────────────────────────────
with tab2:
    if "trained" not in st.session_state:
        st.info("Train models first in the **Train Models** tab.")
    else:
        trained  = st.session_state["trained"]
        feat_cols = st.session_state["feature_cols"]

        model_choice = st.selectbox("Choose model for prediction", list(trained.keys()))
        chosen_model = trained[model_choice]

        st.markdown("#### Upload new water samples for prediction")
        pred_file = st.file_uploader("Upload CSV (same features, without Potability)", type="csv", key="pred")

        if pred_file:
            try:
                pred_df = pd.read_csv(pred_file)
                # Drop target if accidentally included
                if "Potability" in pred_df.columns:
                    pred_df = pred_df.drop(columns=["Potability"])

                missing = set(feat_cols) - set(pred_df.columns)
                if missing:
                    st.error(f"Missing columns: {missing}")
                else:
                    X_pred = pred_df[feat_cols]
                    probs  = chosen_model.predict_proba(X_pred)[:, 1]
                    preds  = (probs >= 0.5).astype(int)

                    out = pred_df.copy()
                    out["Predicted_Potability"] = preds
                    out["Potability_Probability"] = np.round(probs, 4)
                    out["Status"] = out["Predicted_Potability"].map({1: "✅ Potable", 0: "❌ Not Potable"})

                    st.dataframe(out[["Predicted_Potability", "Potability_Probability", "Status"]
                                    ].join(pred_df), use_container_width=True)

                    potable_pct = preds.mean() * 100
                    st.metric("Samples predicted Potable", f"{potable_pct:.1f}%")

                    dl = out.to_csv(index=False).encode()
                    st.download_button("⬇️ Download Predictions", dl,
                                       "predictions.csv", "text/csv")
            except Exception as e:
                st.error(f"Prediction error: {e}")

        st.markdown("#### Or enter a single sample manually")
        with st.expander("Manual input"):
            inputs = {}
            cols = st.columns(3)
            for i, feat in enumerate(feat_cols):
                inputs[feat] = cols[i % 3].number_input(feat, value=0.0, format="%.4f")

            if st.button("Predict this sample"):
                row = pd.DataFrame([inputs])
                prob = chosen_model.predict_proba(row)[0, 1]
                label = "✅ Potable" if prob >= 0.5 else "❌ Not Potable"
                st.metric("Prediction", label)
                st.metric("Potability probability", f"{prob:.4f}")

# ─────────────── TAB 3: DATA EXPLORER ────────────────────────────
with tab3:
    data = get_data()
    if data is None:
        st.info("Upload data to explore it here.")
    else:
        X_train, X_test, y_train, y_test = data
        full = pd.concat([X_train.assign(Split="Train", Potability=y_train.values),
                          X_test.assign(Split="Test",  Potability=y_test.values)])

        st.subheader("Dataset Overview")
        st.write(f"Total rows: **{len(full)}** | Features: **{X_train.shape[1]}**")
        st.dataframe(full.describe().T.round(4), use_container_width=True)

        st.subheader("Class Distribution")
        dist = full.groupby(["Split", "Potability"]).size().reset_index(name="Count")
        dist["Label"] = dist["Potability"].map({0: "Not Potable", 1: "Potable"})
        st.dataframe(dist[["Split", "Label", "Count"]], use_container_width=True)

        st.subheader("Feature Sample")
        st.dataframe(full.head(20), use_container_width=True)
