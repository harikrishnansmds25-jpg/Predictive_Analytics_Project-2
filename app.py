import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import joblib

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="💧 Water Quality Analyzer",
    page_icon="💧",
    layout="centered",
)

st.markdown("""
<style>
/* ── Sidebar hidden ───────────────────────────────────────────── */
[data-testid="stSidebar"] { display: none; }

/* ── Light mode (default) ─────────────────────────────────────── */
[data-testid="stAppViewContainer"] { background: #f0f7ff; }

.safe-box {
    background: #d4edda; border: 2px solid #28a745;
    border-radius: 14px; padding: 2rem; text-align: center;
}
.safe-box h2 { color: #155724; font-size: 2.2rem; margin: 0; }
.safe-box p  { color: #155724; margin: 0.3rem 0 0; font-size: 1rem; }

.unsafe-box {
    background: #f8d7da; border: 2px solid #dc3545;
    border-radius: 14px; padding: 2rem; text-align: center;
}
.unsafe-box h2 { color: #721c24; font-size: 2.2rem; margin: 0; }
.unsafe-box p  { color: #721c24; margin: 0.3rem 0 0; font-size: 1rem; }

.stButton > button {
    background: #1a73e8; color: white; font-weight: 700;
    border: none; border-radius: 10px;
    padding: 0.6rem 2rem; font-size: 1rem; width: 100%;
}
.stButton > button:hover { background: #1558b0; }

/* ── Dark mode overrides ──────────────────────────────────────── */
@media (prefers-color-scheme: dark) {
    [data-testid="stAppViewContainer"] { background: #0e1117; }

    /* Streamlit auto sets dark bg, but we also fix the main content area */
    [data-testid="stMain"]            { background: #0e1117; }
    [data-testid="block-container"]   { background: #0e1117; }

    /* General text visibility */
    h1, h2, h3, h4, h5, h6, p, label, span, div {
        color: #e8eaed;
    }

    .safe-box {
        background: #1a3a2a; border: 2px solid #34c759;
    }
    .safe-box h2 { color: #7dff9e; }
    .safe-box p  { color: #a8f0b8; }

    .unsafe-box {
        background: #3a1a1a; border: 2px solid #ff453a;
    }
    .unsafe-box h2 { color: #ff8080; }
    .unsafe-box p  { color: #f0a8a8; }

    /* Expander / card backgrounds */
    [data-testid="stExpander"] {
        background: #1e2329 !important;
        border: 1px solid #2d3748 !important;
    }

    /* Info / warning / success / error boxes */
    [data-testid="stAlert"] { background: #1e2329; }

    /* File uploader area */
    [data-testid="stFileUploader"] {
        background: #1e2329;
        border: 2px dashed #4a5568;
        border-radius: 10px;
    }

    /* Dataframe / table */
    [data-testid="stDataFrame"] { background: #1e2329; }

    /* Number inputs */
    input[type="number"] {
        background: #1e2329 !important;
        color: #e8eaed !important;
        border: 1px solid #4a5568 !important;
    }

    /* Caption */
    [data-testid="stCaptionContainer"] { color: #8b9ab0; }

    /* Divider */
    hr { border-color: #2d3748; }
}

/* ── Streamlit's own dark theme class (used when user picks Dark in the menu) */
[data-theme="dark"] [data-testid="stAppViewContainer"],
.st-emotion-cache-1y4p8pa[data-theme="dark"] {
    background: #0e1117;
}

[data-theme="dark"] .safe-box {
    background: #1a3a2a; border: 2px solid #34c759;
}
[data-theme="dark"] .safe-box h2 { color: #7dff9e; }
[data-theme="dark"] .safe-box p  { color: #a8f0b8; }

[data-theme="dark"] .unsafe-box {
    background: #3a1a1a; border: 2px solid #ff453a;
}
[data-theme="dark"] .unsafe-box h2 { color: #ff8080; }
[data-theme="dark"] .unsafe-box p  { color: #f0a8a8; }
</style>
""", unsafe_allow_html=True)

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy.stats import randint, uniform, loguniform

try:
    import xgboost as xgb;   XGB_OK = True
except ImportError:
    XGB_OK = False

try:
    import lightgbm as lgbm; LGBM_OK = True
except ImportError:
    LGBM_OK = False

FEATURE_COLS = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity",
    "ph_deviation", "mineral_load", "chloramines_conductivity",
    "thm_per_carbon", "turbidity_solids_ratio",
]
TARGET      = "Potability"
TRAIN_CSV   = "train_preprocessed.csv"
TEST_CSV    = "test_preprocessed.csv"
MODEL_PATH  = "best_model.joblib"
THRESH_PATH = "best_threshold.txt"

# ── Reduced search spaces for fast training (~30-60 sec) ──────────
HP_XGB = {
    "n_estimators":     randint(50, 150),
    "max_depth":        randint(3, 7),
    "learning_rate":    loguniform(0.01, 0.3),
    "subsample":        uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.5, 0.5),
}
HP_LGBM = {
    "n_estimators":  randint(50, 150),
    "max_depth":     randint(3, 7),
    "learning_rate": loguniform(0.01, 0.3),
    "num_leaves":    randint(20, 60),
}
HP_RF = {
    "n_estimators":      randint(50, 150),
    "max_depth":         [None, 10, 15],
    "min_samples_split": randint(2, 10),
    "max_features":      ["sqrt", "log2"],
    "class_weight":      [None, "balanced"],
}


@st.cache_data(show_spinner=False)
def load_train_test():
    train = pd.read_csv(TRAIN_CSV)
    test  = pd.read_csv(TEST_CSV)
    return (train[FEATURE_COLS], test[FEATURE_COLS],
            train[TARGET],       test[TARGET])


def _tune(base, params, X, y, n_iter=5):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    s  = RandomizedSearchCV(
        base, params, n_iter=n_iter, cv=cv,
        scoring="f1", n_jobs=-1, random_state=42,
        verbose=0, refit=True,
    )
    s.fit(X, y)
    return s.best_estimator_


def _best_threshold(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.1, 0.9, 100):
        f1 = f1_score(y_test, (probs >= t).astype(int),
                      average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return round(float(best_t), 3)


def train_and_save():
    X_train, X_test, y_train, y_test = load_train_test()
    trained = {}
    total   = sum([XGB_OK, LGBM_OK, True, True])
    step    = 0
    bar     = st.progress(0, text="Starting …")

    if XGB_OK:
        bar.progress(step / total, "🔵 Training XGBoost …")
        base = xgb.XGBClassifier(
            eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0,
        )
        trained["XGBoost"] = _tune(base, HP_XGB, X_train, y_train)
        step += 1

    if LGBM_OK:
        bar.progress(step / total, "🟢 Training LightGBM …")
        base = lgbm.LGBMClassifier(random_state=42, n_jobs=-1, verbosity=-1)
        trained["LightGBM"] = _tune(base, HP_LGBM, X_train, y_train)
        step += 1

    bar.progress(step / total, "🟠 Training Random Forest …")
    trained["Random Forest"] = _tune(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        HP_RF, X_train, y_train,
    )
    step += 1

    bar.progress(step / total, "🔴 Building Stacking Ensemble …")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    stack = StackingClassifier(
        estimators=list(trained.items()),
        final_estimator=LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42,
        ),
        cv=cv, stack_method="predict_proba", n_jobs=-1,
    )
    stack.fit(X_train, y_train)
    trained["Stacking"] = stack
    step += 1
    bar.progress(1.0, "✅ Done!")

    best_name, best_model, best_f1, best_thresh = "", None, 0.0, 0.5
    rows = []
    for name, m in trained.items():
        t     = _best_threshold(m, X_test, y_test)
        probs = m.predict_proba(X_test)[:, 1]
        preds = (probs >= t).astype(int)
        f1    = f1_score(y_test, preds, average="macro", zero_division=0)
        acc   = accuracy_score(y_test, preds)
        auc   = roc_auc_score(y_test, probs)
        rows.append({
            "Model":    name,
            "Accuracy": round(acc, 4),
            "F1":       round(f1,  4),
            "ROC-AUC":  round(auc, 4),
        })
        if f1 > best_f1:
            best_f1, best_name, best_model, best_thresh = f1, name, m, t

    joblib.dump(best_model, MODEL_PATH)
    with open(THRESH_PATH, "w") as f:
        f.write(str(best_thresh))

    return best_name, pd.DataFrame(rows).sort_values("F1", ascending=False)


def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(THRESH_PATH):
        m = joblib.load(MODEL_PATH)
        with open(THRESH_PATH) as f:
            t = float(f.read().strip())
        return m, t
    return None, None


# ══════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════
st.title("💧 Water Quality Analyzer")
st.markdown("Upload a CSV of water sample data to check if the water is **safe to drink**.")
st.divider()

model, threshold = load_model()

# ── Step 1: Train ──────────────────────────────────────────────────
with st.expander(
    "⚙️ Model Training" + (
        " ✅  (trained — ready to predict)" if model
        else " ⚠️  Train before predicting"
    ),
    expanded=(model is None),
):
    if model is None:
        st.info(
            "Models need to be trained once. "
            "With optimised settings this takes about **30–60 seconds**. "
            "After training, the model is saved and loads instantly on every restart."
        )

    col_train, col_retrain = st.columns([3, 1])
    with col_train:
        if st.button("🚀 Train Models", disabled=(model is not None), use_container_width=True):
            try:
                best_name, comp_df = train_and_save()
                st.success(f"✅ Best model selected: **{best_name}**")
                st.dataframe(comp_df.reset_index(drop=True), use_container_width=True)
                model, threshold = load_model()
                st.rerun()
            except Exception as e:
                st.error(f"Training failed: {e}")

    with col_retrain:
        if model is not None:
            if st.button("🔄 Retrain", use_container_width=True,
                         help="Delete saved model and retrain from scratch"):
                if os.path.exists(MODEL_PATH):  os.remove(MODEL_PATH)
                if os.path.exists(THRESH_PATH): os.remove(THRESH_PATH)
                st.rerun()

    if model is not None:
        st.success(f"✅ Model loaded and ready.  Threshold: `{threshold}`")

st.divider()

# ── Step 2: Upload CSV ─────────────────────────────────────────────
st.subheader("📂 Upload Water Sample CSV")
st.markdown("""
Your CSV must contain these **14 columns**:

`ph · Hardness · Solids · Chloramines · Sulfate · Conductivity · Organic_carbon ·
Trihalomethanes · Turbidity · ph_deviation · mineral_load · chloramines_conductivity ·
thm_per_carbon · turbidity_solids_ratio`

The `Potability` column is **not needed** — the app predicts it for you.
""")

uploaded = st.file_uploader("Choose a CSV file", type="csv")

if uploaded and model:
    try:
        df = pd.read_csv(uploaded)

        if TARGET in df.columns:
            df = df.drop(columns=[TARGET])

        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            st.error(f"❌ Missing columns: {', '.join(missing)}")
            st.stop()

        X     = df[FEATURE_COLS]
        probs = model.predict_proba(X)[:, 1]
        preds = (probs >= threshold).astype(int)

        n_safe   = int(preds.sum())
        n_unsafe = len(preds) - n_safe
        n_total  = len(preds)

        st.divider()

        # Single sample verdict
        if n_total == 1:
            if preds[0] == 1:
                st.markdown(f"""
                <div class="safe-box">
                  <h2>✅ SAFE TO DRINK</h2>
                  <p>Potability confidence: {probs[0]*100:.1f}%</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="unsafe-box">
                  <h2>❌ NOT SAFE TO DRINK</h2>
                  <p>Non-potable confidence: {(1-probs[0])*100:.1f}%</p>
                </div>""", unsafe_allow_html=True)

        # Multi-sample summary
        else:
            st.subheader("📊 Summary")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Samples", n_total)
            c2.metric("✅ Safe",   n_safe,   delta=f"{n_safe/n_total*100:.1f}%")
            c3.metric("❌ Unsafe", n_unsafe,
                      delta=f"-{n_unsafe/n_total*100:.1f}%",
                      delta_color="inverse")

            safe_pct = n_safe / n_total * 100
            if safe_pct >= 70:
                st.markdown(
                    f'<div class="safe-box"><h2>✅ Mostly Safe</h2>'
                    f'<p>{safe_pct:.1f}% of samples are potable</p></div>',
                    unsafe_allow_html=True,
                )
            elif safe_pct >= 40:
                st.warning(f"⚠️ Mixed quality — {safe_pct:.1f}% potable. Check individual results below.")
            else:
                st.markdown(
                    f'<div class="unsafe-box"><h2>❌ Mostly Unsafe</h2>'
                    f'<p>Only {safe_pct:.1f}% of samples are potable</p></div>',
                    unsafe_allow_html=True,
                )

        # Results table
        st.subheader("📋 Detailed Results")
        result_df = df[FEATURE_COLS].copy()
        result_df.insert(0, "Prediction",
                         ["✅ Safe" if p == 1 else "❌ Not Safe" for p in preds])
        result_df.insert(1, "Confidence %", (probs * 100).round(2))
        st.dataframe(result_df, use_container_width=True)

        csv_out = result_df.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download Results CSV", csv_out,
            "water_quality_predictions.csv", "text/csv",
        )

    except Exception as e:
        st.error(f"Error: {e}")

elif uploaded and model is None:
    st.warning("⬆️ Please train the models first (expand the section above).")

# ── Step 3: Manual input ───────────────────────────────────────────
if model:
    st.divider()
    st.subheader("🧪 Test a Single Sample Manually")

    with st.form("manual_form"):
        cols = st.columns(3)
        inputs = {}
        for i, feat in enumerate(FEATURE_COLS):
            inputs[feat] = cols[i % 3].number_input(feat, value=0.0, format="%.4f")
        submitted = st.form_submit_button("🔍 Predict This Sample", use_container_width=True)

    if submitted:
        row  = pd.DataFrame([inputs])
        prob = model.predict_proba(row)[0, 1]
        pred = int(prob >= threshold)
        if pred == 1:
            st.markdown(
                f'<div class="safe-box"><h2>✅ SAFE TO DRINK</h2>'
                f'<p>Potability probability: {prob*100:.1f}%</p></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="unsafe-box"><h2>❌ NOT SAFE TO DRINK</h2>'
                f'<p>Non-potable probability: {(1-prob)*100:.1f}%</p></div>',
                unsafe_allow_html=True,
            )

st.divider()
st.caption("💧 Water Quality Analyzer · XGBoost + LightGBM + Random Forest + Stacking Ensemble")
