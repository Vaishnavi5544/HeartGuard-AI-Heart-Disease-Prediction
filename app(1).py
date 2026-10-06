import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from tensorflow.keras.models import load_model

# ============================================================
# HEARTGUARD AI - HEART DISEASE RISK PREDICTION
# ============================================================

st.set_page_config(
    page_title="HeartGuard AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------- CSS -----------------------------

st.markdown("""
<style>
.main { background: #f7f9fc; }
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}
.hero {
    padding: 30px 34px;
    border-radius: 22px;
    background: linear-gradient(135deg, #111827, #1e3a5f, #2563eb);
    color: white;
    margin-bottom: 24px;
    box-shadow: 0 12px 35px rgba(0,0,0,.12);
}
.hero h1 { margin: 0; font-size: 2.35rem; font-weight: 800; }
.hero p { margin: 8px 0 0 0; font-size: 1.05rem; opacity: .92; }
.result-card {
    background: white;
    border-radius: 20px;
    padding: 28px;
    text-align: center;
    border: 1px solid #e5e7eb;
    box-shadow: 0 10px 30px rgba(0,0,0,.08);
    margin-top: 24px;
}
.section-title {
    font-size: 1.25rem;
    font-weight: 800;
    margin: 18px 0 14px 0;
}
div.stButton > button {
    width: 100%;
    border-radius: 12px;
    min-height: 3rem;
    font-weight: 800;
    font-size: 1rem;
}
[data-testid="stSidebar"] { background: #ffffff; }
</style>
""", unsafe_allow_html=True)


# ---------------------- File paths ----------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "Heart_diseases.keras"
SCALER_PATH = BASE_DIR / "scalerHeart.pkl"
CSV_PATH = BASE_DIR / "heart_disease_risk_2026.csv"


# ---------------------- Load resources ------------------------

@st.cache_resource
def load_resources():
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    reference_df = pd.read_csv(CSV_PATH)
    return model, scaler, reference_df


try:
    model, scaler, reference_df = load_resources()
except Exception as e:
    st.error("❌ Required project files could not be loaded.")
    st.write("Keep these 3 files in the same folder as app.py:")
    st.code(
        "Heart_diseases.keras\n"
        "scalerHeart.pkl\n"
        "heart_disease_risk_2026.csv"
    )
    st.exception(e)
    st.stop()


# ---------------------- Training structure --------------------

TARGET = "has_heart_disease"
ID_COLUMN = "patient_id"

if TARGET not in reference_df.columns:
    st.error(f"Target column '{TARGET}' was not found in the CSV.")
    st.stop()

raw_feature_columns = [
    c for c in reference_df.columns
    if c not in [TARGET, ID_COLUMN]
]

reference_x = reference_df[raw_feature_columns].copy()

# SAME one-hot encoding approach used in training
reference_encoded = pd.get_dummies(
    reference_x,
    drop_first=True
)

# The saved scaler is the final authority for feature names/order.
# This fixes the sex -> sex_Male mismatch.
if hasattr(scaler, "feature_names_in_"):
    feature_columns = list(scaler.feature_names_in_)
else:
    feature_columns = list(reference_encoded.columns)


# ---------------------- Imputation ----------------------------

reference_encoded = reference_encoded.copy()

for col in reference_encoded.columns:
    if pd.api.types.is_numeric_dtype(reference_encoded[col]):
        mean_value = reference_encoded[col].mean()
        if pd.isna(mean_value):
            mean_value = 0
        reference_encoded[col] = reference_encoded[col].fillna(mean_value)

reference_encoded = reference_encoded.reindex(
    columns=feature_columns,
    fill_value=0
)

training_means = reference_encoded.mean(numeric_only=True).to_dict()


# ---------------------- UI Header -----------------------------

st.markdown("""
<div class="hero">
    <h1>❤️ HeartGuard AI</h1>
    <p>AI-powered heart disease risk prediction using a trained Neural Network.</p>
</div>
""", unsafe_allow_html=True)


# ---------------------- Sidebar -------------------------------

with st.sidebar:
    st.markdown("## ❤️ HeartGuard AI")
    st.divider()

    st.markdown("### 🧠 About the Project")
    st.write(
        "HeartGuard AI is a machine-learning based application "
        "that estimates the probability of heart disease from "
        "patient health information."
    )

    st.markdown("### ⚙️ Model")
    st.info(
        "Neural Network\n\n"
        f"{len(raw_feature_columns)} raw input features\n\n"
        "One-hot encoding + StandardScaler"
    )

    st.markdown("### 🔧 Prediction Pipeline")
    st.write("1. Enter patient information")
    st.write("2. Encode categorical values")
    st.write("3. Match training feature names")
    st.write("4. Apply saved StandardScaler")
    st.write("5. Neural Network prediction")

    st.divider()
    st.caption(
        "Educational/project demonstration only. "
        "This application is not a medical diagnostic tool."
    )


# ---------------------- Helper functions ----------------------

def pretty_name(name):
    return str(name).replace("_", " ").replace("-", " ").title()


def is_boolean_series(series):
    return pd.api.types.is_bool_dtype(series)


def get_numeric_default(series):
    numeric = pd.to_numeric(series, errors="coerce")
    median = numeric.median()
    return 0.0 if pd.isna(median) else float(median)


def get_numeric_bounds(series):
    numeric = pd.to_numeric(series, errors="coerce").dropna()

    if numeric.empty:
        return 0.0, 100.0

    min_value = float(numeric.min())
    max_value = float(numeric.max())

    if min_value == max_value:
        min_value -= 1
        max_value += 1

    return min_value, max_value


# ---------------------- Intro metrics -------------------------

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("🤖 Model", "Neural Network")

with m2:
    st.metric("📊 Training Rows", f"{len(reference_df):,}")

with m3:
    st.metric("🧩 Model Features", str(len(feature_columns)))

with m4:
    st.metric("🎯 Output", "Risk Probability")


st.markdown(
    '<div class="section-title">🩺 Enter Patient Information</div>',
    unsafe_allow_html=True
)

st.caption(
    "Enter the patient's health information below. "
    "Categorical fields are automatically encoded in the same way as training."
)


# ---------------------- Input UI ------------------------------

input_values = {}

groups = {
    "🫀 Basic & Blood Pressure": [
        "age",
        "sex",
        "resting_bp_systolic",
        "resting_bp_diastolic",
        "cholesterol_total",
        "hdl",
        "ldl",
        "triglycerides",
    ],
    "🩸 Blood Sugar & Body": [
        "fasting_blood_sugar",
        "hba1c",
        "bmi",
    ],
    "🏃 Lifestyle & Activity": [
        "daily_steps",
        "stress_score",
        "diet_quality_score",
        "wearable_owner",
    ],
    "❤️ Medical & Lifestyle History": [
        "family_history",
        "smoker_status",
        "alcohol_use",
        "chest_pain_type",
    ],
}

already_grouped = set()
for values in groups.values():
    already_grouped.update(values)

remaining = [
    c for c in raw_feature_columns
    if c not in already_grouped
]

if remaining:
    groups["📋 Additional Health Information"] = remaining


for group_name, columns in groups.items():

    valid_columns = [
        col for col in columns
        if col in raw_feature_columns
    ]

    if not valid_columns:
        continue

    st.markdown(
        f'<div class="section-title">{group_name}</div>',
        unsafe_allow_html=True
    )

    cols = st.columns(2)

    for index, col in enumerate(valid_columns):

        series = reference_df[col]
        non_null = series.dropna()

        with cols[index % 2]:

            # Boolean column
            if is_boolean_series(series):
                default_bool = False

                if not non_null.empty:
                    mode = non_null.mode()
                    if not mode.empty:
                        default_bool = bool(mode.iloc[0])

                input_values[col] = st.checkbox(
                    pretty_name(col),
                    value=default_bool,
                    key=f"input_{col}"
                )

            # Categorical/object column
            elif (
                pd.api.types.is_object_dtype(series)
                or isinstance(series.dtype, pd.CategoricalDtype)
            ):
                choices = [
                    str(x) for x in non_null.unique().tolist()
                    if pd.notna(x)
                ]

                if choices:
                    input_values[col] = st.selectbox(
                        pretty_name(col),
                        options=choices,
                        key=f"input_{col}"
                    )
                else:
                    input_values[col] = st.text_input(
                        pretty_name(col),
                        value="",
                        key=f"input_{col}"
                    )

            # Numeric column
            else:
                default = get_numeric_default(series)
                min_value, max_value = get_numeric_bounds(series)

                input_values[col] = st.number_input(
                    pretty_name(col),
                    min_value=min_value,
                    max_value=max_value,
                    value=default,
                    step=0.1,
                    key=f"input_{col}"
                )


# ---------------------- Prediction ----------------------------

st.divider()

predict_left, predict_mid, predict_right = st.columns([1, 2, 1])

with predict_mid:
    predict_button = st.button(
        "🔍 PREDICT HEART DISEASE RISK",
        type="primary"
    )


if predict_button:

    try:
        # 1. Raw patient data
        input_raw_df = pd.DataFrame([input_values])

        # 2. SAME one-hot encoding as training
        input_encoded = pd.get_dummies(
            input_raw_df,
            drop_first=True
        )

        # 3. EXACT feature names + EXACT order used by scaler.
        # Example:
        # training: sex_Male
        # UI input: sex
        # reindex creates sex_Male correctly.
        input_encoded = input_encoded.reindex(
            columns=feature_columns,
            fill_value=0
        )

        # 4. Numeric conversion
        input_encoded = input_encoded.apply(
            pd.to_numeric,
            errors="coerce"
        )

        # 5. Fill missing values
        for col in input_encoded.columns:
            if input_encoded[col].isna().any():
                input_encoded[col] = input_encoded[col].fillna(
                    training_means.get(col, 0.0)
                )

        input_encoded = input_encoded.astype(float)

        # 6. Final feature-order safety check
        if list(input_encoded.columns) != feature_columns:
            st.error("❌ Feature order mismatch.")
            st.write("Expected features:")
            st.write(feature_columns)
            st.stop()

        # 7. Apply saved StandardScaler
        input_scaled = scaler.transform(input_encoded)

        # 8. Neural Network prediction
        probability = float(
            model.predict(input_scaled, verbose=0).ravel()[0]
        )

        probability = max(0.0, min(1.0, probability))
        risk_percentage = probability * 100

        # 0 = No heart disease, 1 = Heart disease
        prediction = 1 if probability >= 0.5 else 0

        confidence = (
            probability if prediction == 1
            else 1 - probability
        )

        # 9. Result
        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        if prediction == 1:
            st.error(
                f"❤️‍🩹 Higher Predicted Risk — {risk_percentage:.1f}%"
            )
            st.markdown(
                f"<h2>Heart Disease Risk Detected</h2>"
                f"<p>Model probability: <b>{risk_percentage:.1f}%</b></p>"
                f"<p>Prediction confidence: "
                f"<b>{confidence * 100:.1f}%</b></p>",
                unsafe_allow_html=True
            )
        else:
            st.success(
                f"💚 Lower Predicted Risk — {risk_percentage:.1f}%"
            )
            st.markdown(
                f"<h2>Lower Heart Disease Risk</h2>"
                f"<p>Model probability: <b>{risk_percentage:.1f}%</b></p>"
                f"<p>Prediction confidence: "
                f"<b>{confidence * 100:.1f}%</b></p>",
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # 10. Probability
        st.write("")
        st.markdown("### 📈 Risk Probability")
        st.progress(int(round(risk_percentage)))

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Risk Probability",
                f"{risk_percentage:.1f}%"
            )

        with c2:
            st.metric(
                "Prediction",
                "Disease Risk" if prediction == 1 else "Lower Risk"
            )

        with c3:
            st.metric(
                "Input Features",
                len(feature_columns)
            )

        # 11. Technical details
        with st.expander("🔧 Technical Details"):
            st.write("Raw input columns:")
            st.write(list(input_raw_df.columns))

            st.write("Encoded features sent to scaler/model:")
            st.write(list(input_encoded.columns))

            st.write("Input shape:")
            st.write(input_scaled.shape)

            st.success(
                "Feature names and order successfully matched "
                "with the saved StandardScaler."
            )

        st.info(
            "⚠️ This prediction is for educational/project demonstration "
            "purposes only and should not be used as a medical diagnosis."
        )

    except Exception as e:
        st.error("❌ Prediction failed.")
        st.write(
            "Please make sure the saved model, scaler and CSV were created "
            "using the same preprocessing pipeline."
        )
        st.exception(e)


# ---------------------- Footer --------------------------------

st.divider()

st.markdown(
    "<p style='text-align:center; color:#64748b;'>"
    "❤️ HeartGuard AI • Neural Network Heart Disease Risk Prediction"
    "</p>",
    unsafe_allow_html=True
)
