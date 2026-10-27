import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Medical Insurance Predictor",
    page_icon="✚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
        .stApp { background: #f5f7fb; }
        .block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem; }
        .hero {
            background: linear-gradient(120deg, #123b5d 0%, #176b75 100%);
            padding: 2rem 2.2rem; border-radius: 22px; color: white; margin-bottom: 1.5rem;
        }
        .hero-kicker { text-transform: uppercase; letter-spacing: .13em; font-size: .75rem;
            font-weight: 700; color: #b9e6e5; margin-bottom: .6rem; }
        .hero h1 { color: white; font-size: clamp(2rem, 4vw, 2.8rem); line-height: 1.12; margin: 0 0 .7rem; }
        .hero p { color: #e1eef5; font-size: 1rem; max-width: 720px; margin: 0; }
        .section-card { background: white; border: 1px solid #e4eaf1; border-radius: 18px;
            padding: 1.25rem 1.35rem .7rem; margin-bottom: 1rem; }
        .section-heading { color: #17354d; font-size: 1.12rem; font-weight: 700; margin-bottom: .2rem; }
        .section-subtitle { color: #64748b; font-size: .88rem; margin-bottom: 1rem; }
        .result-card { background: white; border: 1px solid #d9e8e8; border-radius: 18px;
            padding: 1.5rem; margin-top: 1rem; }
        .result-label { color: #526578; font-size: .9rem; font-weight: 600; }
        .result-value { color: #126b66; font-size: clamp(2rem, 4vw, 2.8rem);
            font-weight: 700; line-height: 1.2; margin: .35rem 0 .55rem; }
        .muted-note { color: #64748b; font-size: .82rem; }
        div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
            background: #126b66; color: white; border: 0; border-radius: 10px;
            padding: .7rem 1.2rem; font-weight: 700; min-height: 3rem; width: 100%;
        }
        div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
            background: #0e5753; color: white; border: 0;
        }
        [data-testid="stSidebar"] { background: #edf3f7; }
        [data-testid="stSidebar"] h2 { color: #17354d; }
        div[data-testid="stMetric"] { background: #f7fafc; border: 1px solid #e4eaf1;
            padding: .8rem 1rem; border-radius: 12px; }
        .footer { color: #718096; font-size: .78rem; text-align: center; padding-top: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

APP_DIR = Path(__file__).parent
REQUIRED_FILES = ["medical_insurance.pkl", "x_scaler.pkl", "y_scaler.pkl"]


def artifact_versions():
    # Modification times act as the cache key, so re-saving a .pkl from the
    # notebook reloads it instead of serving the copy cached at startup
    missing = [name for name in REQUIRED_FILES if not (APP_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model file(s): " + ", ".join(missing)
            + ". Place them in the same folder as app.py."
        )
    return tuple((APP_DIR / name).stat().st_mtime for name in REQUIRED_FILES)


@st.cache_resource
def load_artifacts(versions):
    model, x_scaler, y_scaler = (joblib.load(APP_DIR / name) for name in REQUIRED_FILES)

    # The y scaler must be fitted on charges in dollars; one fitted on 0-1 data
    # would leave predictions in scaled units (e.g. $0.21)
    y_center = getattr(y_scaler, "mean_", getattr(y_scaler, "data_max_", [0]))[0]
    if y_center < 100:
        raise ValueError(
            "y_scaler.pkl was not fitted on charges in dollars. Re-save it from the "
            "notebook with joblib.dump(y_standard_scaler, 'y_scaler.pkl')."
        )
    return model, x_scaler, y_scaler

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Health cost estimation</div>
        <h1>Medical Insurance Predictor</h1>
        <p>Enter a few details to estimate medical insurance charges using a
        machine-learning model. Review your inputs before generating an estimate.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## About this tool")
    st.write("This demo estimates medical insurance charges from the personal and lifestyle details entered.")
    st.divider()
    st.markdown("### Input guide")
    st.caption("BMI means body mass index.")
    st.caption("Region encoding must match the mapping used when training the model.")
    st.divider()
    st.caption("For educational demonstration only. This is not an insurance quote, medical advice, or a guarantee of actual costs.")

try:
    model, x_scaler, y_scaler = load_artifacts(artifact_versions())
    artifacts_loaded = True
except Exception as error:
    artifacts_loaded = False
    st.error(f"Model files could not be loaded: {error}")
    st.caption("Confirm that model, scalers, feature order, and categorical encodings match the training notebook.")

# Encodings must match the label encoding in the training notebook
SEX_MAP = {"Male": 0, "Female": 1}
SMOKER_MAP = {"No": 0, "Yes": 1}
REGION_MAP = {"Southeast": 0, "Southwest": 1, "Northeast": 2, "Northwest": 3}
FEATURE_ORDER = ["age", "sex", "bmi", "smoker", "region"]

if artifacts_loaded:
    

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            # Training data covers ages 18-64; the model cannot extrapolate beyond that
            age = st.slider("Age", min_value=18, max_value=64, value=40)
            sex = st.radio("Sex", list(SEX_MAP), horizontal=True)
            bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=30.0, step=0.1)
        with col2:
            smoker = st.radio("Smoker", list(SMOKER_MAP), horizontal=True)
            region = st.selectbox("Region", list(REGION_MAP))
            # Reserved spot under Region so the estimate shows without scrolling
            result_slot = st.empty()
        submitted = st.form_submit_button("Estimate charges")

    if submitted:
        try:
            input_df = pd.DataFrame(
                [[age, SEX_MAP[sex], bmi, SMOKER_MAP[smoker], REGION_MAP[region]]],
                columns=FEATURE_ORDER,
            )
            scaled_input = x_scaler.transform(input_df)
            prediction_scaled = model.predict(scaled_input)
            prediction = y_scaler.inverse_transform(
                np.asarray(prediction_scaled).reshape(-1, 1)
            )[0, 0]

            result_slot.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Estimated annual medical insurance charges</div>
                    <div class="result-value">${max(prediction, 0):,.2f}</div>
                    <div class="muted-note">Model estimate, not an insurance quote.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception as error:
            result_slot.error(f"Prediction could not be completed: {error}")

st.markdown(
    '<div class="footer">Medical Insurance Predictor · Machine-learning project · Estimates are informational only</div>',
    unsafe_allow_html=True,
)
