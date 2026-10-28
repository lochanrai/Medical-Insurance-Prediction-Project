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
        .result-inline { color: #526578; font-size: 1rem; font-weight: 600; margin: 0; text-align: center; }
        .result-inline + .muted-note { text-align: center; }
        /* Divider between the left (Age, Sex, BMI) and right (Smoker, Region) sections */
        /* Drawn in the middle of the 4rem column gap (gap="large"), which is the form's centre */
        .st-key-input_sections [data-testid="stColumn"]:nth-child(2) { position: relative; overflow: visible; }
        .st-key-input_sections [data-testid="stColumn"]:nth-child(2)::before {
            content: ""; position: absolute; left: -2rem; top: 0; bottom: 0; border-left: 1px solid #d9e2ec; }
        /* Spread the Region options so the last one lines up with the bottom of the BMI box */
        .st-key-region [role="radiogroup"] { gap: 0.7rem; }
        .result-value { color: #126b66; font-size: 1.6rem; font-weight: 700; margin-left: .4rem; }
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
        /* Slimmer top bar (Deploy / Share / star / GitHub / menu); default height is 3.75rem */
        header[data-testid="stHeader"] { height: 2.00rem; min-height: 2.00rem; }
        header[data-testid="stHeader"] [data-testid="stToolbar"] { top: 0; height: 2.00rem; align-items: center; }
        header[data-testid="stHeader"] button { padding-top: 0; padding-bottom: 0; }
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
        # Keyed container so the CSS can draw a divider between the two input sections
        with st.container(key="input_sections"):
            col1, col2 = st.columns(2, gap="large")
        with col1:
            # Training data covers ages 18-64; the model cannot extrapolate beyond that
            age = st.slider("Age", min_value=18, max_value=64, value=40)
            sex = st.radio("Sex", list(SEX_MAP), horizontal=True)
            bmi = st.number_input("BMI (Enter values between 10 - 60)", min_value=10.0, max_value=60.0, value=30.0, step=0.1)
        with col2:
            smoker = st.radio("Smoker", list(SMOKER_MAP), horizontal=True)
            # Single choice (the model takes exactly one region), stacked to fill the space beside BMI
            region = st.radio("Region", list(REGION_MAP), key="region")
        # Middle third of the form, so the button and result sit centered
        button_col, result_col = st.columns(2, gap="large", vertical_alignment="center")
        with button_col:
            submitted = st.form_submit_button("Estimate charges", width="stretch")
        with result_col:
            result_slot = st.empty()

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
                <div class="result-inline">Estimated annual charges:
                    <span class="result-value">${max(prediction, 0):,.2f}</span></div>
                
                """,
                unsafe_allow_html=True,
            )
        except Exception as error:
            result_slot.error(f"Prediction could not be completed: {error}")

st.markdown(
    '<div class="footer">Medical Insurance Predictor · Machine-learning project · Estimates are informational only</div>',
    unsafe_allow_html=True,
)
