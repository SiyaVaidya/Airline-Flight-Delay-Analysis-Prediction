import streamlit as st
import pandas as pd
import joblib
from datetime import date
from pathlib import Path

# Keep this file and best_flight_delay_regression_model.pkl in the same folder.
MODEL_PATH = Path(__file__).parent / "best_flight_delay_regression_model.pkl"

st.set_page_config(
    page_title="Airline Flight Delay Prediction",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Professional dark dashboard styling
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0B1220 0%, #171F3B 55%, #242044 100%);
    color: #F8FAFC;
}
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }
h1, h2, h3, p, label, .stMarkdown { color: #F8FAFC; }
.hero {
    padding: 1.5rem 1.7rem;
    border: 1px solid rgba(148,163,184,.25);
    border-radius: 20px;
    background: rgba(30,41,70,.72);
    margin-bottom: 1.4rem;
}
.subtle { color: #B8C4DC !important; }
[data-testid="stMetric"] {
    background: rgba(30,41,70,.88);
    padding: 1.2rem;
    border: 1px solid rgba(148,163,184,.28);
    border-radius: 16px;
}
div.stButton > button {
    width: 100%;
    border: 0;
    border-radius: 12px;
    min-height: 3rem;
    background: linear-gradient(90deg, #7C3AED, #4F46E5);
    color: white;
    font-weight: 700;
}
div.stButton > button:hover { color: white; border: 1px solid #A78BFA; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>✈️ Airline Flight Delay Prediction</h1>
  <p class="subtle">Enter flight details to estimate arrival delay using your trained machine-learning model.</p>
</div>
""", unsafe_allow_html=True)

if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Put 'best_flight_delay_regression_model.pkl' "
        "in the same folder as app.py."
    )
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:
    st.error(
        "The model could not be loaded. This is often caused by a scikit-learn "
        "version mismatch. In the VS Code terminal run: "
        "`python -m pip install scikit-learn==1.6.1` and try again."
    )
    st.code(str(exc))
    st.stop()

st.subheader("Flight details")
st.markdown('<p class="subtle">All fields are required for a prediction.</p>', unsafe_allow_html=True)

left, right = st.columns(2, gap="large")

with left:
    airline = st.selectbox(
        "Airline code",
        ["AA", "AS", "B6", "DL", "F9", "G4", "HA", "MQ",
         "NK", "OH", "OO", "UA", "WN", "YV", "YX", "9E"],
        help="Choose an airline code represented in the training dataset."
    )
    departure_delay = st.number_input(
        "Departure delay (minutes)",
        min_value=-60.0,
        max_value=1500.0,
        value=10.0,
        step=1.0,
        help="Negative values mean the flight departed early."
    )

with right:
    distance = st.number_input(
        "Flight distance (miles)",
        min_value=1.0,
        max_value=10000.0,
        value=500.0,
        step=10.0
    )
    flight_date = st.date_input("Flight date", value=date.today())

st.caption("This model uses departure delay, distance, month, day of week, and airline. It does not retrieve live flight data.")

if st.button("Predict arrival delay"):
    input_data = pd.DataFrame([{
        "DepDelay": departure_delay,
        "Distance": distance,
        "Month": flight_date.month,
        "DayOfWeek": flight_date.weekday(),
        "Reporting_Airline": airline,
    }])

    try:
        prediction = float(model.predict(input_data)[0])
        status = (
            "Likely delayed (15+ minutes)" if prediction >= 15
            else "Slight arrival delay" if prediction > 0
            else "Expected on time or early"
        )

        st.divider()
        st.subheader("Prediction result")
        result_col, status_col = st.columns(2, gap="large")

        with result_col:
            st.metric("Estimated arrival delay", f"{prediction:.1f} minutes")

        with status_col:
            st.markdown("#### Estimated status")
            if prediction >= 15:
                st.warning(status)
            elif prediction > 0:
                st.info(status)
            else:
                st.success(status)

        st.markdown("#### Input summary")
        st.dataframe(input_data, use_container_width=True, hide_index=True)
        st.caption("Prediction is an estimate, not an official or live flight status.")

    except Exception as exc:
        st.error("Prediction failed. Confirm the model was trained with the same five feature names and input format.")
        st.code(str(exc))


# --------------------------------------------------
# MODEL COMPARISON TABLE & EVALUATION METRICS
# --------------------------------------------------

st.divider()

st.markdown("## 📊 Complete Performance Table")

METRICS_PATH = Path(__file__).parent / "regression_model_comparison.csv"

if METRICS_PATH.exists():
    try:
        metrics_df = pd.read_csv(METRICS_PATH)

        required_columns = {"Model", "MAE", "RMSE", "R2 Score"}

        if required_columns.issubset(metrics_df.columns):

            # Ensure evaluation metrics are numeric
            for col in ["MAE", "RMSE", "R2 Score"]:
                metrics_df[col] = pd.to_numeric(
                    metrics_df[col], errors="coerce"
                )

            if "MSE" in metrics_df.columns:
                metrics_df["MSE"] = pd.to_numeric(
                    metrics_df["MSE"], errors="coerce"
                )

            metrics_df = metrics_df.dropna(
                subset=["MAE", "RMSE", "R2 Score"]
            )

            # Display complete model comparison table
            st.dataframe(
                metrics_df.round(4),
                use_container_width=True,
                hide_index=True
            )

            if not metrics_df.empty:

                # Best model based on the lowest RMSE
                best_model = metrics_df.loc[
                    metrics_df["RMSE"].idxmin()
                ]

                st.success(
                    f"🏆 Best model by RMSE in this comparison: "
                    f"{best_model['Model']} "
                    f"(RMSE = {best_model['RMSE']:.4f})."
                )

                # Evaluation metrics at the bottom
                st.markdown("## 📈 Key Performance Indicators")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Models Compared",
                        len(metrics_df)
                    )

                with col2:
                    st.metric(
                        "Best RMSE",
                        f"{best_model['RMSE']:.4f}"
                    )

                with col3:
                    st.metric(
                        "Best MAE",
                        f"{metrics_df['MAE'].min():.4f}"
                    )

                with col4:
                    st.metric(
                        "Best R² Score",
                        f"{metrics_df['R2 Score'].max():.4f}"
                    )

                st.caption(
                    "MAE and RMSE measure prediction error; lower is better. "
                    "R² measures the proportion of variation explained by "
                    "the model; higher is generally better."
                )

        else:
            st.error(
                "The comparison CSV must contain these columns: "
                "Model, MAE, RMSE, R2 Score."
            )
            st.write("Columns found:", list(metrics_df.columns))

    except Exception as exc:
        st.error("Could not read the model comparison file.")
        st.code(str(exc))

else:
    st.warning(
        "Place regression_model_comparison.csv in the same folder as app.py "
        "to display model performance."
    )
