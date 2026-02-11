import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# =========================
# Page Config
# =========================
st.set_page_config(page_title="Forest Fire Prediction", page_icon="🔥")

st.title("🔥 Forest Fire Occurrence Prediction")
st.write("Adjust inputs to predict whether a forest fire is likely.")

# =========================
# Load model artifacts safely
# =========================
BASE_DIR = Path(__file__).parent

model_path = BASE_DIR / "fire_model.pkl"
scaler_path = BASE_DIR / "scaler.pkl"
feature_order_path = BASE_DIR / "feature_order.pkl"

try:
    loaded_model = joblib.load(model_path)
except Exception as e:
    st.error(f"❌ Could not load model: {e}")
    st.stop()

# Scaler is optional
loaded_scaler = None
if scaler_path.exists():
    try:
        loaded_scaler = joblib.load(scaler_path)
    except:
        st.warning("⚠️ Scaler found but could not load. Continuing without scaling.")

try:
    loaded_feature_order = joblib.load(feature_order_path)
except Exception as e:
    st.error(f"❌ Could not load feature order: {e}")
    st.stop()

if not isinstance(loaded_feature_order, list):
    st.error("❌ feature_order.pkl must contain a list.")
    st.stop()

# =========================
# Categorical mappings
# =========================
month_map = {
    'jan': 'month_jan', 'feb': 'month_feb', 'mar': 'month_mar',
    'apr': 'month_apr', 'may': 'month_may', 'jun': 'month_jun',
    'jul': 'month_jul', 'aug': 'month_aug', 'sep': 'month_sep',
    'oct': 'month_oct', 'nov': 'month_nov', 'dec': 'month_dec'
}

day_map = {
    'mon': 'day_mon', 'tue': 'day_tue', 'wed': 'day_wed',
    'thu': 'day_thu', 'fri': 'day_fri',
    'sat': 'day_sat', 'sun': 'day_sun'
}

# =========================
# User Inputs
# =========================
selected_month = st.selectbox("Month", list(month_map.keys()))
selected_day = st.selectbox("Day", list(day_map.keys()))

user_input = {}

# Initialize categorical columns
for col in loaded_feature_order:
    if col.startswith("month_") or col.startswith("day_"):
        user_input[col] = 0

# Activate selected
if
