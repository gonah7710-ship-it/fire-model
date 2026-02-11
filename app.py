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
if month_map[selected_month] in loaded_feature_order:
    user_input[month_map[selected_month]] = 1

if day_map[selected_day] in loaded_feature_order:
    user_input[day_map[selected_day]] = 1

# Numeric inputs
user_input['X'] = st.slider('X (1–9)', 1, 9, 5)
user_input['Y'] = st.slider('Y (2–9)', 2, 9, 5)
user_input['FFMC'] = st.slider('FFMC', 18.0, 97.0, 90.0)
user_input['DMC'] = st.slider('DMC', 1.0, 292.0, 110.0)
user_input['DC'] = st.slider('DC', 7.0, 861.0, 500.0)
user_input['ISI'] = st.slider('ISI', 0.0, 57.0, 9.0)
user_input['temp'] = st.slider('Temperature (°C)', 2.0, 34.0, 18.0)
user_input['RH'] = st.slider('Relative Humidity (%)', 15, 100, 45)
user_input['wind'] = st.slider('Wind (km/h)', 0.0, 10.0, 4.0)
user_input['rain'] = st.slider('Rain (mm)', 0.0, 7.0, 0.0)

# =========================
# Prepare input safely
# =========================
input_df = pd.DataFrame([user_input])

final_input_df = input_df.reindex(
    columns=loaded_feature_order,
    fill_value=0
).astype(float)

# Apply scaling only if scaler exists
if loaded_scaler:
    try:
        final_input = loaded_scaler.transform(final_input_df)
    except Exception as e:
        st.error(f"Scaling error: {e}")
        st.stop()
else:
    final_input = final_input_df

# =========================
# Prediction
# =========================
if st.button("Predict Fire Occurrence"):

    try:
        prediction = loaded_model.predict(final_input)

        if hasattr(loaded_model, "predict_proba"):
            prediction_proba = loaded_model.predict_proba(final_input)
            prob = prediction_proba[0][1]
        else:
            prob = None

        if prediction[0] == 1:
            if prob is not None:
                st.error(f"🔥 Fire Likely (Probability: {prob:.2f})")
            else:
                st.error("🔥 Fire Likely")
        else:
            if prob is not None:
                st.success(f"✅ Fire Unlikely (Probability: {1 - prob:.2f})")
            else:
                st.success("✅ Fire Unlikely")

    except Exception as e:
        st.error(f"Prediction error: {e}")
