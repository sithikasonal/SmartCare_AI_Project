import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SmartCare AI", page_icon="🏥", layout="wide")


@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)


try:
    model_pipeline = load_model()
except Exception as e:
    st.error(
        f"Model එක Load කිරීමට අපහසුයි. 'model.pkl' file එක තිබේදැයි බලන්න. Error: {e}"
    )

st.title("🏥 Appointment No-Show Predictor")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (වයස)", min_value=0, max_value=120, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        lead_time = st.number_input(
            "Lead Time (අත්තිකාරම් දින ගණන)", min_value=0, value=9
        )
        previous_no_shows = st.number_input(
            "Previous No-Shows (කලින් නොපැමිණි වාර)", min_value=0, value=14
        )

    with col2:
        distance = st.number_input(
            "Distance to Clinic (km)", min_value=0.0, value=100.0, step=1.0
        )
        reminder_sent = st.selectbox(
            "Reminder Sent (පණිවිඩයක් යැවුවේද?)", ["No", "Yes"]
        )
        specialty = st.selectbox(
            "Specialty (වෛද්‍ය අංශය)",
            [
                "General Practice",
                "Cardiology",
                "Pediatrics",
                "Radiology",
                "Neurology",
                "Orthopedics",
            ],
        )

    submit = st.form_submit_button("🔮 Predict Risk")

if submit:
    # Colab එකේ Train කළ Column Names වලට අනුගතව mapping එක සෑදීම
    reminder_val = 1 if reminder_sent == "Yes" else 0

    input_dict = {
        "age": age,
        "waiting_days": lead_time,  # Lead time -> waiting_days ලෙස
        "previous_appointments": previous_no_shows
        + 2,  # total estimation
        "missed_previous_appointments": previous_no_shows,
        "total_bill_lkr": 2000,  # default
        "gender": gender,
        "department": (
            "General Medicine"
            if specialty == "General Practice"
            else specialty
        ),
        "payment_status": "Paid",
    }

    df_input = pd.DataFrame([input_dict])

    try:
        proba = model_pipeline.predict_proba(df_input)[0]
        attending_prob = proba[0]
        missing_prob = proba[1]

        st.markdown("---")
        st.subheader("📊 Prediction Result:")

        # Previous No-Shows 14ක් වැනි වැඩි අගයක් තිබේ නම් High Risk ලෙස පෙන්වීමට
        if missing_prob >= 0.50 or previous_no_shows >= 3 or lead_time > 14:
            st.error(
                f"❌ **High Risk (Likely to Miss)** (පැමිණීමේ සම්භාවිතාව: {attending_prob:.1%})"
            )
        else:
            st.success(
                f"✅ **Low Risk (Likely to Show Up)** (පැමිණීමේ සම්භාවිතාව: {attending_prob:.1%})"
            )

    except Exception as err:
        st.error(f"Prediction : {err}")