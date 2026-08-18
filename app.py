import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="SmartCare AI", page_icon="🏥", layout="centered")

@st.cache_resource
def load_models():
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return preprocessor, model

preprocessor, model = load_models()

st.title("🏥 SmartCare AI - Appointment No-Show Predictor")
st.write("Enter patient details below to predict no-show probability.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=0, max_value=100, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        department = st.selectbox("Department", [
            "General Medicine", "Cardiology", "Pediatrics", 
            "Radiology", "Neurology", "Orthopedics", "Laboratory Services"
        ])
        waiting_days = st.number_input("Waiting Days", min_value=0, value=2)
        
    with col2:
        previous_appointments = st.number_input("Previous Appointments", min_value=0, value=1)
        missed_previous = st.number_input("Missed Previous Appointments", min_value=0, value=0)
        total_bill = st.number_input("Total Bill (LKR)", min_value=0, value=2000)
        payment_status = st.selectbox("Payment Status", ["Paid", "Unpaid", "Partially Paid"])

    submit = st.form_submit_button("Predict No-Show Risk")

if submit:
    input_data = {
        "age": age,
        "gender": gender,
        "department": department,
        "waiting_days": waiting_days,
        "previous_appointments": previous_appointments,
        "missed_previous_appointments": missed_previous,
        "total_bill_lkr": total_bill,
        "payment_status": payment_status
    }
    
    df_input = pd.DataFrame([input_data])
    processed = preprocessor.transform(df_input)
    
    prob_no_show = model.predict_proba(processed)[0][1]
    
    missed_ratio = missed_previous / max(1, previous_appointments)
    
    st.markdown("---")
    
    if prob_no_show > 0.35 or missed_ratio >= 0.3 or waiting_days > 7:
        st.error("⚠️ **High Risk:** The patient is likely to MISS the appointment.")
    else:
        st.success("✅ **Low Risk:** The patient is likely to ATTEND the appointment.")