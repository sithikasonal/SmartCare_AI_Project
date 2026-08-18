import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Page configuration
st.set_page_config(page_title="SmartCare AI", page_icon="🏥", layout="wide")

# Custom CSS for Green Modern Medical Theme
st.markdown(
    """
    <style>
    /* Main Background */
    .stApp {
        background-color: #f4f8f6;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styling */
    .main-title {
        color: #1a4332;
        font-weight: 700;
        font-size: 2.2rem;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #5d7a6e;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    /* Container/Card Styling */
    div[data-testid="stForm"] {
        background-color: #ffffff;
        border-radius: 20px;
        padding: 35px;
        border: 1px solid #e1ebe5;
        box-shadow: 0 10px 25px rgba(26, 67, 50, 0.05);
    }
    
    /* Input Labels */
    label {
        color: #2c4a3e !important;
        font-weight: 600 !important;
    }

    /* Input Controls */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        border-radius: 12px !important;
        border: 1px solid #d0dfd7 !important;
        background-color: #fcfdfe !important;
    }

    /* Submit Button */
    div[data-testid="stForm"] button {
        background: linear-gradient(135deg, #2ecc71 0%, #1abc9c 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 12px 28px;
        font-weight: 600;
        font-size: 1.05rem;
        width: 100%;
        box-shadow: 0 4px 15px rgba(46, 204, 113, 0.3);
        transition: all 0.3s ease;
    }
    div[data-testid="stForm"] button:hover {
        background: linear-gradient(135deg, #27ae60 0%, #16a085 100%);
        transform: translateY(-2px);
    }

    /* Alert Boxes */
    .stAlert {
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Gauge Chart Function (Mint-Green Palette)
def create_gauge_chart(probability_val):
  percentage = probability_val * 100
  fig = go.Figure(
      go.Indicator(
          mode="gauge+number",
          value=percentage,
          number={"suffix": "%", "font": {"size": 42, "color": "#1a4332"}},
          gauge={
              "axis": {
                  "range": [0, 100],
                  "tickwidth": 1,
                  "tickcolor": "#5d7a6e",
              },
              "bar": {"color": "#e74c3c" if percentage >= 50 else "#2ecc71"},
              "steps": [
                  {"range": [0, 40], "color": "#e8f8f0"},
                  {"range": [40, 70], "color": "#fef9e7"},
                  {"range": [70, 100], "color": "#fadbd8"},
              ],
          },
      )
  )
  fig.update_layout(
      height=260,
      margin=dict(l=20, r=20, t=30, b=20),
      paper_bgcolor="rgba(0,0,0,0)",
  )
  return fig


# Model Loading
@st.cache_resource
def load_model():
  try:
    return joblib.load("model.pkl")
  except Exception:
    return joblib.load("model.joblib")


try:
  model_pipeline = load_model()
except Exception as e:
  st.error(f"Model එක Load කිරීමට අපහසුයි. Error: {e}")

# Header UI
st.markdown(
    '<div class="main-title">🏥 Appointment No-Show Predictor</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">SmartCare AI Clinical Attendance Risk Assessment'
    " Engine</div>",
    unsafe_allow_html=True,
)

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
  reminder_val = 1 if reminder_sent == "Yes" else 0

  input_dict = {
      "age": age,
      "waiting_days": lead_time,
      "previous_appointments": previous_no_shows + 2,
      "missed_previous_appointments": previous_no_shows,
      "total_bill_lkr": 2000,
      "gender": gender,
      "department": (
          "General Medicine" if specialty == "General Practice" else specialty
      ),
      "payment_status": "Paid",
  }

  df_input = pd.DataFrame([input_dict])

  try:
    proba = model_pipeline.predict_proba(df_input)[0]
    attending_prob = proba[0]
    missing_prob = proba[1]

    st.markdown("---")

    # High / Low Risk Alert
    if missing_prob >= 0.50 or previous_no_shows >= 3 or lead_time > 14:
      st.error(
          f"⚠️ **HIGH RISK** — Predicted probability of 30-day no-show:"
          f" **{missing_prob:.1%}**"
      )
      is_high_risk = True
    else:
      st.success(
          f"✅ **LOW RISK** — Predicted probability of attendance:"
          f" **{attending_prob:.1%}**"
      )
      is_high_risk = False

    # Gauge Chart Display
    st.plotly_chart(create_gauge_chart(missing_prob), use_container_width=True)

    # Recommended Action Card
    st.subheader("📋 Recommended Action Plan")
    if is_high_risk:
      st.info("""
            * **Standard discharge planning** with a follow-up appointment reminder.
            * **Monitor closely** at the next scheduled visit.
            * **Automated Reminders:** Send SMS and WhatsApp reminders 24 hours prior.
            * **Direct Outreach:** Assign a clinic coordinator to make a direct confirmation call.
            """)
    else:
      st.info("""
            * **Standard Follow-Up:** Proceed with routine clinic appointment workflows.
            * **Automated Notification:** Send a single SMS reminder 1 day prior.
            """)

  except Exception as err:
    st.error(f"Prediction Error: {err}")