import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Page configuration
st.set_page_config(page_title="SmartCare AI", page_icon="🏥", layout="wide")

# Custom CSS for Dark Glassmorphism Theme
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Dark Theme Background */
    .stApp {
        background-color: #0B132B;
        color: #E0E6ED;
    }
    
    /* Header Styling */
    .main-title {
        color: #FFFFFF;
        font-weight: 700;
        font-size: 2.3rem;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .sub-title {
        color: #64748B;
        font-size: 0.95rem;
        font-weight: 500;
        margin-bottom: 1.8rem;
    }

    /* Glassmorphism Form Container */
    div[data-testid="stForm"] {
        background: rgba(28, 37, 65, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 24px;
        padding: 35px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0px 20px 40px rgba(0, 0, 0, 0.4);
    }

    /* Input Labels */
    label {
        color: #94A3B8 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }

    /* Dark Input Controls & Dropdowns */
    .stNumberInput input, div[data-baseweb="select"] > div {
        border-radius: 14px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        background-color: #111B35 !important;
        color: #FFFFFF !important;
    }

    /* Neon Green / Cyan Pill Action Button */
    div[data-testid="stForm"] button {
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%) !important;
        color: #0B132B !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 14px 28px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        width: 100% !important;
        box-shadow: 0px 10px 25px rgba(0, 242, 254, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    div[data-testid="stForm"] button:hover {
        background: linear-gradient(135deg, #38EF7D 0%, #11998E 100%) !important;
        box-shadow: 0px 14px 30px rgba(56, 239, 125, 0.4) !important;
        transform: translateY(-2px);
    }

    /* Section Headers */
    h3 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    /* Alert Boxes */
    .stAlert {
        border-radius: 16px !important;
        background-color: rgba(28, 37, 65, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #FFFFFF !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Dark Theme Gauge Chart
def create_gauge_chart(probability_val):
  percentage = probability_val * 100
  fig = go.Figure(
      go.Indicator(
          mode="gauge+number",
          value=percentage,
          number={"suffix": "%", "font": {"size": 42, "color": "#FFFFFF"}},
          gauge={
              "axis": {
                  "range": [0, 100],
                  "tickwidth": 1,
                  "tickcolor": "#64748B",
              },
              "bar": {"color": "#FF4D4F" if percentage >= 50 else "#00F2FE"},
              "steps": [
                  {"range": [0, 40], "color": "rgba(0, 242, 254, 0.1)"},
                  {"range": [40, 70], "color": "rgba(255, 215, 0, 0.1)"},
                  {"range": [70, 100], "color": "rgba(255, 77, 79, 0.1)"},
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