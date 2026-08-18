import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Page configuration
st.set_page_config(page_title="SmartCare AI", page_icon="🏥", layout="wide")

# Custom CSS for Royal Blue Modern Medical Theme
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    /* Global Typography & Background */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background-color: #f4f7fe;
    }
    
    /* Header Styling */
    .main-title {
        color: #1b2559;
        font-weight: 700;
        font-size: 2.3rem;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .sub-title {
        color: #a3edd9;
        color: #707eae;
        font-size: 0.95rem;
        font-weight: 500;
        margin-bottom: 1.8rem;
    }

    /* Form Card Container */
    div[data-testid="stForm"] {
        background-color: #ffffff;
        border-radius: 24px;
        padding: 35px;
        border: none;
        box-shadow: 0px 18px 40px rgba(112, 144, 176, 0.12);
    }

    /* Input Labels */
    label {
        color: #1b2559 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }

    /* Input Fields & Dropdowns Styling */
    .stNumberInput input, div[data-baseweb="select"] > div {
        border-radius: 14px !important;
        border: 1px solid #e9edf7 !important;
        background-color: #f8fafc !important;
        color: #1b2559 !important;
    }

    /* Pill-Style Action Button */
    div[data-testid="stForm"] button {
        background: linear-gradient(135deg, #2f54eb 0%, #1d39c4 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 14px 28px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        width: 100% !important;
        box-shadow: 0px 10px 20px rgba(47, 84, 235, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    div[data-testid="stForm"] button:hover {
        background: linear-gradient(135deg, #1d39c4 0%, #096dd9 100%) !important;
        box-shadow: 0px 14px 26px rgba(47, 84, 235, 0.4) !important;
        transform: translateY(-2px);
    }

    /* Subheaders & Info Cards */
    h3 {
        color: #1b2559 !important;
        font-weight: 700 !important;
    }

    .stAlert {
        border-radius: 16px !important;
        border: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Royal Blue Palette Gauge Chart
def create_gauge_chart(probability_val):
  percentage = probability_val * 100
  fig = go.Figure(
      go.Indicator(
          mode="gauge+number",
          value=percentage,
          number={"suffix": "%", "font": {"size": 42, "color": "#1B2559"}},
          gauge={
              "axis": {
                  "range": [0, 100],
                  "tickwidth": 1,
                  "tickcolor": "#707EAE",
              },
              "bar": {"color": "#FF4D4F" if percentage >= 50 else "#2F54EB"},
              "steps": [
                  {"range": [0, 40], "color": "#F0F5FF"},
                  {"range": [40, 70], "color": "#FFFBE6"},
                  {"range": [70, 100], "color": "#FFF1F0"},
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