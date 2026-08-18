import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Page configuration
st.set_page_config(page_title="SmartCare AI", page_icon="🏥", layout="wide")

# Custom CSS for Premium Modern Dark Theme (Fixes Overlapping & High Contrast Dropdowns)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Overall Background */
    .stApp {
        background: #0d1117;
        color: #e6edf3;
    }
    
    /* Header Styling */
    .main-title {
        color: #ffffff;
        font-weight: 700;
        font-size: 2.2rem;
        letter-spacing: -0.5px;
    }
    .sub-title {
        color: #8b949e;
        font-size: 0.95rem;
        font-weight: 500;
        margin-bottom: 2rem;
    }

    /* Main Form Container - Modern Dark Glass */
    div[data-testid="stForm"] {
        background-color: #161b22;
        border-radius: 20px;
        padding: 35px;
        border: 1px solid #30363d;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }

    /* Input Labels */
    label, div[data-testid="stMarkdownContainer"] p {
        color: #c9d1d9 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }

    /* Clean Input Fields, Number Buttons & Dropdowns Fix */
    .stNumberInput input, div[data-baseweb="select"] > div {
        background-color: #21262d !important;
        border: 1px solid #30363d !important;
        color: #ffffff !important;
        border-radius: 10px !important;
    }

    /* Plus/Minus Buttons Fix */
    button[aria-label="Increase value"], button[aria-label="Decrease value"] {
        background-color: #30363d !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        border: none !important;
    }

    /* Dropdown Options Popup Fix */
    ul[data-baseweb="menu"] {
        background-color: #21262d !important;
        border: 1px solid #30363d !important;
    }

    /* Gradient Action Button */
    div[data-testid="stForm"] button[kind="secondaryFormSubmit"] {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        width: 100% !important;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3) !important;
        transition: all 0.2s ease-in-out;
    }
    div[data-testid="stForm"] button[kind="secondaryFormSubmit"]:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.5) !important;
    }

    /* Result Alert Cards */
    .stAlert {
        border-radius: 12px !important;
        border: 1px solid #30363d !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# High-Contrast Gauge Chart
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
                  "tickcolor": "#8B949E",
              },
              "bar": {"color": "#EF4444" if percentage >= 50 else "#10B981"},
              "steps": [
                  {"range": [0, 40], "color": "rgba(16, 185, 129, 0.15)"},
                  {"range": [40, 70], "color": "rgba(245, 158, 11, 0.15)"},
                  {"range": [70, 100], "color": "rgba(239, 68, 68, 0.15)"},
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