import joblib
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(page_title="SmartCare AI", page_icon="🏥", layout="wide")

# Custom CSS matching the Dark Theme
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Dynamic Mesh Gradient Background */
    .stApp {
        background: radial-gradient(at 0% 0%, #0d3b66 0px, transparent 50%),
                    radial-gradient(at 100% 0%, #00b4d8 0px, transparent 50%),
                    radial-gradient(at 50% 100%, #3a0ca3 0px, transparent 50%),
                    #0a0e1a;
        background-attachment: fixed;
        color: #f8fafc;
    }
    
    /* Header Styling */
    .main-title {
        color: #ffffff;
        font-weight: 700;
        font-size: 2.3rem;
        letter-spacing: -0.5px;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 0.95rem;
        font-weight: 500;
        margin-bottom: 2rem;
    }

    /* Translucent Modern Glass Card */
    div[data-testid="stForm"] {
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 24px;
        padding: 35px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
    }

    /* Input Labels */
    label, div[data-testid="stMarkdownContainer"] p {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }

    /* Clean Dark Inputs & Dropdowns */
    .stNumberInput input, div[data-baseweb="select"] > div {
        background-color: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        color: #ffffff !important;
        border-radius: 12px !important;
    }

    /* Plus / Minus Buttons */
    button[aria-label="Increase value"], button[aria-label="Decrease value"] {
        background-color: rgba(51, 65, 85, 0.8) !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        border: none !important;
    }

    /* Dropdown Options Menu Fix */
    ul[data-baseweb="menu"] {
        background-color: #0f172a !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
    }

    /* Gradient Glowing Action Button */
    div[data-testid="stForm"] button[kind="secondaryFormSubmit"] {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%) !important;
        color: #030712 !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 14px 28px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        width: 100% !important;
        box-shadow: 0 8px 25px rgba(0, 242, 254, 0.35) !important;
        transition: all 0.3s ease-in-out;
    }
    
    div[data-testid="stForm"] button[kind="secondaryFormSubmit"]:hover {
        background: linear-gradient(135deg, #38ef7d 0%, #11998e 100%) !important;
        box-shadow: 0 10px 30px rgba(56, 239, 125, 0.45) !important;
        transform: translateY(-2px);
    }

    /* Alert Styling */
    .stAlert {
        background-color: rgba(15, 23, 42, 0.85) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #ffffff !important;
    }

    /* Gauge Container */
    .gauge-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 20px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Custom Semi-Arc Semi-Circle Gauge (Matching the provided Image)
def render_arc_gauge(probability_val):
  percent = round(probability_val * 100, 1)

  # Semi-circle radius = 90, circumference of semi-circle = π * r = 3.14159 * 90 = 282.74
  arc_length = 282.74

  # Calculate filled stroke offset according to probability percentage
  offset = arc_length - (arc_length * (percent / 100.0))

  svg_code = f"""
    <div class="gauge-container">
        <svg width="280" height="180" viewBox="0 0 220 140">
            <defs>
                <!-- Yellow to Green to Cyan Gradient (Matching image style) -->
                <linearGradient id="arcGrad" x1="0%" y1="100%" x2="100%" y2="0%">
                    <stop offset="0%" stop-color="#FFE600" />
                    <stop offset="45%" stop-color="#A3E635" />
                    <stop offset="75%" stop-color="#22D3EE" />
                    <stop offset="100%" stop-color="#00A3FF" />
                </linearGradient>
            </defs>

            <!-- Background Dark Track Arc -->
            <path d="M 20 120 A 90 90 0 0 1 200 120" 
                  fill="none" 
                  stroke="#2D3139" 
                  stroke-width="18" 
                  stroke-linecap="round" />

            <!-- Active Gradient Meter Arc with Rounded Ends -->
            <path d="M 20 120 A 90 90 0 0 1 200 120" 
                  fill="none" 
                  stroke="url(#arcGrad)" 
                  stroke-width="18" 
                  stroke-linecap="round" 
                  stroke-dasharray="{arc_length}" 
                  stroke-dashoffset="{offset}" 
                  style="transition: stroke-dashoffset 1s ease-in-out;" />

            <!-- Percentage Display Text -->
            <text x="110" y="110" 
                  font-family="'Plus Jakarta Sans', sans-serif" 
                  font-size="44" 
                  font-weight="300" 
                  fill="#FFFFFF" 
                  text-anchor="middle">
                {percent}%
            </text>
        </svg>
    </div>
    """
  st.markdown(svg_code, unsafe_allow_html=True)


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

    # Render Image-styled Arc Gauge Meter
    render_arc_gauge(missing_prob)

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