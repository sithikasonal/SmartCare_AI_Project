import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="SmartCare AI", page_icon="🏥", layout="wide")


def create_gauge_chart(probability_val):
  percentage = probability_val * 100
  fig = go.Figure(
      go.Indicator(
          mode="gauge+number",
          value=percentage,
          number={"suffix": "%", "font": {"size": 40, "color": "white"}},
          gauge={
              "axis": {
                  "range": [0, 100],
                  "tickwidth": 1,
                  "tickcolor": "white",
              },
              "bar": {"color": "#e74c3c" if percentage >= 50 else "#2ecc71"},
              "steps": [
                  {"range": [0, 40], "color": "rgba(46, 204, 113, 0.2)"},
                  {"range": [40, 70], "color": "rgba(241, 196, 15, 0.2)"},
                  {"range": [70, 100], "color": "rgba(231, 76, 60, 0.2)"},
              ],
          },
      )
  )
  fig.update_layout(
      height=280,
      margin=dict(l=20, r=20, t=30, b=20),
      paper_bgcolor="rgba(0,0,0,0)",
      font={"color": "white"},
  )
  return fig


@st.cache_resource
def load_model():
  try:
    # model.pkl හෝ model.joblib ලෙස ඇත්නම් load කරගනී
    return joblib.load("model.pkl")
  except Exception:
    return joblib.load("model.joblib")


try:
  model_pipeline = load_model()
except Exception as e:
  st.error(f"Model එක Load කිරීමට අපහසුයි. Error: {e}")

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

    st.plotly_chart(create_gauge_chart(missing_prob), use_container_width=True)

    st.subheader("📋 Recommended action")
    if is_high_risk:
      st.info("""
            * **Standard discharge planning** with a follow-up appointment reminder.
            * **Monitor** at next scheduled visit.
            * **Automated Reminders:** Send SMS and WhatsApp reminders 24 hours prior.
            * **Direct Follow-up:** Assign a clinic coordinator to confirm attendance via phone call.
            """)
    else:
      st.info("""
            * **Standard Process:** Follow standard appointment and reminder workflow.
            * **Automated Notification:** Send routine SMS reminder 1 day prior.
            """)

  except Exception as err:
    st.error(f"Prediction Error: {err}")