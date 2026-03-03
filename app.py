import streamlit as st
import numpy as np
import pickle
import pandas as pd
import streamlit.components.v1 as components

# ---------------------------
# Load saved ML objects
# ---------------------------
model = pickle.load(open("bmi_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# ---------------------------
# BMI Label Mapping (0-5)
# ---------------------------
bmi_labels = {
    0: "Extremely Weak",
    1: "Weak",
    2: "Normal",
    3: "Overweight",
    4: "Obesity",
    5: "Extreme Obesity"
}

# ---------------------------
# Diet Recommendation AI
# ---------------------------
diet_advice = {
    "Extremely Weak": "Increase calorie intake. Eat nuts, milk, bananas, rice and protein rich food.",
    "Weak": "Balanced high calorie diet with healthy fats and proteins.",
    "Normal": "Maintain balanced diet. Stay active and hydrated.",
    "Overweight": "Reduce sugar, fried food. Start daily exercise.",
    "Obesity": "Low calorie diet. High fiber foods. Regular cardio.",
    "Extreme Obesity": "Strict diet plan + medical consultation recommended."
}

# ---------------------------
# Page settings
# ---------------------------
st.set_page_config(page_title="BMI Predictor", layout="centered")

# ---------------------------
# APPLE STYLE UI + GLASS UI
# ---------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #1f1c2c, #928dab);
    color: white;
}

.main {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(15px);
    padding: 30px;
    border-radius: 20px;
}

h1 {
    text-align:center;
    font-weight:700;
}

.stButton>button {
    background: linear-gradient(45deg,#00c6ff,#0072ff);
    color:white;
    height:50px;
    border-radius:12px;
    font-size:18px;
    border:none;
    transition:0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
}

.card {
    background: rgba(255,255,255,0.1);
    padding:25px;
    border-radius:20px;
    text-align:center;
    box-shadow: 0 10px 25px rgba(0,0,0,0.3);
    margin-top:20px;
}

.small {
    opacity:0.9;
    font-size:14px;
    margin-top:6px;
}

.match {
    margin-top:12px;
    padding:10px;
    border-radius:12px;
    font-weight:700;
}

.good { 
    background: rgba(0,255,140,0.15); 
    border: 1px solid rgba(0,255,140,0.35); 
}

.bad  { 
    background: rgba(255,80,80,0.15); 
    border: 1px solid rgba(255,80,80,0.35); 
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# TITLE
# ---------------------------
st.markdown("<h1>💪 BMI Health Predictor</h1>", unsafe_allow_html=True)
st.markdown("<center>Smart Machine Learning Body Analysis</center>", unsafe_allow_html=True)

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.title("About App")
st.sidebar.write("ML powered BMI health analysis")
st.sidebar.write("Enter height & weight to check health condition")

# ---------------------------
# User Inputs
# ---------------------------
gender = st.selectbox("Gender", ["Male", "Female"])
height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0)
weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0)

# ---------------------------
# Prediction button
# ---------------------------
if st.button("Predict BMI Category"):

    # ---------------------------
    # 1) BMI FORMULA (DISPLAY + FORMULA CATEGORY)
    # ---------------------------
    bmi_value = round((weight * 10000) / (height ** 2), 2)

    # Map BMI value -> label 0..5
    if bmi_value < 16:
        formula_label = 0
    elif bmi_value < 18.5:
        formula_label = 1
    elif bmi_value < 25:
        formula_label = 2
    elif bmi_value < 30:
        formula_label = 3
    elif bmi_value < 35:
        formula_label = 4
    else:
        formula_label = 5

    formula_category = bmi_labels[formula_label]

    # ---------------------------
    # 2) MODEL PREDICTION (UNCHANGED)
    # ---------------------------
    gender_male = 1 if gender == "Male" else 0
    input_data = pd.DataFrame([[height, weight, gender_male]], columns=columns)
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    model_label = int(prediction)
    model_category = bmi_labels.get(model_label, "Unknown")

    advice = diet_advice.get(formula_category, "No advice available.")

    aligned = (formula_label == model_label)

    # ---------------------------
    # RESULT CARD (FIXED: HTML RENDER)
    # ---------------------------
    st.markdown(f"""
    <div class="card">
        <h2>Prediction Result</h2>
        <h1>{formula_category}</h1>

        <div class="small">Calculated BMI (Formula): <b>{bmi_value}</b></div>
        <div class="small">Formula Level Code: <b>{formula_label}</b></div>
        <div class="small">Model Predicted Code: <b>{model_label}</b> → <b>{model_category}</b></div>

        <div class="match {'good' if aligned else 'bad'}">
            {"✅ Model & Formula are aligned" if aligned else "⚠ Model & Formula differ (dataset labels may differ)"}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---------------------------
    # BMI HORIZONTAL VISUAL SCALE (Formula-based pointer)
    # ---------------------------
    segment_width = 100 / 6
    bmi_position = (formula_label * segment_width) + (segment_width / 2)

    components.html(f"""
    <div class="bmi-wrapper">

      <div class="bmi-bar">
          <div class="bmi-pointer" style="left:{bmi_position}%"></div>
      </div>

      <div class="bmi-labels">
          <div>Extremely Weak</div>
          <div>Weak</div>
          <div>Normal</div>
          <div>Overweight</div>
          <div>Obesity</div>
          <div>Extreme Obesity</div>
      </div>

      <div class="bmi-numbers">
          <div>0</div>
          <div>1</div>
          <div>2</div>
          <div>3</div>
          <div>4</div>
          <div>5</div>
      </div>

    </div>

    <style>
    .bmi-wrapper {{
      width: 100%;
      margin-top: 30px;
      padding: 0 6px;
      box-sizing: border-box;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
    }}

    .bmi-bar {{
      position: relative;
      width: 100%;
      height: 30px;
      border-radius: 30px;
      overflow: hidden;
      background: linear-gradient(to right,
        #00bfff 0%,
        #87cefa 16.6%,
        #00c853 33.3%,
        #ff9800 50%,
        #ff5252 66.6%,
        #b71c1c 100%
      );
      box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }}

    /* upside-down triangle */
    .bmi-pointer {{
      position: absolute;
      top: -22px;
      transform: translateX(-50%);
      width: 0;
      height: 0;
      border-left: 14px solid transparent;
      border-right: 14px solid transparent;
      border-top: 20px solid #ffffff;
      transition: left 0.9s cubic-bezier(.25,.8,.25,1);
      filter: drop-shadow(0 0 8px rgba(255,255,255,0.9));
    }}

    .bmi-labels {{
      display: grid;
      grid-template-columns: repeat(6, 1fr);
      text-align: center;
      margin-top: 12px;
      font-size: 12px;
      color: rgba(255,255,255,0.95);
      line-height: 1.2;
    }}

    .bmi-numbers {{
      display: grid;
      grid-template-columns: repeat(6, 1fr);
      text-align: center;
      margin-top: 5px;
      font-size: 14px;
      font-weight: 700;
      color: #ffffff;
      letter-spacing: 0.5px;
    }}

    @media (max-width: 480px) {{
      .bmi-labels {{
        font-size: 10px;
      }}
      .bmi-numbers {{
        font-size: 12px;
      }}
    }}
    </style>
    """, height=200)

    # ---------------------------
    # Diet Recommendation (Design preserved)
    # ---------------------------
    st.markdown(f"""
    <div class="card">
        <h2>Diet Recommendation</h2>
        <p>{advice}</p>
    </div>
    """, unsafe_allow_html=True)