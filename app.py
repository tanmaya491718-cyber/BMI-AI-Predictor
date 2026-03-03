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
# BMI Label Mapping
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

st.title("💪 BMI Health Predictor")
st.write("AI + Formula Based BMI Analysis")

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
    # 1️⃣ BMI FORMULA CALCULATION
    # ---------------------------
    bmi = weight * 10000 / (height ** 2)
    bmi = round(bmi, 2)

    # Formula category mapping
    if bmi < 16:
        formula_label = 0
    elif bmi < 18.5:
        formula_label = 1
    elif bmi < 25:
        formula_label = 2
    elif bmi < 30:
        formula_label = 3
    elif bmi < 35:
        formula_label = 4
    else:
        formula_label = 5

    formula_category = bmi_labels[formula_label]

    # ---------------------------
    # 2️⃣ MODEL PREDICTION
    # ---------------------------
    gender_male = 1 if gender == "Male" else 0
    input_data = pd.DataFrame([[height, weight, gender_male]], columns=columns)
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    model_label = int(prediction)
    model_category = bmi_labels.get(model_label, "Unknown")

    advice = diet_advice.get(formula_category)

    # ---------------------------
    # RESULT CARD
    # ---------------------------
    st.success(f"Calculated BMI: {bmi}")
    st.info(f"Formula Category: {formula_category}")
    st.warning(f"Model Prediction: {model_category}")

    if formula_label == model_label:
        st.success("✅ Formula and Model are aligned")
    else:
        st.error("⚠ Model and Formula differ")

    # ---------------------------
    # VISUAL SCALE (Formula Based)
    # ---------------------------
    segment_width = 100 / 6
    bmi_position = (formula_label * segment_width) + (segment_width / 2)

    components.html(f"""
    <div style="width:100%;margin-top:30px;font-family:sans-serif">

      <div style="
        position:relative;
        width:100%;
        height:30px;
        border-radius:30px;
        background: linear-gradient(to right,
          #00bfff 0%,
          #87cefa 16.6%,
          #00c853 33.3%,
          #ff9800 50%,
          #ff5252 66.6%,
          #b71c1c 100%);
        box-shadow:0 6px 20px rgba(0,0,0,0.3);
      ">

        <div style="
          position:absolute;
          top:-22px;
          left:{bmi_position}%;
          transform:translateX(-50%);
          width:0;height:0;
          border-left:14px solid transparent;
          border-right:14px solid transparent;
          border-top:20px solid white;
          transition:left 0.9s;
        "></div>

      </div>

      <div style="
        display:grid;
        grid-template-columns:repeat(6,1fr);
        text-align:center;
        margin-top:10px;
        font-size:12px;
      ">
        <div>Extremely Weak</div>
        <div>Weak</div>
        <div>Normal</div>
        <div>Overweight</div>
        <div>Obesity</div>
        <div>Extreme Obesity</div>
      </div>

      <div style="
        display:grid;
        grid-template-columns:repeat(6,1fr);
        text-align:center;
        margin-top:4px;
        font-weight:700;
      ">
        <div>0</div>
        <div>1</div>
        <div>2</div>
        <div>3</div>
        <div>4</div>
        <div>5</div>
      </div>

    </div>
    """, height=190)

    # ---------------------------
    # DIET RECOMMENDATION
    # ---------------------------
    st.subheader("Diet Recommendation")
    st.write(advice)