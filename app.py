import streamlit as st
import numpy as np
import pickle
import pandas as pd
import plotly.graph_objects as go

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

</style>
""", unsafe_allow_html=True)

# ---------------------------
# TITLE
# ---------------------------
st.markdown("<h1>💪 AI BMI Health Predictor</h1>", unsafe_allow_html=True)
st.markdown("<center>Smart Machine Learning Body Analysis</center>", unsafe_allow_html=True)

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.title("About App")
st.sidebar.write("AI powered BMI health analysis")
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

    gender_male = 1 if gender == "Male" else 0

    input_data = pd.DataFrame(
        [[height, weight, gender_male]],
        columns=columns
    )

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    label = int(prediction)
    category = bmi_labels.get(label, "Unknown")
    advice = diet_advice.get(category)

    # ---------------------------
    # 3D RESULT CARD
    # ---------------------------
    st.markdown(f"""
    <div class="card">
    <h2>Prediction Result</h2>
    <h1>{category}</h1>
    <p>BMI Level Code: {label}</p>
    </div>
    """, unsafe_allow_html=True)

    # ---------------------------
    # BMI GAUGE METER
    # ---------------------------
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=label,
        title={'text': "BMI Level"},
        gauge={
            'axis': {'range': [0, 5]},
            'bar': {'color': "cyan"},
            'steps': [
                {'range': [0, 1], 'color': "blue"},
                {'range': [1, 2], 'color': "lightblue"},
                {'range': [2, 3], 'color': "green"},
                {'range': [3, 4], 'color': "orange"},
                {'range': [4, 5], 'color': "red"},
            ],
        }
    ))
    st.plotly_chart(fig)

    # ---------------------------
    # AI DIET RECOMMENDATION CARD
    # ---------------------------
    st.markdown(f"""
    <div class="card">
    <h2>AI Diet Recommendation</h2>
    <p>{advice}</p>
    </div>
    """, unsafe_allow_html=True)