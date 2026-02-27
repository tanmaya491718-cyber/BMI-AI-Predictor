import streamlit as st
import numpy as np
import pickle
import pandas as pd

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

/* BMI VISUAL SCALE */

.bmi-wrapper {
width:100%;
margin-top:30px;
padding:0 6px;
box-sizing:border-box;
}

.bmi-bar {
position:relative;
width:100%;
height:26px;
border-radius:20px;
overflow:hidden;

background: linear-gradient(
to right,
#00bfff 0%,
#87cefa 16.6%,
#00c853 33.3%,
#ff9800 50%,
#ff5252 66.6%,
#b71c1c 100%
);

box-shadow:0 4px 15px rgba(0,0,0,0.3);
}

/* FIXED POINTER (DOWNWARD) */
.bmi-pointer {
position:absolute;
top:-18px;
transform:translateX(-50%);
width:0;
height:0;

border-left:12px solid transparent;
border-right:12px solid transparent;
border-top:18px solid white;

transition:left 0.8s ease-in-out;
filter: drop-shadow(0 0 6px white);
}

/* CLEAN LABEL ALIGNMENT */
.bmi-labels {
display:grid;
grid-template-columns: repeat(6, 1fr);
text-align:center;
font-size:12px;
margin-top:8px;
width:100%;
opacity:0.9;
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
    # RESULT CARD
    # ---------------------------
    st.markdown(f"""
    <div class="card">
    <h2>Prediction Result</h2>
    <h1>{category}</h1>
    <p>BMI Level Code: {label}</p>
    </div>
    """, unsafe_allow_html=True)

    # ---------------------------
    # BMI VISUAL SCALE
    # ---------------------------
    bmi_position = (label / 5) * 100

    st.markdown(f"""
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

    </div>
    """, unsafe_allow_html=True)

    # ---------------------------
    # AI DIET RECOMMENDATION CARD
    # ---------------------------
    st.markdown(f"""
    <div class="card">
    <h2>AI Diet Recommendation</h2>
    <p>{advice}</p>
    </div>
    """, unsafe_allow_html=True)