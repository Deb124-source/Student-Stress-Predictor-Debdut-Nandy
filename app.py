import streamlit as st
import pandas as pd
import joblib
import json


# -----------------------------
# Load Model
# -----------------------------

model = joblib.load(
    "Student_Stress_AI.pkl"
)


with open(
    "features.json",
    "r"
) as f:
    features = json.load(f)



# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(

    page_title="Student Stress AI",

    layout="centered"

)


st.title(
    "Student Stress Detection AI"
)


st.write(
    "Machine Learning based student stress prediction system"
)



# -----------------------------
# User Inputs
# -----------------------------


st.subheader(
    "Enter Student Information"
)



anxiety_level = st.slider(
    "Anxiety Level (1 = Very Low, 5 = Very High)",
    1,
    5,
    3
)


self_esteem = st.slider(
    "Self Esteem (1 = Very Low, 5 = Very High)",
    1,
    5,
    3
)


headache = st.slider(
    "Headache Frequency (1 = Never, 5 = Very Frequent)",
    1,
    5,
    3
)


blood_pressure = st.slider(
    "Blood Pressure Level (1 = Normal, 5 = Very High)",
    1,
    5,
    3
)


sleep_quality = st.slider(
    "Sleep Quality (1 = Poor, 5 = Excellent)",
    1,
    5,
    3
)


breathing_problem = st.slider(
    "Breathing Problem Severity (1 = None, 5 = Severe)",
    1,
    5,
    3
)


noise_level = st.slider(
    "Noise Disturbance Level (1 = Low, 5 = Very High)",
    1,
    5,
    3
)


living_conditions = st.slider(
    "Living Conditions (1 = Poor, 5 = Excellent)",
    1,
    5,
    3
)


safety = st.slider(
    "Safety Level (1 = Unsafe, 5 = Very Safe)",
    1,
    5,
    3
)


basic_needs = st.slider(
    "Basic Needs Satisfaction (1 = Poor, 5 = Fully Satisfied)",
    1,
    5,
    3
)


academic_performance = st.slider(
    "Academic Performance (1 = Poor, 5 = Excellent)",
    1,
    5,
    3
)


study_load = st.slider(
    "Study Load Pressure (1 = Very Low, 5 = Very High)",
    1,
    5,
    3
)


teacher_student_relationship = st.slider(
    "Teacher-Student Relationship (1 = Poor, 5 = Excellent)",
    1,
    5,
    3
)


future_career_concerns = st.slider(
    "Future Career Concerns (1 = Low, 5 = Very High)",
    1,
    5,
    3
)


social_support = st.slider(
    "Social Support (1 = Very Low, 5 = Very High)",
    1,
    5,
    3
)


peer_pressure = st.slider(
    "Peer Pressure (1 = Low, 5 = Very High)",
    1,
    5,
    3
)



extracurricular_text = st.selectbox(

    "Extracurricular Activities",

    ["No", "Yes"]

)


extracurricular_activities = (

    1 if extracurricular_text == "Yes"

    else 0

)



bullying_text = st.selectbox(

    "Bullying",

    ["No", "Yes"]

)


bullying = (

    1 if bullying_text == "Yes"

    else 0

)




# -----------------------------
# Prediction
# -----------------------------


if st.button(
    "Predict Stress Level"
):


    input_data = pd.DataFrame(

        [[

            anxiety_level,

            self_esteem,

            mental_health_history,

            depression,

            headache,

            blood_pressure,

            sleep_quality,

            breathing_problem,

            noise_level,

            living_conditions,

            safety,

            basic_needs,

            academic_performance,

            study_load,

            teacher_student_relationship,

            future_career_concerns,

            social_support,

            peer_pressure,

            extracurricular_activities,

            bullying

        ]],

        columns=features

    )


    prediction = model.predict(

        input_data

    )


    probability = model.predict_proba(

        input_data

    )



    stress_class = prediction[0]



    confidence = round(

        max(probability[0]) * 100,

        2

    )



    if stress_class == 2:

        result = "High Stress"

        st.error(result)



    elif stress_class == 1:

        result = "Medium Stress"

        st.warning(result)



    else:

        result = "Low Stress"

        st.success(result)



    st.metric(

        "Prediction Confidence",

        f"{confidence}%"

    )
