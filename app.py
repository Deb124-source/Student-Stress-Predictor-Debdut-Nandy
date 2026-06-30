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
# Page Setup
# -----------------------------

st.set_page_config(

    page_title="Student Stress AI",

    layout="centered"

)


st.title(
    "Student Stress Detection AI"
)


st.write(
    "XGBoost based stress level prediction system"
)



# -----------------------------
# Inputs
# -----------------------------


st.subheader(
    "Student Details"
)



anxiety_level = st.slider(
    "Anxiety Level",
    1,5,3
)


self_esteem = st.slider(
    "Self Esteem",
    1,5,3
)


mental_health_history = st.selectbox(
    "Mental Health History",
    [0,1]
)


depression = st.selectbox(
    "Depression",
    [0,1]
)


headache = st.slider(
    "Headache",
    1,5,3
)


blood_pressure = st.slider(
    "Blood Pressure",
    1,5,3
)


sleep_quality = st.slider(
    "Sleep Quality",
    1,5,3
)


breathing_problem = st.slider(
    "Breathing Problem",
    1,5,3
)


noise_level = st.slider(
    "Noise Level",
    1,5,3
)


living_conditions = st.slider(
    "Living Conditions",
    1,5,3
)


safety = st.slider(
    "Safety",
    1,5,3
)


basic_needs = st.slider(
    "Basic Needs",
    1,5,3
)


academic_performance = st.slider(
    "Academic Performance",
    1,5,3
)


study_load = st.slider(
    "Study Load",
    1,5,3
)


teacher_student_relationship = st.slider(
    "Teacher Student Relationship",
    1,5,3
)


future_career_concerns = st.slider(
    "Future Career Concerns",
    1,5,3
)


social_support = st.slider(
    "Social Support",
    1,5,3
)


peer_pressure = st.slider(
    "Peer Pressure",
    1,5,3
)


extracurricular_activities = st.selectbox(
    "Extracurricular Activities",
    [0,1]
)


bullying = st.selectbox(
    "Bullying",
    [0,1]
)



# -----------------------------
# Prediction
# -----------------------------


if st.button("Predict Stress"):


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
        max(probability[0])*100,
        2
    )



    st.subheader(
        "Prediction"
    )



    if stress_class == 2:

        st.error(
            "High Stress"
        )


    elif stress_class == 1:

        st.warning(
            "Medium Stress"
        )


    else:

        st.success(
            "Low Stress"
        )



    st.metric(

        "Confidence",

        f"{confidence}%"

    )
