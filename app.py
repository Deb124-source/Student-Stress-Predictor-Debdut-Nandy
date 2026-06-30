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
    "AI-powered student stress prediction using XGBoost"
)



# -----------------------------
# Inputs
# -----------------------------

st.subheader(
    "Student Information"
)


anxiety_level = st.slider(
    "Anxiety Level (1 = Very Low, 5 = Very High)",
    1,5,3
)


self_esteem = st.slider(
    "Self Esteem (1 = Very Low, 5 = Very High)",
    1,5,3
)



mental_health_history_text = st.radio(

    "Mental Health History",

    ["No","Yes"],

    horizontal=True

)

mental_health_history = (

    1 if mental_health_history_text == "Yes"

    else 0

)



depression_text = st.radio(

    "Depression",

    ["No","Yes"],

    horizontal=True

)


depression = (

    1 if depression_text == "Yes"

    else 0

)



headache = st.slider(
    "Headache Frequency (1 = Never, 5 = Very Frequent)",
    1,5,3
)



blood_pressure = st.slider(
    "Blood Pressure Level (1 = Normal, 5 = Very High)",
    1,5,3
)



sleep_quality = st.slider(
    "Sleep Quality (1 = Poor, 5 = Excellent)",
    1,5,3
)



breathing_problem = st.slider(
    "Breathing Problem Severity (1 = None, 5 = Severe)",
    1,5,3
)



noise_level = st.slider(
    "Noise Disturbance (1 = Low, 5 = Very High)",
    1,5,3
)



living_conditions = st.slider(
    "Living Conditions (1 = Poor, 5 = Excellent)",
    1,5,3
)



safety = st.slider(
    "Safety Level (1 = Unsafe, 5 = Very Safe)",
    1,5,3
)



basic_needs = st.slider(
    "Basic Needs Satisfaction (1 = Poor, 5 = Fully Satisfied)",
    1,5,3
)



academic_performance = st.slider(
    "Academic Performance (1 = Poor, 5 = Excellent)",
    1,5,3
)



study_load = st.slider(
    "Study Load Pressure (1 = Low, 5 = Very High)",
    1,5,3
)



teacher_student_relationship = st.slider(
    "Teacher Student Relationship (1 = Poor, 5 = Excellent)",
    1,5,3
)



future_career_concerns = st.slider(
    "Future Career Concerns (1 = Low, 5 = Very High)",
    1,5,3
)



social_support = st.slider(
    "Social Support (1 = Low, 5 = Very High)",
    1,5,3
)



peer_pressure = st.slider(
    "Peer Pressure (1 = Low, 5 = Very High)",
    1,5,3
)



extracurricular_text = st.radio(

    "Extracurricular Activities",

    ["No","Yes"],

    horizontal=True

)


extracurricular_activities = (

    1 if extracurricular_text == "Yes"

    else 0

)



bullying_text = st.radio(

    "Bullying",

    ["No","Yes"],

    horizontal=True

)


bullying = (

    1 if bullying_text == "Yes"

    else 0

)




# -----------------------------
# Prediction
# -----------------------------


st.divider()


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



    st.subheader(
        "Prediction Result"
    )



    if stress_class == 2:

        st.error(
            "High Stress Risk"
        )


    elif stress_class == 1:

        st.warning(
            "Medium Stress Risk"
        )


    else:

        st.success(
            "Low Stress Risk"
        )



    st.metric(

        "Prediction Confidence",

        f"{confidence}%"

    )


    st.caption(
        "Prediction generated using XGBoost Machine Learning Model"
    )
