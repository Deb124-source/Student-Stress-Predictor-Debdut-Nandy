# Student Stress Detection AI

An AI-powered machine learning application that predicts student stress levels using academic, psychological, health, and lifestyle factors.

The project uses an XGBoost classification model and provides an interactive Streamlit web application where users can enter student information and get a stress risk prediction with confidence score.

---

## Live Demo

https://student-stress-predictor-debdut-nandy.streamlit.app/

---

## Project Overview

Student stress is influenced by multiple factors such as academic workload, sleep quality, anxiety, social support, mental health conditions, and lifestyle patterns.

This project analyzes these factors and builds a machine learning model to classify students into:

- Low Stress
- Medium Stress
- High Stress

The final model is deployed as a user-friendly AI application using Streamlit.

---

## Features

- Machine learning based stress prediction
- XGBoost classification model
- Interactive Streamlit interface
- User-friendly input system
- Confidence score prediction
- Real-time stress risk assessment
- No model retraining required during deployment

---

## Tech Stack

### Programming Language
- Python

### Machine Learning
- XGBoost
- Scikit-learn

### Data Processing
- Pandas
- NumPy

### Visualization / Deployment
- Streamlit

### Model Saving
- Joblib

---

## Dataset Features

The model uses different student-related factors:

- Anxiety Level
- Self Esteem
- Mental Health History
- Depression
- Headache Frequency
- Blood Pressure
- Sleep Quality
- Breathing Problems
- Noise Level
- Living Conditions
- Safety
- Basic Needs Satisfaction
- Academic Performance
- Study Load
- Teacher Student Relationship
- Future Career Concerns
- Social Support
- Peer Pressure
- Extracurricular Activities
- Bullying

Target:

```

stress_level

```

Classes:

```

0 → Low Stress
1 → Medium Stress
2 → High Stress

```

---

## Machine Learning Workflow

```

Dataset
|
Data Cleaning
|
Feature Selection
|
Train-Test Split
|
XGBoost Classifier
|
Model Evaluation
|
Streamlit Deployment

```

---

## Model Used

### XGBoost Classifier

Reasons for choosing XGBoost:

- Handles tabular data efficiently
- Good performance on classification tasks
- Handles feature importance analysis
- Robust against complex relationships

---

## Model Evaluation

The model was evaluated using:

- Accuracy Score
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## Application Preview

The Streamlit application allows users to enter:

- Mental health indicators
- Academic factors
- Lifestyle conditions
- Social factors

and generates:

```

Predicted Stress Level

Confidence Score

```

---

## Project Structure

```

Student-Stress-Predictor

│
├── app.py
│
├── Student_Stress_AI.pkl
│
├── features.json
│
├── requirements.txt
│
└── README.md

````

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/student-stress-predictor.git
````

Navigate into the project:

```bash
cd student-stress-predictor
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Application

Start Streamlit:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## Streamlit Deployment

The application can be deployed using:

* Streamlit Cloud

Required files:

```
app.py
Student_Stress_AI.pkl
features.json
requirements.txt
```

---

## Future Improvements

* Add personalized stress management suggestions
* Add historical tracking dashboard
* Improve prediction using larger datasets
* Add explainable AI using SHAP values
* Integrate student wellness analytics dashboard

---

## Disclaimer

This application is developed for educational and analytical purposes. It is not a medical diagnostic system.

---

## Author

Debdut Nandy
