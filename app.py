import streamlit as st
import pandas as pd
import pickle


# ==========================
# Load Model
# ==========================

with open("model.pkl", "rb") as file:
    model = pickle.load(file)


# ==========================
# Page Configuration
# ==========================

st.set_page_config(
    page_title="Hospital Readmission Predictor",
    page_icon="🏥",
    layout="wide"
)


st.title("🏥 Hospital Readmission Predictor")
st.write(
    "Predict whether a diabetic patient is likely to be readmitted within 30 days."
)

st.markdown(
"""
### AI-Based Healthcare Risk Prediction System

This application uses a Machine Learning Decision Tree model
to predict the probability of 30-day diabetic patient readmission.
"""
)

st.divider()


# ==========================
# User Inputs
# ==========================

st.subheader("🧑‍⚕️ Patient Information")

col1, col2, col3 = st.columns(3)


age = st.selectbox(
    "Age Group",
    [
        "[0-10)",
        "[10-20)",
        "[20-30)",
        "[30-40)",
        "[40-50)",
        "[50-60)",
        "[60-70)",
        "[70-80)",
        "[80-90)",
        "[90-100)"
    ]
)


gender = st.selectbox(
    "Gender",
    [
        "Male",
        "Female"
    ]
)


time_in_hospital = st.slider(
    "Number of days in hospital",
    1,
    14,
    3
)


num_lab_procedures = st.slider(
    "Number of lab procedures",
    1,
    100,
    40
)


num_procedures = st.slider(
    "Number of procedures",
    0,
    6,
    1
)


num_medications = st.slider(
    "Number of medications",
    1,
    80,
    15
)


number_outpatient = st.number_input(
    "Outpatient visits",
    min_value=0,
    value=0
)


number_emergency = st.number_input(
    "Emergency visits",
    min_value=0,
    value=0
)


number_inpatient = st.number_input(
    "Previous inpatient visits",
    min_value=0,
    value=0
)


insulin = st.selectbox(
    "Insulin",
    [
        "No",
        "Steady",
        "Up",
        "Down"
    ]
)


diabetesMed = st.selectbox(
    "Diabetes Medication",
    [
        "Yes",
        "No"
    ]
)


# ==========================
# Prediction
# ==========================

if st.button("🔍 Predict Readmission"):

    input_data = pd.DataFrame(
        {
            "race": ["Caucasian"],
            "gender": [gender],
            "age": [age],
            "admission_type_id": [1],
            "discharge_disposition_id": [1],
            "admission_source_id": [1],
            "time_in_hospital": [time_in_hospital],
            "num_lab_procedures": [num_lab_procedures],
            "num_procedures": [num_procedures],
            "num_medications": [num_medications],
            "number_outpatient": [number_outpatient],
            "number_emergency": [number_emergency],
            "number_inpatient": [number_inpatient],
            "diag_1": ["250"],
            "diag_2": ["250"],
            "diag_3": ["250"],
            "number_diagnoses": [5],
            "max_glu_serum": ["None"],
            "A1Cresult": ["None"],
            "metformin": ["No"],
            "repaglinide": ["No"],
            "nateglinide": ["No"],
            "chlorpropamide": ["No"],
            "glimepiride": ["No"],
            "acetohexamide": ["No"],
            "glipizide": ["No"],
            "glyburide": ["No"],
            "tolbutamide": ["No"],
            "pioglitazone": ["No"],
            "rosiglitazone": ["No"],
            "acarbose": ["No"],
            "miglitol": ["No"],
            "troglitazone": ["No"],
            "tolazamide": ["No"],
            "examide": ["No"],
            "citoglipton": ["No"],
            "insulin": [insulin],
            "glyburide-metformin": ["No"],
            "glipizide-metformin": ["No"],
            "glimepiride-pioglitazone": ["No"],
            "metformin-rosiglitazone": ["No"],
            "metformin-pioglitazone": ["No"],
            "change": ["No"],
            "diabetesMed": [diabetesMed]
        }
    )


    prediction = model.predict(input_data)

probability = model.predict_proba(input_data)


risk = probability[0][1] * 100


st.subheader("Prediction Result")


if prediction[0] == 1:
    st.error(
        "⚠️ High Risk: Patient may be readmitted within 30 days."
    )

else:
    st.success(
        "✅ Low Risk: Patient is unlikely to be readmitted within 30 days."
    )


st.metric(
    "Readmission Risk Probability",
    f"{risk:.2f}%"
)