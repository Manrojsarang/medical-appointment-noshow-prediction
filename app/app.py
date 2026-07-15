import plotly.express as px
import streamlit as st
import joblib
import pandas as pd

#  Page Config
st.set_page_config(
    page_title="Medical Appointment Predictor",
    page_icon="🏥",
    layout="centered"
)

#  Load Model 
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "models", "medical_no_show_model.pkl"))
feature_columns = joblib.load(os.path.join(BASE_DIR, "models", "feature_columns.pkl"))
feature_importance = pd.read_csv(os.path.join(BASE_DIR, "models", "feature_importance.csv"))


#  Sidebar 

st.sidebar.title(" Medical Appointment Predictor")

st.sidebar.markdown("---")

st.sidebar.subheader(" Model")
st.sidebar.success("Random Forest (Tuned)")

st.sidebar.subheader(" Dataset")
st.sidebar.info("110,527 Medical Appointments")

st.sidebar.subheader(" Model Accuracy")
st.sidebar.metric("Accuracy", "80.4%")

st.sidebar.subheader(" Purpose")
st.sidebar.write(
    """
Predict whether a patient
is likely to miss a scheduled
medical appointment.
"""
)

st.sidebar.markdown("---")

st.sidebar.caption("Developed using")
st.sidebar.write("• Python")
st.sidebar.write("• Scikit-learn")
st.sidebar.write("• Streamlit")
st.sidebar.write("• Pandas")

st.sidebar.markdown("---")
st.sidebar.caption("Built using Streamlit & Scikit-learn")
st.sidebar.markdown("---")

st.sidebar.subheader(" Top 10 Important Features")

fig = px.bar(
    feature_importance.head(10),
    x="Importance",
    y="Feature",
    orientation="h"
)

st.sidebar.plotly_chart(
    fig,
    use_container_width=True
)

#  Main Page 
st.title(" Medical Appointment No-Show Prediction")
st.info("Fill in the patient's information below and click Predict.")

#  Inputs 
st.subheader(" Patient Information")
col1, col2 = st.columns(2)

# Left Column
with col1:

    age = st.number_input(
        "Age",
        min_value=0,
        max_value=120,
        value=25
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    scholarship = st.selectbox(
        "Scholarship",
        ["No", "Yes"]
    )

    diabetes = st.selectbox(
        "Diabetes",
        ["No", "Yes"]
    )

    handicap = st.selectbox(
        "Handicap",
        ["No", "Yes"]
    )

    appointment_weekday = st.selectbox(
        "Appointment Weekday",
        [
            "Monday",
            "Saturday",
            "Thursday",
            "Tuesday",
            "Wednesday"
        ]
    )

# Right Column
with col2:

    waiting_days = st.number_input(
        "Waiting Days",
        min_value=0,
        max_value=365,
        value=10
    )

    sms_received = st.selectbox(
        "SMS Received",
        ["No", "Yes"]
    )

    hypertension = st.selectbox(
        "Hypertension",
        ["No", "Yes"]
    )

    alcoholism = st.selectbox(
        "Alcoholism",
        ["No", "Yes"]
    )

    scheduled_hour = st.number_input(
        "Scheduled Hour",
        min_value=0,
        max_value=23,
        value=10
    )

    neighbourhoods = sorted([col.replace("neighbourhood_", "")
    for col in feature_columns
    if col.startswith("neighbourhood_")])

neighbourhood = st.selectbox(
    "Neighbourhood",
    neighbourhoods
)

st.write("---")

#  Predict Button 

left, center, right = st.columns([1, 2, 1])

with center:
    predict = st.button("🔍 Predict", use_container_width=True)

#  Prediction 

if predict:

    input_data = pd.DataFrame({

        "gender": [1 if gender == "Male" else 0],
        "age": [age],
        "scholarship": [1 if scholarship == "Yes" else 0],
        "hypertension": [1 if hypertension == "Yes" else 0],
        "diabetes": [1 if diabetes == "Yes" else 0],
        "alcoholism": [1 if alcoholism == "Yes" else 0],
        "handicap": [1 if handicap == "Yes" else 0],
        "sms_received": [1 if sms_received == "Yes" else 0],
        "waiting_days": [waiting_days],
        "scheduled_hour": [scheduled_hour]

    })

    # Weekday Encoding
    weekday_columns = [
        "Monday",
        "Saturday",
        "Thursday",
        "Tuesday",
        "Wednesday"
    ]

    for day in weekday_columns:
        input_data[f"appointment_weekday_{day}"] = 0

    input_data[f"appointment_weekday_{appointment_weekday}"] = 1

    # Neighbourhood Encoding
    for col in feature_columns:
        if col.startswith("neighbourhood_"):
            input_data[col] = 0

    neighbourhood_column = f"neighbourhood_{neighbourhood}"

    if neighbourhood_column in input_data.columns:
        input_data[neighbourhood_column] = 1

    # Match training features
    for col in feature_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[feature_columns]

    # Prediction
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    attend_prob = probability[0][0]
    no_show_prob = probability[0][1]

    st.markdown("---")
    st.header(" Prediction Result")

    if prediction[0] == 1:

        st.error(" Prediction: Patient is likely to MISS the appointment.")

        st.metric(
            "No-Show Risk",
            f"{no_show_prob*100:.2f}%"
        )

        st.progress(float(no_show_prob))

        st.caption(
            f"Model Confidence: {no_show_prob*100:.2f}%"
        )

    else:

        st.success(" Prediction: Patient is likely to ATTEND the appointment.")

        st.metric(
            "Attendance Probability",
            f"{attend_prob*100:.2f}%"
        )

        st.progress(float(attend_prob))

        st.caption(
            f"Model Confidence: {attend_prob*100:.2f}%"
        )

    # Probability Chart

    prob_df = pd.DataFrame({
        "Class": ["Attend", "No Show"],
        "Probability": [attend_prob, no_show_prob]
    })

    st.markdown(" Prediction Probabilities")

    fig = px.bar(
        prob_df,
        x="Class",
        y="Probability",
        text="Probability"
    )

    fig.update_traces(
        texttemplate="%{text:.2%}",
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Feature Importance

    st.markdown(" Top 10 Important Features")

    st.dataframe(
        feature_importance.head(10),
        use_container_width=True
    )
 #  footer 
st.markdown("---")

st.caption(
    "developed by manrojpreet--- Built with ❤️ using Streamlit"
)