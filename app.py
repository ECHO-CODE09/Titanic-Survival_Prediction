import streamlit as st
import pandas as pd
import numpy as np
import joblib


model = joblib.load("survival_prediction_model.jb")

scaler = joblib.load("scaler.jb")

st.title("🚢 Titanic Survival Predictor")

st.write("Enter passenger details:")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 1, 80, 25)
sibsp = st.number_input("Siblings/Spouse", 0, 10, 0)
parch = st.number_input("Parents/Children", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 500.0, 50.0)

if st.button("Predict"):

    sex = 1 if sex == 'female' else 0


    data = pd.DataFrame([[pclass, sex, age, sibsp, parch, fare]],
                        columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'])


    data_scaled = scaler.transform(data)

    
    result = model.predict(data_scaled)
    prob = model.predict_proba(data_scaled)[0][1]

    
    if result[0] == 1:
        st.success(f"✅ Survived (Confidence: {prob:.2f})")
    else:
        st.error(f"❌ Not Survived (Confidence: {1 - prob:.2f})")
