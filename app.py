
import streamlit as st, os, numpy as np, joblib
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(BASE), "outputs")
st.title("Diabetes Risk Prediction")
import os

st.subheader("Model Performance & Results")


model_p = os.path.join(OUT,"stacking_model.joblib")
scaler_p = os.path.join(OUT,"scaler.joblib")
if not (os.path.exists(model_p) and os.path.exists(scaler_p)):
    st.error("Model not found. Run train.py (or run_train.bat) first.")
    st.stop()
clf = joblib.load(model_p); scaler = joblib.load(scaler_p)
st.write("Enter patient details:")
cols = st.columns(2)
with cols[0]:
    preg = st.number_input("Pregnancies", 0, 20, 1)
    bp = st.number_input("BloodPressure", 0.0, 200.0, 70.0)
    insulin = st.number_input("Insulin", 0.0, 900.0, 85.0)
    dpf = st.number_input("DiabetesPedigreeFunction", 0.0, 5.0, 0.5)
with cols[1]:
    glucose = st.number_input("Glucose", 0.0, 300.0, 120.0)
    skin = st.number_input("SkinThickness", 0.0, 100.0, 20.0)
    bmi = st.number_input("BMI", 10.0, 80.0, 28.0)
    age = st.number_input("Age", 1, 120, 33)
if st.button("Predict Risk"):
    import numpy as np
    x = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    x = scaler.transform(x)
    pred = clf.predict(x)[0]
    proba = clf.predict_proba(x)[0][1]

    st.write(f"**Probability of Diabetes:** {proba:.2f}")

    if pred == 0:
        st.success("✅ Low Risk")
    else:
        st.error("⚠️ High Risk")