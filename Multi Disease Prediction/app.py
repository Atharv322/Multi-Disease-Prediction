import streamlit as st
import pickle
import numpy as np
from streamlit_lottie import st_lottie
import requests

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(page_title="Multi Disease Prediction", page_icon="🩺", layout="wide")

# Load Lottie Animations
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_health = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json")
lottie_heart = load_lottieurl("https://assets2.lottiefiles.com/private_files/lf30_5ttqPi.json")
lottie_diabetes = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_qp1q7mct.json")
lottie_kidney = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_x1ikbkcj.json")
lottie_liver = load_lottieurl("https://assets9.lottiefiles.com/private_files/lf30_editor_yz3rlj.json")   # ✅ working liver animation
lottie_stroke = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_c9gmsx1z.json")            # ✅ working stroke animation
lottie_parkinson = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_tutvdkg0.json")

# ---------------------------------------------------
# Load Models
# ---------------------------------------------------
dbt_model = pickle.load(open(r"C:\Users\athar\OneDrive\Documents\Desktop\streamlit\Multi Disease Prediction\diabetes\model_d.pkl","rb"))
dbt_scaler = pickle.load(open(r"C:\Users\athar\OneDrive\Documents\Desktop\streamlit\Multi Disease Prediction\diabetes\scaler_d.pkl","rb"))

heart_model = pickle.load(open(r"C:\Users\athar\OneDrive\Documents\Desktop\streamlit\Multi Disease Prediction\heart\model_h.pkl","rb"))

pk_model = pickle.load(open(r"C:\Users\athar\OneDrive\Documents\Desktop\streamlit\Multi Disease Prediction\parkinson\model_pk.pkl","rb"))
pk_scaler = pickle.load(open(r"C:\Users\athar\OneDrive\Documents\Desktop\streamlit\Multi Disease Prediction\parkinson\scaler_pk.pkl","rb"))

kd_model = pickle.load(open(r"C:\Users\athar\OneDrive\Documents\Desktop\streamlit\Multi Disease Prediction\kidney\model_kd.pkl","rb"))
kd_scaler = pickle.load(open(r"C:\Users\athar\OneDrive\Documents\Desktop\streamlit\Multi Disease Prediction\kidney\scaler_kd.pkl","rb"))

li_model = pickle.load(open(r"C:\Users\athar\OneDrive\Documents\Desktop\streamlit\Multi Disease Prediction\liver\model_li.pkl","rb"))
li_scaler = pickle.load(open(r"C:\Users\athar\OneDrive\Documents\Desktop\streamlit\Multi Disease Prediction\liver\scaler_li.pkl","rb"))

sp_model = pickle.load(open(r"C:\Users\athar\OneDrive\Documents\Desktop\streamlit\Multi Disease Prediction\stroke\model_sp.pkl","rb"))
sp_scaler = pickle.load(open(r"C:\Users\athar\OneDrive\Documents\Desktop\streamlit\Multi Disease Prediction\stroke\scaler_sp.pkl","rb"))

# ---------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------
st.sidebar.title("🩺 Multi-Disease Prediction System")
page = st.sidebar.radio("Go to:",
                           ("🏠 Home", "Diabetes", "Heart Disease", "Parkinson's", "Kidney Disease", "Liver Disease", "Stroke"))

# ---------------------------------------------------
# Home Page
# ---------------------------------------------------
if page == "🏠 Home":
    st.title("🩺 Multi-Disease Prediction System")
    st.markdown("### An AI powered system to predict multiple diseases with high accuracy 🚀")
    if lottie_health:
        st_lottie(lottie_health, height=300, key="health")
    st.info("👈 Select a disease from the sidebar to start prediction")

# ---------------------------------------------------
# Diabetes
# ---------------------------------------------------
elif page == "Diabetes":
    st.title("🩸 Diabetes Prediction")
    if lottie_diabetes:
        st_lottie(lottie_diabetes, height=200, key="dbt")

    col1, col2 = st.columns(2)
    with col1:
        Pregnancies = st.number_input("Pregnancies", 0, 20)
        Glucose = st.number_input("Glucose Level", 0, 300)
        BloodPressure = st.number_input("Blood Pressure", 0, 200)
        SkinThickness = st.number_input("Skin Thickness", 0, 100)
    with col2:
        Insulin = st.number_input("Insulin", 0, 900)
        BMI = st.number_input("BMI", 0.0, 70.0)
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", 0.0, 5.0)
        Age = st.number_input("Age", 1, 120)

    if st.button("🔍 Predict Diabetes"):
        features = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                              BMI, DiabetesPedigreeFunction, Age]])
        scaled = dbt_scaler.transform(features)
        prediction = dbt_model.predict(scaled)

        if prediction[0] == 1:
            st.error("⚠️ The person is likely to have **Diabetes**.")
        else:
            st.success("✅ The person is not likely to have **Diabetes**.")

# ---------------------------------------------------
# Heart Disease
# ---------------------------------------------------
elif page == "Heart Disease":
    st.title("❤️ Heart Disease Prediction")
    if lottie_heart:
        st_lottie(lottie_heart, height=200, key="heart")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 1, 120)
        sex = st.selectbox("Sex", ("Female", "Male"))
        sex = 1 if sex == "Male" else 0
        cp = st.number_input("Chest Pain Type (0-3)", 0, 3)
        trestbps = st.number_input("Resting Blood Pressure", 0, 250)
        chol = st.number_input("Cholesterol", 0, 600)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", (0, 1))
    with col2:
        restecg = st.number_input("Rest ECG (0-2)", 0, 2)
        thalach = st.number_input("Max Heart Rate", 0, 300)
        exang = st.selectbox("Exercise Induced Angina", (0, 1))
        oldpeak = st.number_input("Oldpeak", 0.0, 10.0)
        slope = st.number_input("Slope (0-2)", 0, 2)
        ca = st.number_input("CA (0-4)", 0, 4)
        thal = st.number_input("Thal (0-3)", 0, 3)

    if st.button("🔍 Predict Heart Disease"):
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                              thalach, exang, oldpeak, slope, ca, thal]])
        prediction = heart_model.predict(features)

        if prediction[0] == 1:
            st.error("⚠️ The person is likely to have **Heart Disease**.")
        else:
            st.success("✅ The person is not likely to have **Heart Disease**.")

# ---------------------------------------------------
# Parkinson's
# ---------------------------------------------------
elif page == "Parkinson's":
    st.title("🧠 Parkinson's Disease Prediction")
    if lottie_parkinson:
        st_lottie(lottie_parkinson, height=200, key="pk")

    params = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)",
              "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)",
              "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA",
              "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"]

    features = []
    for p in params:
        features.append(st.number_input(p, value=0.0))

    if st.button("🔍 Predict Parkinson's"):
        features = np.array([features])
        scaled = pk_scaler.transform(features)
        prediction = pk_model.predict(scaled)

        if prediction[0] == 1:
            st.error("⚠️ The person is likely to have **Parkinson's Disease**.")
        else:
            st.success("✅ The person is not likely to have **Parkinson's Disease**.")

# ---------------------------------------------------
# Kidney Disease
# ---------------------------------------------------
elif page == "Kidney Disease":
    st.title("🧪 Chronic Kidney Disease Prediction")
    if lottie_kidney:
        st_lottie(lottie_kidney, height=200, key="kd")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 0, 120, 45)
        bp = st.number_input("Blood Pressure", 0, 200, 60)
        sg = st.number_input("Specific Gravity", 1.0, 2.0, 1.02, step=0.01)
        al = st.number_input("Albumin", 0.0, 10.0, 0.0)
        su = st.number_input("Sugar", 0.0, 10.0, 0.0)
        rbc = st.selectbox("Red Blood Cells", ["Abnormal (0)", "Normal (1)"])
        rbc = 1 if "Normal" in rbc else 0
        pc = st.selectbox("Pus Cell", ["Abnormal (0)", "Normal (1)"])
        pc = 1 if "Normal" in pc else 0
        pcc = st.selectbox("Pus Cell Clumps", ["No (0)", "Yes (1)"])
        pcc = 1 if "Yes" in pcc else 0
        ba = st.selectbox("Bacteria", ["No (0)", "Yes (1)"])
        ba = 1 if "Yes" in ba else 0
        bgr = st.number_input("Blood Glucose Random", 0.0, 500.0, 114.0)
        bu = st.number_input("Blood Urea", 0.0, 200.0, 26.0)
        sc = st.number_input("Serum Creatinine", 0.0, 15.0, 0.7)

    with col2:
        sod = st.number_input("Sodium", 100.0, 200.0, 141.0)
        pot = st.number_input("Potassium", 2.0, 10.0, 4.2)
        hemo = st.number_input("Hemoglobin", 3.0, 20.0, 15.0)
        pcv = st.number_input("Packed Cell Volume", 10.0, 60.0, 43.0)
        wc = st.number_input("White Blood Cell Count", 1000.0, 30000.0, 9200.0)
        rc = st.number_input("Red Blood Cell Count", 2.0, 10.0, 5.8)
        htn = st.selectbox("Hypertension", ["No (0)", "Yes (1)"])
        htn = 1 if "Yes" in htn else 0
        dm = st.number_input("Diabetes Mellitus", 0,3)
        cad = st.selectbox("Coronary Artery Disease", ["No (0)", "Yes (1)"])
        cad = 1 if "Yes" in cad else 0
        appet = st.selectbox("Appetite", ["Good (0)", "Poor (1)"])
        appet = 1 if "Poor" in appet else 0
        pe = st.selectbox("Pedal Edema", ["No (0)", "Yes (1)"])
        pe = 1 if "Yes" in pe else 0
        ane = st.selectbox("Anemia", ["No (0)", "Yes (1)"])
        ane = 1 if "Yes" in ane else 0

    # Collect inputs into array
    features = np.array([[age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod,
                          pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane]])

    # Predict button
    if st.button("🔍 Predict Kidney Disease"):
        scaled = kd_scaler.transform(features)
        prediction = kd_model.predict(scaled)

        if prediction[0] == 1:
            st.error("⚠️ The person is likely to have **Kidney Disease**.")
        else:
            st.success("✅ The person is not likely to have **Kidney Disease**.")

# ---------------------------------------------------
# Liver Disease
# ---------------------------------------------------
elif page == "Liver Disease":
    st.title("🧬 Liver Disease Prediction")
    if lottie_liver:
        st_lottie(lottie_liver, height=200, key="liver")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 1, 120)
        gender = st.selectbox("Gender", ("Female", "Male"))
        gender = 1 if gender == "Male" else 0
        tb = st.number_input("Total Bilirubin", 0.0, 10.0)
        db = st.number_input("Direct Bilirubin", 0.0, 10.0)
        alkphos = st.number_input("Alkaline Phosphotase", 0, 2000)
    with col2:
        sgpt = st.number_input("Alamine Aminotransferase (SGPT)", 0, 2000)
        sgot = st.number_input("Aspartate Aminotransferase (SGOT)", 0, 2000)
        tp = st.number_input("Total Proteins", 0.0, 10.0)
        alb = st.number_input("Albumin", 0.0, 10.0)
        ag = st.number_input("Albumin and Globulin Ratio", 0.0, 5.0)

    if st.button("🔍 Predict Liver Disease"):
        features = np.array([[age, gender, tb, db, alkphos, sgpt, sgot, tp, alb, ag]])
        scaled = li_scaler.transform(features)
        prediction = li_model.predict(scaled)

        if prediction[0] == 1:
            st.error("⚠️ The person is likely to have **Liver Disease**.")
        else:
            st.success("✅ The person is not likely to have **Liver Disease**.")

# ---------------------------------------------------
# Stroke
# ---------------------------------------------------
elif page == "Stroke":
    st.title("🧠 Stroke Prediction")
    if lottie_stroke:
        st_lottie(lottie_stroke, height=200, key="sp")

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ("Female", "Male"))
        gender = 1 if gender == "Male" else 0
        age = st.number_input("Age", 1, 120)
        hypertension = st.selectbox("Hypertension", (0, 1))
        heart_disease = st.selectbox("Heart Disease", (0, 1))
        ever_married = st.selectbox("Ever Married", (0, 1))
    with col2:
        work_type = st.number_input("Work Type (0-3)", 0, 3)
        Residence_type = st.selectbox("Residence Type", ("Rural", "Urban"))
        Residence_type = 1 if Residence_type == "Urban" else 0
        avg_glucose_level = st.number_input("Average Glucose Level", 0.0, 300.0)
        bmi = st.number_input("BMI", 0.0, 70.0)
        smoking_status = st.number_input("Smoking Status (0-3)", 0, 3)

    if st.button("🔍 Predict Stroke"):
        features = np.array([[gender, age, hypertension, heart_disease, ever_married,
                              work_type, Residence_type, avg_glucose_level, bmi, smoking_status]])
        scaled = sp_scaler.transform(features)
        prediction = sp_model.predict(scaled)

        if prediction[0] == 1:
            st.error("⚠️ The person is likely to have **Stroke**.")
        else:
            st.success("✅ The person is not likely to have **Stroke**.")

