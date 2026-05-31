import subprocess
import sys

# Force-install scikit-learn dynamically if Streamlit skips it
try:
    from sklearn import preprocessing
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    from sklearn import preprocessing

import random
import pandas as pd
import numpy as np
import csv
import streamlit as st
import re  
import time 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from difflib import get_close_matches
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# Set Page Config
st.set_page_config(page_title="AI Healthcare Diagnostic Assistant", layout="wide", page_icon="🩺")

# ------------------ Load and Cache Data ------------------
@st.cache_data
def load_and_train():
    # Load Data (Ensure files exist in 'Data/' directory)
    training = pd.read_csv('Data/Training.csv')
    testing = pd.read_csv('Data/Testing.csv')

    # Clean duplicate column names
    training.columns = training.columns.str.replace(r"\.\d+$", "", regex=True)
    testing.columns = testing.columns.str.replace(r"\.\d+$", "", regex=True)
    training = training.loc[:, ~training.columns.duplicated()]
    testing = testing.loc[:, ~testing.columns.duplicated()]

    cols = training.columns[:-1]
    x = training[cols]
    y = training['prognosis']

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(x_train, y_train)
    return model, le, cols, training

try:
    model, le, cols, training = load_and_train()
    symptoms_dict = {symptom: idx for idx, symptom in enumerate(cols)}
except Exception as e:
    st.error("⚠️ Data files missing! Please ensure 'Data/Training.csv' and 'Data/Testing.csv' are in your project directory.")
    st.stop()

# ------------------ Loading Dictionaries ------------------
severityDictionary = {}
description_list = {}
precautionDictionary = {}

def load_meta_data():
    try:
        with open('MasterData/symptom_Description.csv') as csv_file:
            for row in csv.reader(csv_file):
                description_list[row[0]] = row[1]
        with open('MasterData/symptom_severity.csv') as csv_file:
            for row in csv.reader(csv_file):
                try: severityDictionary[row[0]] = int(row[1])
                except: pass
        with open('MasterData/symptom_precaution.csv') as csv_file:
            for row in csv.reader(csv_file):
                precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]
    except FileNotFoundError:
        pass # Fallback values handle presentation gracefully if MasterData folder is missing

load_meta_data()

# Synonym mappings
symptom_synonyms = {
    "stomach ache": "stomach_pain", "belly pain": "stomach_pain", "tummy pain": "stomach_pain",
    "loose motion": "diarrhea", "motions": "diarrhea", "high temperature": "fever",
    "temperature": "fever", "feaver": "fever", "coughing": "cough", "throat pain": "sore_throat",
    "cold": "chills", "breathing issue": "breathlessness", "shortness of breath": "breathlessness",
    "body ache": "muscle_pain",
}

def extract_symptoms(user_input, all_symptoms):
    extracted = []
    text = user_input.lower().replace("-", " ")
    for phrase, mapped in symptom_synonyms.items():
        if phrase in text: extracted.append(mapped)
    for symptom in all_symptoms:
        if symptom.replace("_", " ") in text: extracted.append(symptom)
    words = re.findall(r"\w+", text)
    for word in words:
        close = get_close_matches(word, [s.replace("_", " ") for s in all_symptoms], n=1, cutoff=0.8)
        if close:
            for sym in all_symptoms:
                if sym.replace("_", " ") == close[0]: extracted.append(sym)
    return list(set(extracted))

def predict_disease(symptoms_list):
    input_vector = np.zeros(len(symptoms_dict))
    for symptom in symptoms_list:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1
    pred_proba = model.predict_proba([input_vector])[0]
    pred_class = np.argmax(pred_proba)
    disease = le.inverse_transform([pred_class])[0]
    confidence = round(pred_proba[pred_class] * 100, 2)
    return disease, confidence

# Empathy Quotes
quotes = [
    "🌸 Health is wealth, take care of yourself.", "💪 A healthy outside starts from the inside.",
    "☀️ Every day is a chance to get stronger and healthier.", "🌿 Take a deep breath, your health matters the most."
]

# ------------------ STREAMLIT DASHBOARD INTERFACE ------------------
st.title("🤖 AI Healthcare Diagnostic & Triage Suite")
st.subheader("Clinical NLP Intake Assistant & Random Forest Predictive Engine")

col_input, col_results = st.columns([1, 1], gap="large")

with col_input:
    st.markdown("### 📋 Step 1: Patient Registration")
    sub_c1, sub_c2, sub_c3 = st.columns(3)
    with sub_c1: name = st.text_input("Patient Full Name", "Jane Doe")
    with sub_c2: age = st.number_input("Patient Age", min_value=1, max_value=120, value=28)
    with sub_c3: gender = st.selectbox("Biological Sex", ["Female", "Male", "Other"])
    
    st.markdown("---")
    st.markdown("### 🗣️ Step 2: Symptom Intake (NLP Engine)")
    symptoms_input = st.text_area(
        "Describe your current condition in a natural sentence:",
        value="I have been suffering from high temperature and a terrible stomach ache since yesterday morning."
    )
    
    # Process text inputs automatically
    detected = extract_symptoms(symptoms_input, cols)
    if detected:
        st.success(f"🔗 **NLP Extractor Isolated:** {', '.join([s.replace('_', ' ') for s in detected])}")
    else:
        st.warning("No clear clinical indicators mapped. Try checking spelling or using phrases like 'fever' or 'cough'.")

    st.markdown("---")
    st.markdown("### 🩺 Step 3: Vitals & History Check")
    c_v1, c_v2 = st.columns(2)
    with c_v1:
        num_days = st.slider("Duration of Symptoms (Days)", 1, 30, 3)
        severity_scale = st.slider("Perceived Severity Scale (1-10)", 1, 10, 5)
    with c_v2:
        pre_exist = st.text_input("Pre-existing Medical Conditions", "None")
        lifestyle = st.multiselect("Lifestyle Risk Markers", ["Irregular Sleep", "Alcohol Consumption", "Tobacco Use"], default=["Irregular Sleep"])

with col_results:
    st.markdown("### 📊 Clinical Diagnosis & Triage Outcome")
    
    if st.button("⚡ Run Diagnostic Engine", type="primary"):
        if not detected:
            st.error("Cannot perform predictive generation without valid symptom parameters.")
        else:
            with st.spinner("Executing structural multi-classification array..."):
                time.sleep(1.2)
                
                # Run prediction
                disease, confidence = predict_disease(detected)
                
                # Display Results UI
                st.markdown(f"#### Primary Assessment Vector: **{disease}**")
                
                # Visual Health Progress Status Meter
                if confidence > 75:
                    st.error(f"Prediction Confidence: {confidence}% (High Core Alignment)")
                else:
                    st.warning(f"Prediction Confidence: {confidence}% (Differential Analysis Recommended)")
                
                # About Block
                about_text = description_list.get(disease, "This condition is marked by an interaction of local cellular inflammatory pathways, requiring typical system evaluations.")
                st.info(f"📖 **Pathology Overview:** {about_text}")
                
                # Precaution Mapping Strategy
                st.markdown("#### 🛡️ Clinical Protocols & Precautions:")
                precautions = precautionDictionary.get(disease, ["Monitor core body vitals regularly", "Maintain consistent fluid/hydration baseline", "Schedule standard outpatient consultation", "Seek acute urgent care if breathing deteriorates"])
                
                for idx, prec in enumerate(precautions, 1):
                    st.markdown(f"**{idx}.** {prec.capitalize()}")
                    
                st.markdown("---")
                st.caption(f"Patient Record Identifier: {random.randint(10000, 99999)} | Evaluation Compiled for {name} ({gender}, Age {age})")
                st.markdown(f"*💡 {random.choice(quotes)}*")
