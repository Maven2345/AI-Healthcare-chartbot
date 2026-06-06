from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from typing import List

app = FastAPI(title="AI Healthcare Diagnostic API")

# Allow your future Frontend to talk to this backend safely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for ML
model = None
label_encoder = None
symptom_columns = []
symptom_descriptions = {}
symptom_precautions = {}

@app.on_event("startup")
def load_and_train_model():
    global model, label_encoder, symptom_columns, symptom_descriptions, symptom_precautions
    try:
        # Load your CSV data files
        training_df = pd.read_csv("Data/Training.csv")
        desc_df = pd.read_csv("MasterData/symptom_Description.csv")
        prec_df = pd.read_csv("MasterData/symptom_precaution.csv")
        
        # Build lookup dictionaries
        for _, row in desc_df.iterrows():
            symptom_descriptions[row['Disease']] = row['Description']
            
        for _, row in prec_df.iterrows():
            symptom_precautions[row['Disease']] = [
                row['Precaution_1'], row['Precaution_2'], 
                row['Precaution_3'], row['Precaution_4']
            ]

        # Process features and targets
        X = training_df.drop('prognosis', axis=1)
        y = training_df['prognosis']
        symptom_columns = list(X.columns)

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        # Train the Random Forest Model instantly on startup
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y_encoded)
        print("\n🚀 SUCCESS: Machine Learning model trained and API is live!\n")
        
    except Exception as e:
        print(f"\n❌ STARTUP ERROR: Could not load files or train model. Details: {str(e)}\n")

class PredictionRequest(BaseModel):
    symptoms: List[str]

@app.post("/api/predict")
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model training failed on startup.")

    # Create a matching blank input array (all zeros)
    input_vector = np.zeros(len(symptom_columns))
    symptoms_found = False

    # Check incoming symptoms and flip bits from 0 to 1
    for symptom in request.symptoms:
        formatted_symptom = symptom.strip().replace(" ", "_").lower()
        if formatted_symptom in symptom_columns:
            idx = symptom_columns.index(formatted_symptom)
            input_vector[idx] = 1
            symptoms_found = True

    if not symptoms_found:
        raise HTTPException(status_code=400, detail="No matching symptoms found in clinical database.")

    # Run machine learning prediction
    pred_idx = model.predict([input_vector])[0]
    predicted_disease = label_encoder.inverse_transform([pred_idx])[0]

    return {
        "prognosis": predicted_disease,
        "description": symptom_descriptions.get(predicted_disease, "No description available."),
        "precautions": [p for p in symptom_precautions.get(predicted_disease, ["Consult a doctor."]) if pd.notna(p)]
    }
