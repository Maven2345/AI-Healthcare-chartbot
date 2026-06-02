from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from typing import List

app = FastAPI(title="AI Healthcare Diagnostic API", version="2.0")

# Enable CORS so your Next.js frontend can communicate with it safely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, swap "*" for your actual frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold our trained model and data structures
model = None
label_encoder = None
symptom_columns = []

# Data mapping dictionaries
symptom_descriptions = {}
symptom_precautions = {}

class PredictionRequest(BaseModel):
    symptoms: List[str]

class PredictionResponse(BaseModel):
    prognosis: str
    description: str
    precautions: List[str]

@app.on_event("startup")
def load_and_train_model():
    global model, label_encoder, symptom_columns, symptom_descriptions, symptom_precautions
    try:
        # Load Datasets
        training_df = pd.read_csv("Data/Training.csv")
        desc_df = pd.read_csv("MasterData/symptom_Description.csv")
        prec_df = pd.read_csv("MasterData/symptom_precaution.csv")
        
        # Parse descriptions and precautions into lookup dictionaries
        for _, row in desc_df.iterrows():
            symptom_descriptions[row['Disease']] = row['Description']
            
        for _, row in prec_df.iterrows():
            symptom_precautions[row['Disease']] = [
                row['Precaution_1'], row['Precaution_2'], 
                row['Precaution_3'], row['Precaution_4']
            ]

        # Prepare ML Training Data
        X = training_df.drop('prognosis', axis=1)
        y = training_df['prognosis']
        symptom_columns = list(X.columns)

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        # Train Model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y_encoded)
        print("🚀 Machine Learning model trained and ready for API requests.")
        
    except Exception as e:
        print(f"❌ Error during server startup: {str(e)}")

@app.post("/api/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not initialized.")

    # Create a blank input vector matching the exact format of Training.csv
    input_vector = np.zeros(len(symptom_columns))
    
    # Flip the index bit to 1 for symptoms provided by the user
    symptoms_found = False
    for symptom in request.symptoms:
        formatted_symptom = symptom.strip().replace(" ", "_").lower()
        if formatted_symptom in symptom_columns:
            idx = symptom_columns.index(formatted_symptom)
            input_vector[idx] = 1
            symptoms_found = True

    if not symptoms_found:
        raise HTTPException(status_code=400, detail="None of the provided symptoms match our clinical dataset.")

    # Run Prediction
    pred_idx = model.predict([input_vector])[0]
    predicted_disease = label_encoder.inverse_transform([pred_idx])[0]

    # Pull metadata with fallbacks
    description = symptom_descriptions.get(predicted_disease, "No detailed description available.")
    precautions = symptom_precautions.get(predicted_disease, ["Rest and stay hydrated.", "Consult a doctor if symptoms worsen."])

    return {
        "prognosis": predicted_disease,
        "description": description,
        "precautions": [p for p in precautions if pd.notna(p)]
    }
