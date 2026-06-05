import os
import asyncio
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from livekit import agents
from livekit.agents import JobContext, WorkerOptions, cli, llm
from livekit.plugins import deepgram, openai

load_dotenv()

# =====================================================================
# 📊 LAYER 1: DYNAMIC PATHS & MACHINE LEARNING MODEL INTEGRATION
# =====================================================================
print("🧠 Initializing Health AI Diagnostics Module...")

# Detect exactly where agent.py lives to prevent Windows folder errors
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

training_path = os.path.join(BASE_DIR, "Data", "Training.csv")
desc_path = os.path.join(BASE_DIR, "MasterData", "symptom_Description.csv")
prec_path = os.path.join(BASE_DIR, "MasterData", "symptom_precaution.csv")

# Load your custom clinical datasets using absolute references
training_df = pd.read_csv(training_path)

# Fix for missing header row in description file
desc_df = pd.read_csv(desc_path, header=None, names=['Disease', 'Description'])

# Fix for missing header row in precaution file (Disease + 4 precautions columns)
prec_df = pd.read_csv(prec_path, header=None, names=['Disease', 'Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4'])

# Clean string paddings or trailing whitespace if any exist in data text cells
desc_df['Disease'] = desc_df['Disease'].str.strip()
prec_df['Disease'] = prec_df['Disease'].str.strip()

# Build data lookup mappings matching your custom column structural names
symptom_descriptions = dict(zip(desc_df['Disease'], desc_df['Description']))
symptom_precautions = {}
for _, row in prec_df.iterrows():
    symptom_precautions[row['Disease']] = [
        row['Precaution_1'], row['Precaution_2'], 
        row['Precaution_3'], row['Precaution_4']
    ]

# Train your exact Random Forest Classifier
X = training_df.drop('prognosis', axis=1)
y = training_df['prognosis']
symptom_columns = list(X.columns)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y_encoded)
print("🚀 Health AI Module loaded. Random Forest Model trained successfully.")

# Custom internal helper function to match spoken words to your dataset
def diagnose_voice_symptoms(spoken_text: str) -> dict:
    input_vector = np.zeros(len(symptom_columns))
    symptoms_found = []
    
    cleaned_text = spoken_text.lower().replace(" ", "_")
    for symptom in symptom_columns:
        if symptom in cleaned_text or symptom.replace("_", " ") in spoken_text.lower():
            idx = symptom_columns.index(symptom)
            input_vector[idx] = 1
            symptoms_found.append(symptom.replace("_", " "))
            
    if not symptoms_found:
        return None

    # Compute machine learning prognosis
    pred_idx = rf_model.predict([input_vector])[0]
    predicted_disease = label_encoder.inverse_transform([pred_idx])[0]
    
    return {
        "disease": predicted_disease,
        "description": symptom_descriptions.get(predicted_disease.strip(), "No detailed data available."),
        "precautions": [p for p in symptom_precautions.get(predicted_disease.strip(), []) if pd.notna(p)]
    }

# =====================================================================
# 🎙️ LAYER 2: REAL-TIME LIVEKIT AUDIO PIPELINE
# =====================================================================
async def entrypoint(ctx: JobContext):
    print(f"📞 Connected to Client Audio Stream. Job ID: {ctx.job.id}")
    await ctx.connect(auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY)

    chat_context = llm.ChatContext().append(
        role="system",
        text=(
            "You are an empathetic, clinical voice assistant. Your job is to list user symptoms "
            "and provide the diagnostic breakdown calculated by the machine learning model. "
            "Keep speech completely natural, friendly, and concise. Speak in short paragraphs."
        )
    )

    assistant = agents.voice_assistant.VoiceAssistant(
        vad=openai.VAD.with_device_config(),
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(),
        chat_context=chat_context
    )

    @assistant.on("user_speech_committed")
    def on_user_speech(msg: llm.ChatMessage):
        user_spoken_phrase = msg.text
        print(f"🗣️ Client Said: '{user_spoken_phrase}'")
        
        diagnosis_result = diagnose_voice_symptoms(user_spoken_phrase)
        
        if diagnosis_result:
            print(f"🎯 ML Match Found: {diagnosis_result['disease']}")
            precautions_text = ", ".join(diagnosis_result['precautions'])
            
            custom_clinical_response = (
                f"Based on the symptoms you mentioned, our Random Forest analysis suggests a strong correlation with "
                f"{diagnosis_result['disease']}. To give you a bit of context: {diagnosis_result['description']} "
                f"For your safety, here are the recommended precautions you should consider right away: {precautions_text}. "
                f"Please remember to consult a medical professional if these symptoms persist."
            )
            
            asyncio.create_task(assistant.say(custom_clinical_response, allow_interruptions=True))

    assistant.start(ctx.room)
    await assistant.say("Hello, I am your AI Health Assistant. Please tell me what symptoms you are experiencing today.", allow_interruptions=True)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))