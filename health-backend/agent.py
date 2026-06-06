import asyncio
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from livekit.agents import AgentServer, AgentSession, JobContext, cli
from livekit.plugins import deepgram, openai, silero

load_dotenv()

# =====================================================================
# 📊 LAYER 1: TRAIN YOUR CLINICAL HEALTHCARE MODEL ON STARTUP
# =====================================================================
print("🧠 System Booting: Initializing Health AI Datasets...")

# Load your exact CSV dataset matrices
training_df = pd.read_csv("Data/Training.csv")
desc_df = pd.read_csv("MasterData/symptom_Description.csv")
prec_df = pd.read_csv("MasterData/symptom_precaution.csv")

# Generate lightning-fast memory lookup dicts
symptom_descriptions = dict(zip(desc_df['Disease'], desc_df['Description']))
symptom_precautions = {}
for _, row in prec_df.iterrows():
    symptom_precautions[row['Disease']] = [
        row['Precaution_1'], row['Precaution_2'], 
        row['Precaution_3'], row['Precaution_4']
    ]

# Split target column 'prognosis' and fit feature structure
X = training_df.drop('prognosis', axis=1)
y = training_df['prognosis']
symptom_columns = list(X.columns)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Fit Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X, y_encoded)
print("🚀 SUCCESS: Machine Learning model successfully trained and locked in memory.")


# Internal function to map human speech directly into machine learning arrays
def evaluate_clinical_speech(spoken_text: str) -> dict:
    # Initialize a clean, blank zero vector matching Training.csv length
    input_vector = np.zeros(len(symptom_columns))
    symptoms_found = []
    
    cleaned_speech = spoken_text.lower().replace(" ", "_")
    
    # Track down if any dataset symptom strings match the spoken phrase
    for symptom in symptom_columns:
        if symptom in cleaned_speech or symptom.replace("_", " ") in spoken_text.lower():
            idx = symptom_columns.index(symptom)
            input_vector[idx] = 1
            symptoms_found.append(symptom.replace("_", " "))
            
    if not symptoms_found:
        return None

    # Run raw math matrix calculation prediction
    prediction_index = rf_classifier.predict([input_vector])[0]
    predicted_condition = label_encoder.inverse_transform([prediction_index])[0]
    
    return {
        "condition": predicted_condition,
        "description": symptom_descriptions.get(predicted_condition, "No baseline analysis mapping available."),
        "precautions": [p for p in symptom_precautions.get(predicted_condition, []) if pd.notna(p)]
    }

# =====================================================================
# 🎙️ LAYER 2: INTERACTIVE LIVEKIT REAL-TIME VOICE PIPELINE
# =====================================================================
server = AgentServer()

@server.rtc_session(agent_name="health-assistant")
async def entrypoint(ctx: JobContext):
    print(f"📡 Establishing continuous audio hand-shake. Session ID: {ctx.job.id}")
    
    # Accept user audio channel connection
    await ctx.connect()

    # Construct the base streaming multi-modal environment container
    session = AgentSession(
        vad=silero.VAD.load(), # Ultra-low latency voice activity detector
        stt=deepgram.STT(model="deepgram/nova-3", language="en"), # Real-time speech-to-text
        llm=openai.LLM(model="openai/gpt-4.1-mini"), # Handles language routing
        tts=openai.TTS(), # Renders voice soundwaves
    )

    # Boot the workspace room session
    await session.start(room=ctx.room)
    
    # Introduce the agent to the caller using high-fidelity TTS audio
    await session.generate_reply(
        instructions="Greet the patient warmly, tell them you are an automated AI Healthcare agent, and ask what physical symptoms they are experiencing."
    )

    # Intercept speech tokens the exact millisecond the user stops talking
    @session.on("user_turn_completed")
    async def process_patient_turn(turn_info):
        # Extract the text transcription produced by Deepgram
        patient_text = turn_info.transcript.text
        print(f"🗣️ Patient Verbalized: '{patient_text}'")
        
        # Route the text directly through our local Random Forest model calculation
        diagnostic_insights = evaluate_clinical_speech(patient_text)
        
        if diagnostic_insights:
            print(f"🎯 Classifier Prediction: {diagnostic_insights['condition']}")
            precautions_string = ", ".join(diagnostic_insights['precautions'])
            
            # Formulate the response containing your explicit dataset contents
            clinical_response_template = (
                f"Based on the clinical indicators you described, our Random Forest analytical module detects a strong alignment with "
                f"{diagnostic_insights['condition']}. To help you understand this better: {diagnostic_insights['description']} "
                f"Please ensure you follow these safety precautions carefully: {precautions_string}. "
                f"Always make sure to see a human practitioner for comprehensive validation."
            )
            
            # Bypass generic AI chatter and force the voice generator to speak your explicit data response
            await session.say(text=clinical_response_template)
        else:
            # If no explicit symptom matches were hit, fall back to helpful AI clarifying tracking
            await session.generate_reply(
                instructions=f"The user said: '{patient_text}'. Acknowledge gracefully that you need more clarity, and ask them to describe their symptoms with more specific medical terms."
            )

if __name__ == "__main__":
    cli.run_app(server)
