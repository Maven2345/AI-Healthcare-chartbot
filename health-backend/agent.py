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
# 📊 LAYER 1: YOUR MACHINE LEARNING MODEL INTEGRATION
# =====================================================================
print("🧠 Initializing Health AI Diagnostics Module...")

# Load your custom clinical datasets
training_df = pd.read_csv("Data/Training.csv")
desc_df = pd.read_csv("MasterData/symptom_Description.csv")
prec_df = pd.read_csv("MasterData/symptom_precaution.csv")

# Build data lookup mappings
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
    
    # Check if any column names from Training.csv are hidden inside the spoken words
    cleaned_text = spoken_text.lower().replace(" ", "_")
    for symptom in symptom_columns:
        # e.g., if user says "i have a skin rash", matches "skin_round" or "skin_rash"
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
        "description": symptom_descriptions.get(predicted_disease, "No detailed data available."),
        "precautions": [p for p in symptom_precautions.get(predicted_disease, []) if pd.notna(p)]
    }

# =====================================================================
# 🎙️ LAYER 2: REAL-TIME LIVEKIT AUDIO PIPELINE
# =====================================================================
async def entrypoint(ctx: JobContext):
    print(f"📞 Connected to Client Audio Stream. Job ID: {ctx.job.id}")
    await ctx.connect(auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY)

    # Base prompt instructions for structural vocal pacing
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

    # Intercept the user's spoken words before the LLM generates a response
    @assistant.on("user_speech_committed")
    def on_user_speech(msg: llm.ChatMessage):
        user_spoken_phrase = msg.text
        print(f"🗣️ Client Said: '{user_spoken_phrase}'")
        
        # Run the text through our Random Forest pipeline
        diagnosis_result = diagnose_voice_symptoms(user_spoken_phrase)
        
        if diagnosis_result:
            print(f"🎯 ML Match Found: {diagnosis_result['disease']}")
            precautions_text = ", ".join(diagnosis_result['precautions'])
            
            # Rewrite the LLM's brain memory on the fly with your precise clinical data
            custom_clinical_response = (
                f"Based on the symptoms you mentioned, our Random Forest analysis suggests a strong correlation with "
                f"{diagnosis_result['disease']}. To give you a bit of context: {diagnosis_result['description']} "
                f"For your safety, here are the recommended precautions you should consider right away: {precautions_text}. "
                f"Please remember to consult a medical professional if these symptoms persist."
            )
            
            # Inject it into the chat track so the voice synthesizer speaks this text
            asyncio.create_task(assistant.say(custom_clinical_response, allow_interruptions=True))

    assistant.start(ctx.room)
    await assistant.say("Hello, I am your AI Health Assistant. Please tell me what symptoms you are experiencing today.", allow_interruptions=True)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
