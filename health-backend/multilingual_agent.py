import asyncio
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from livekit.agents import AgentServer, AgentSession, JobContext, cli
from livekit.plugins import openai, silero

load_dotenv()

# =====================================================================
# 📊 LAYER 1: MACHINE LEARNING INITIALIZATION
# =====================================================================
print("🇳🇦 Booting Namibian Multilingual Health AI...")

training_df = pd.read_csv("Data/Training.csv")
desc_df = pd.read_csv("MasterData/symptom_Description.csv")
prec_df = pd.read_csv("MasterData/symptom_precaution.csv")

symptom_descriptions = dict(zip(desc_df['Disease'], desc_df['Description']))
symptom_precautions = {}
for _, row in prec_df.iterrows():
    symptom_precautions[row['Disease']] = [
        row['Precaution_1'], row['Precaution_2'], row['Precaution_3'], row['Precaution_4']
    ]

X = training_df.drop('prognosis', axis=1)
y = training_df['prognosis']
symptom_columns = list(X.columns)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X, y_encoded)

# =====================================================================
# 🗣️ LOCALIZED NAMIBIAN SYMPTOM DICTIONARY MAP
# =====================================================================
# This maps spoken local language tokens directly to your machine learning columns
NAMIBIAN_SYMPTOM_MAP = {
    # Oshiwambo variants (Oshikwanyama / Oshindonga)
    "fiva": "high_fever",
    "feba": "high_fever",
    "okusheka": "diarrhoea",
    "okukolola": "cough",
    "omutwe": "headache",
    "oshipute": "skin_rash",
    "ouyahame": "joint_pain",
    
    # Nama / Khoekhoegowab
    "tsû-||gâ": "joint_pain",
    "|amaseb": "high_fever",
    "danatsûb": "headache",
    
    # Otjiherero
    "omuvarandje": "high_fever",
    "otjitwako": "cough",
    "ozombahu": "joint_pain",
    
    # Afrikaans
    "koors": "high_fever",
    "hoes": "cough",
    "hoofpyn": "headache",
    "maagpyn": "stomach_pain",
    "gewrigspyn": "joint_pain"
}

def evaluate_namibian_speech(spoken_text: str) -> dict:
    input_vector = np.zeros(len(symptom_columns))
    symptoms_found = []
    cleaned_speech = spoken_text.lower()
    
    # 1. First scan for localized Namibian words
    for local_word, english_symptom in NAMIBIAN_SYMPTOM_MAP.items():
        if local_word in cleaned_speech:
            if english_symptom in symptom_columns:
                idx = symptom_columns.index(english_symptom)
                input_vector[idx] = 1
                symptoms_found.append(english_symptom)
                
    # 2. Second scan for standard English columns (in case they code-switch)
    for symptom in symptom_columns:
        clean_sym = symptom.replace("_", " ")
        if clean_sym in cleaned_speech:
            idx = symptom_columns.index(symptom)
            input_vector[idx] = 1
            symptoms_found.append(clean_sym)
            
    if not symptoms_found:
        return None

    prediction_index = rf_classifier.predict([input_vector])[0]
    predicted_condition = label_encoder.inverse_transform([prediction_index])[0]
    
    return {
        "condition": predicted_condition,
        "description": symptom_descriptions.get(predicted_condition, "No matching analysis mapping available."),
        "precautions": [p for p in symptom_precautions.get(predicted_condition, []) if pd.notna(p)]
    }

# =====================================================================
# 🎙️ LAYER 2: MULTILINGUAL LIVEKIT VOICE PIPELINE
# =====================================================================
server = AgentServer()

@server.rtc_session(agent_name="namibian-health-assistant")
async def entrypoint(ctx: JobContext):
    await ctx.connect()

    # Switch configuration to OpenAI Whisper to capture local language phonetics natively
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=openai.STT(model="whisper-1"), 
        llm=openai.LLM(model="gpt-4o"), # Using gpt-4o for deep cross-lingual capabilities
        tts=openai.TTS(voice="alloy"), 
    )

    await session.start(room=ctx.room)
    
    # Comprehensive system instruction defining regional capabilities
    regional_system_prompt = (
        "You are an interactive, local Namibian AI Health Representative. You can accept inputs in English, "
        "Oshikwanyama, Oshindonga, Oshikwambi, Oshinganjera, Oshibalantu, Nama/Khoekhoegowab, Otjiherero, "
        "Rukwangali, Silozi, Subiya, Mafwe, Mbukushu, Rumanyo, and Afrikaans. "
        "Always respond in the exact local language the client used to speak to you. "
        "If they speak in Oshindonga, respond back instantly in Oshindonga. If they speak in Afrikaans, "
        "respond in Afrikaans. Keep expressions culturally respectful, warm, and natural."
    )
    
    await session.generate_reply(
        instructions=f"{regional_system_prompt} Greet the user warmly and ask them in a mix of English and common local phrases how they are feeling."
    )

    @session.on("user_turn_completed")
    async def process_patient_turn(turn_info):
        patient_text = turn_info.transcript.text
        print(f"🗣️ User Spoke: '{patient_text}'")
        
        # Analyze the input speech tokens against local maps
        diagnostic_insights = evaluate_namibian_speech(patient_text)
        
        if diagnostic_insights:
            precautions_string = ", ".join(diagnostic_insights['precautions'])
            
            # The base facts derived strictly from your local Random Forest model
            clinical_facts = (
                f"Condition Detected: {diagnostic_insights['condition']}. "
                f"Details: {diagnostic_insights['description']} "
                f"Precautions to take: {precautions_string}."
            )
            
            # We command the LLM to act as our live translator for the text-to-speech engine
            translation_instruction = (
                f"{regional_system_prompt} The clinical diagnosis is: '{clinical_facts}'. "
                f"Translate this comprehensive diagnosis into the user's language smoothly. "
                f"Speak it clearly, reassuring them step by step."
            )
            
            await session.generate_reply(instructions=translation_instruction)
        else:
            # Fallback clarifying question handled natively in the detected tongue
            await session.generate_reply(
                instructions=f"{regional_system_prompt} The user said: '{patient_text}'. They did not mention distinct system symptoms. Ask them nicely in their language to clarify their physical condition."
            )

if __name__ == "__main__":
    cli.run_app(server)
