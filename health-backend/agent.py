import asyncio
from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.plugins import openai, deepgram

# Load your API keys from a .env file (LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, OPENAI_API_KEY)
load_dotenv()

async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AgentOptions.AUDIO_ONLY)
    
    # 1. Your code that loads the Namibian guidelines txt file into a string:
    with open("../KnowledgeBase/namibia_health_guidelines.txt", "r", encoding="utf-8") as f:
        knowledge_context = f.read()

    # 2. PLACE THE NEW PIECE RIGHT HERE:
    regional_system_prompt = (
        f"You are an interactive, local Namibian AI Health Representative. You can accept inputs in English, "
        f"Oshikwanyama, Oshindonga, Oshikwambi, Oshinganjera, Oshibalantu, Nama/Khoekhoegowab, Otjiherero, "
        f"Rukwangali, Silozi, Subiya, Mafwe, Mbukushu, Rumanyo, and Afrikaans.\n\n"
        f"GROUND TRUTH REGIONAL KNOWLEDGE BASE TO USE:\n{knowledge_context}\n"
        f"Always respond in the exact local language the client used to speak to you."
    )

    # 3. Pass it to your agent initialization block right below it:
    agent = VoicePipelineAgent(
        vad=ctx.proc.vad,
        asr=deepgram.ASR(),
        llm=openai.LLM(system_prompt=regional_system_prompt), # <-- Injected here
        tts=openai.TTS(),
    )
    
    agent.start(ctx.room)
    
    # Keep the agent alive while the user is talking
    await assistant.say("Hello, I am your healthcare assistant. What symptoms are you experiencing today?", allow_interruptions=True)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
