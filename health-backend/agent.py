import asyncio
from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.plugins import openai, deepgram

# Load your API keys from a .env file (LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, OPENAI_API_KEY)
load_dotenv()

async def entrypoint(ctx: JobContext):
    print(f"🤖 Voice Agent connecting to room: {ctx.room.name}")
    
    # Connect to the live voice/video room
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Define the assistant's personality and link it to your medical context
    assistant = llm.ChatAssistant(
        fnc_ctx=None, 
        chat_ctx=llm.ChatContext().append(
            role="system",
            text="You are an AI Healthcare Assistant. Help users list their clinical symptoms clearly."
        ),
        # Use Deepgram for speech-to-text, and OpenAI for the brain/voice
        stt=deepgram.STT(),
        llm=openai.LLM(),
        tts=openai.TTS(),
    )

    # Start the voice conversation loop in the room
    assistant.start(ctx.room)
    
    # Keep the agent alive while the user is talking
    await assistant.say("Hello, I am your healthcare assistant. What symptoms are you experiencing today?", allow_interruptions=True)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
