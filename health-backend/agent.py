import asyncio
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import JobContext, WorkerOptions, cli, llm
from livekit.plugins import deepgram, openai

# Load API keys from your environment variables
load_dotenv()

async def entrypoint(ctx: JobContext):
    print(f"📡 New voice connection received! Job ID: {ctx.job.id}")

    # Step 1: Accept the incoming WebRTC connection from the user
    await ctx.connect(auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY)

    # Step 2: Define the persona and conversational rules for the agent
    chat_context = llm.ChatContext().append(
        role="system",
        text=(
            "You are a highly advanced, natural voice assistant built for this platform. "
            "Keep your answers short and conversational (1-2 sentences maximum). "
            "Use natural vocal transitions and casual phrase choices. Do not use lists."
        )
    )

    # Step 3: Initialize the real-time conversational core worker
    assistant = agents.voice_assistant.VoiceAssistant(
        # Fast Speech-to-Text via Deepgram WebSocket streaming
        vad=openai.VAD.with_device_config(),
        stt=deepgram.STT(),
        
        # Fast LLM execution token-by-token
        llm=openai.LLM(model="gpt-4o-mini"),
        
        # Fast Text-to-Speech audio package synthesis
        tts=openai.TTS(),
        
        chat_context=chat_context
    )

    # Step 4: Boot up the assistant session inside the active audio room
    assistant.start(ctx.room)
    
    # Force the agent to speak first when the call connects
    await assistant.say("Hey there! Your live audio pipeline is officially running. How can I help you?", allow_interruptions=True)

if __name__ == "__main__":
    # Launch the multi-threaded network listening worker
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
