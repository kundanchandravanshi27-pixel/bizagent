"""
LiveKit Voice Agent with Sarvam AI TTS (Hindi / Indian Accent).
Provides STT/LLM via LiveKit Cloud and ultra-natural Hindi TTS via Sarvam AI.
Captures full conversation transcript and saves directly to Firebase Realtime Database.
"""

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from dotenv import load_dotenv
import requests

from livekit import api
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    TurnHandlingOptions,
    cli,
    inference,
)
from livekit.agents.beta import EndCallTool

# Load environment variables
load_dotenv(Path(__file__).resolve().parent / ".env.local")
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sarvam-agent")

AGENT_NAME = os.getenv("LIVEKIT_LOCAL_AGENT_NAME", "sarvam-agent")
PARTICIPANT_IDENTITY = "phone-user"

# Firebase & OpenRouter config
FIREBASE_DATABASE_URL = os.getenv(
    "FIREBASE_DATABASE_URL", 
    "https://botxpro-6fb8b-default-rtdb.firebaseio.com"
).rstrip("/")
OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY", 
    "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"
)
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free")


def sanitize_key(key: str) -> str:
    return re.sub(r"[.$#\[\]/]", "_", key)


def update_firebase_direct(phone: str, data: dict):
    """Directly update Firebase call records."""
    try:
        clean = sanitize_key(phone)
        url = f"{FIREBASE_DATABASE_URL}/calls/{clean}.json"
        requests.patch(url, json=data, timeout=5)
        logger.info(f"Firebase updated for {phone}: {list(data.keys())}")
    except Exception as e:
        logger.error(f"Failed to update Firebase: {e}")


def generate_summary(transcript_text: str) -> dict:
    """Summarize transcript via OpenRouter."""
    if not transcript_text:
        return {"summary": "Call completed. No speech recorded.", "sentiment": "Neutral"}

    prompt = f"""You are an expert AI call analyst. Analyze this phone call transcript and return a brief JSON response:

Transcript:
\"\"\"
{transcript_text}
\"\"\"

Format response strictly as JSON:
{{
  "summary": "1-2 concise sentences summarizing the conversation and outcome",
  "sentiment": "Positive" or "Neutral" or "Negative"
}}
"""
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://livekit.io",
        }
        res = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json={
                "model": OPENROUTER_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            },
            timeout=10,
        )
        if res.ok:
            raw = res.json()["choices"][0]["message"]["content"]
            cleaned = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
            return json.loads(cleaned)
    except Exception as e:
        logger.warning(f"Summary generation error: {e}")

    return {
        "summary": transcript_text[:160] + ("..." if len(transcript_text) > 160 else ""),
        "sentiment": "Neutral",
    }


class OutboundAssistant(Agent):
    def __init__(self, customer_name: str = "Customer") -> None:
        super().__init__(
            instructions="""# Prompt for Sarvam Hindi TTS Agent (LiveKit Outbound)

## Welcome Message (spoken by Sarvam TTS)
Namaste! 👋 Welcome to WorkerApp!

Main aapka AI assistant hoon. Main aapko aapki zaroorat ke hisaab se best workers dhoondhne mein help kar sakta hoon.

Aap bas bataiye aapko kis kaam ke liye worker chahiye.

🔍 Electrician, Plumber, Carpenter, Painter ya koi bhi service.

Aap apni requirement normal language mein bata sakte hain. 😊

---

## System Prompt (WorkerApp Search Assistant)
# SYSTEM PROMPT: WORKERAPP – AI NATURAL LANGUAGE WORKER SEARCH AGENT

## 1. YOUR IDENTITY
You are WorkerApp AI, an intelligent, friendly, professional and helpful AI Worker Search Assistant.

You are part of WorkerApp, a platform that connects customers with professional workers and service providers.

Your primary responsibility is to understand what type of worker a customer needs and find suitable workers from the actual WorkerApp database.

You are not a generic chatbot. You are a specialized worker discovery and search assistant.

## 2. PRIMARY OBJECTIVE
Help customers find the right workers based on their requirements.

Customers should be able to search using ordinary conversational language without filling out complicated forms.

## 3. LANGUAGE AND COMMUNICATION
- Reply in the language the customer uses (Hindi, Hinglish, English).
- Keep replies concise, friendly and conversational.
- Use emojis occasionally where appropriate.

## 4. UNDERSTAND CUSTOMER REQUIREMENTS
Extract when possible:
1. Worker category / service type.
2. Customer location.
3. Preferred date / time.
4. Maximum budget.
5. Experience / rating preferences.

Ask only for essential missing information.

## 5. INTERACTION STYLE
- Friendly greeting (already provided above).
- When a worker list is ready, present concise options with name, rating, price, and a short tagline.
- Offer to provide more details or book the worker.
---
""",
            tools=[EndCallTool()],
        )


server = AgentServer()


@server.rtc_session(agent_name=AGENT_NAME)
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info(f"Session started for room: {ctx.room.name}")

    # Parse metadata sent by server.py / make_call.py
    metadata = {}
    if ctx.job.metadata:
        try:
            metadata = json.loads(ctx.job.metadata)
        except Exception:
            metadata = {}

    phone = metadata.get("phone_number", "")
    customer_name = metadata.get("customer_name", "Customer")
    clean_phone = sanitize_key(phone) if phone else sanitize_key(ctx.room.name)

    # 1. Setup TTS engine (Sarvam AI for natural Hindi speech)
    sarvam_key = os.getenv("SARVAM_API_KEY", "sk_gusl77tg_4e4NgXZ8a0s66XZ9pihBgXgI")
    if sarvam_key:
        try:
            from livekit.plugins import sarvam
            tts_engine = sarvam.TTS(
                api_key=sarvam_key,
                target_language_code="hi-IN",
                model="bulbul:v3",
                speaker="meera",
            )
            logger.info("Initialized Sarvam AI TTS (bulbul:v3 / hi-IN / meera)")
        except Exception as e:
            logger.warning(f"Could not initialize Sarvam TTS, falling back: {e}")
            tts_engine = inference.TTS("cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc")
    else:
        tts_engine = inference.TTS("cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc")

    # 2. Setup Agent Session
    session: AgentSession = AgentSession(
        stt=inference.STT("deepgram/nova-3", language="multi"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=tts_engine,
        turn_handling=TurnHandlingOptions(turn_detection=inference.TurnDetector()),
    )

    # 3. Shutdown Callback: Extract transcript & save directly to Firebase
    async def on_call_ended():
        logger.info(f"Call session ended for room {ctx.room.name}. Compiling transcript...")
        try:
            lines = []
            if session.history and session.history.messages:
                for msg in session.history.messages:
                    if msg.role in ["user", "assistant"] and msg.content:
                        speaker = "Agent" if msg.role == "assistant" else customer_name
                        lines.append(f"{speaker}: {msg.content}")

            transcript_str = "\n".join(lines).strip()
            logger.info(f"Full transcript ({len(lines)} turns):\n{transcript_str}")

            summary_info = generate_summary(transcript_str)

            update_data = {
                "status": "Completed",
                "transcript": transcript_str if transcript_str else "Call connected and completed.",
                "ai_summary": summary_info.get("summary", "Call completed successfully with customer."),
                "sentiment": summary_info.get("sentiment", "Positive"),
                "completed_at": {".sv": "timestamp"},
            }

            if phone:
                update_firebase_direct(phone, update_data)
            else:
                update_firebase_direct(ctx.room.name, update_data)

            # Close the room for all participants
            await ctx.api.room.delete_room(api.DeleteRoomRequest(room=ctx.room.name))
        except Exception as err:
            logger.error(f"Error in on_call_ended callback: {err}")

    ctx.add_shutdown_callback(on_call_ended)

    # 4. Start agent session in room
    assistant = OutboundAssistant(customer_name=customer_name)
    await session.start(agent=assistant, room=ctx.room)

    if session.room_io:
        session.room_io.set_participant(PARTICIPANT_IDENTITY)

    # 5. Wait for customer to answer the call
    logger.info(f"Waiting for {PARTICIPANT_IDENTITY} to connect in room {ctx.room.name}...")
    callee = None
    for _ in range(40):
        callee = ctx.room.remote_participants.get(PARTICIPANT_IDENTITY)
        if callee:
            break
        await asyncio.sleep(1)

    if not callee:
        logger.warning(f"Customer did not answer within 40 seconds in {ctx.room.name}")
        if phone:
            update_firebase_direct(phone, {
                "status": "Not Answered / Busy",
                "ai_summary": f"Call to {customer_name} ({phone}) was not answered.",
                "sentiment": "Neutral",
            })
        ctx.shutdown("Customer did not answer")
        return

    logger.info(f"Customer answered! Greet customer immediately...")
    if phone:
        update_firebase_direct(phone, {"status": "In Call / Speaking"})

    # 6. Speak first greeting in Hindi using Sarvam TTS
    greeting = f"Namaste {customer_name}! Main aapki sahayata ke liye call kar raha hoon. Kya aap mujhe sun pa rahe hain?"
    try:
        await session.say(greeting)
    except Exception as e:
        logger.error(f"Failed to speak initial greeting: {e}")

    # Enforce a maximum call duration of 60 seconds
    async def _max_duration_watchdog():
        try:
            await asyncio.sleep(60)
            logger.info("Maximum call duration (60s) reached, ending call.")
            # Initiate shutdown; the shutdown callback will still run to save transcript
            ctx.shutdown("max call duration reached")
        except Exception as exc:
            logger.warning(f"Watchdog error: {exc}")

    # Start the watchdog without awaiting it (runs in background)
    asyncio.create_task(_max_duration_watchdog())



if __name__ == "__main__":
    cli.run_app(server)
