import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from google.protobuf.duration_pb2 import Duration
from pydantic import BaseModel
import requests

from livekit import api
from livekit.protocol import sip as sip_protocol

# Load environment variables
load_dotenv(Path(__file__).resolve().parent / ".env.local")
load_dotenv()  # Also check .env if present

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("outbound-server")

app = FastAPI(title="LiveKit Vobiz Outbound Server", version="1.0.0")

# Enable CORS for InfinityFree / local / any frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenRouter Configuration
OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY", 
    "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"
)
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free")

# Firebase Realtime Database URL
FIREBASE_DATABASE_URL = os.getenv(
    "FIREBASE_DATABASE_URL", 
    "https://botxpro-6fb8b-default-rtdb.firebaseio.com"
).rstrip("/")

# LiveKit & Vobiz defaults
DEFAULT_AGENT_NAME = os.getenv("LIVEKIT_AGENT_NAME", "customer-support-29a")
SIP_OUTBOUND_TRUNK_ID = os.getenv("SIP_OUTBOUND_TRUNK_ID", "ST_rdcBbUEQtLFZ")
VOBIZ_OUTBOUND_NUMBER = os.getenv("VOBIZ_OUTBOUND_NUMBER", "+918065354620")

# Helpers
def sanitize_key(key: str) -> str:
    """Firebase keys cannot contain ., $, #, [, ], or /"""
    return re.sub(r"[.$#\[\]/]", "_", key)

def update_firebase_call(phone: str, data: dict) -> None:
    """Update call record in Firebase Realtime Database via REST API."""
    try:
        clean_phone = sanitize_key(phone)
        url = f"{FIREBASE_DATABASE_URL}/calls/{clean_phone}.json"
        resp = requests.patch(url, json=data, timeout=5)
        if resp.status_code >= 400:
            logger.warning(f"Firebase update failed: {resp.status_code} {resp.text}")
    except Exception as e:
        logger.error(f"Error updating Firebase: {e}")

def generate_ai_summary(transcript_text: str) -> dict:
    """Summarize the call using OpenRouter (Gemma model)."""
    if not transcript_text:
        return {"status": "No Speech / Unknown", "summary": "No transcript available."}

    prompt = f"""You are an expert AI call analyzer. Analyze this phone call transcript and return a short JSON object.

Transcript:
\"\"\"
{transcript_text}
\"\"\"

Respond ONLY in valid JSON matching this structure:
{{
  "call_status": "Answered" or "Rejected / Busy" or "Wrong Number" or "Voicemail",
  "ai_summary": "1-2 crisp sentences describing what was discussed and the customer's response",
  "customer_sentiment": "Positive" or "Neutral" or "Negative" or "Interested" or "Not Interested"
}}
"""
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://livekit.io",
            "X-Title": "LiveKit Call Summarizer",
        }
        body = {
            "model": OPENROUTER_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=body,
            timeout=15,
        )
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"].strip()
            # Clean JSON markdown fences if present
            cleaned = re.sub(r"^```(?:json)?|```$", "", content, flags=re.MULTILINE).strip()
            return json.loads(cleaned)
        else:
            logger.warning(f"OpenRouter returned {resp.status_code}: {resp.text}")
    except Exception as e:
        logger.error(f"AI summarization error: {e}")

    return {
        "call_status": "Answered",
        "ai_summary": transcript_text[:180] + ("..." if len(transcript_text) > 180 else ""),
        "customer_sentiment": "Neutral",
    }


class CallRequest(BaseModel):
    phone_number: str
    name: Optional[str] = "Customer"
    agent_name: Optional[str] = None
    custom_data: Optional[Dict[str, Any]] = None


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "LiveKit Vobiz Outbound Voice Caller",
        "default_agent": DEFAULT_AGENT_NAME,
        "trunk_id": SIP_OUTBOUND_TRUNK_ID,
    }


@app.post("/api/call")
async def trigger_call(req: CallRequest):
    """Trigger an outbound call to a single phone number."""
    phone = req.phone_number.strip()
    if not re.fullmatch(r"\+[1-9]\d{6,14}", phone):
        raise HTTPException(status_code=400, detail="Invalid E.164 phone format (e.g. +919876543210)")

    agent_name = req.agent_name or DEFAULT_AGENT_NAME
    trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID", SIP_OUTBOUND_TRUNK_ID)
    caller_id = os.getenv("VOBIZ_OUTBOUND_NUMBER", VOBIZ_OUTBOUND_NUMBER)

    clean_phone = sanitize_key(phone)
    room_name = f"call_{clean_phone}_{os.urandom(3).hex()}"
    metadata = {
        "phone_number": phone,
        "clean_phone": clean_phone,
        "customer_name": req.name or "Customer",
        "agent_name": agent_name,
        "custom_data": req.custom_data or {},
    }

    # Record initial state in Firebase
    update_firebase_call(phone, {
        "phone": phone,
        "name": req.name or "Customer",
        "agent": agent_name,
        "status": "Dialing...",
        "room_name": room_name,
        "updated_at": {".sv": "timestamp"},
    })

    lkapi = api.LiveKitAPI()
    try:
        # 1. Dispatch agent to the room
        logger.info(f"Dispatching agent '{agent_name}' for {phone} in room {room_name}")
        dispatch = await lkapi.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name=agent_name,
                room=room_name,
                metadata=json.dumps(metadata),
            )
        )

        # 2. Dial phone number via SIP Outbound Trunk
        logger.info(f"Initiating SIP call to {phone} using trunk {trunk_id}")
        participant = await lkapi.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                room_name=room_name,
                sip_trunk_id=trunk_id,
                sip_call_to=phone,
                sip_number=caller_id,
                participant_identity="phone-user",
                include_headers=sip_protocol.SIP_X_HEADERS,
                krisp_enabled=True,
                ringing_timeout=Duration(seconds=45),
                wait_until_answered=False,  # Return immediately so web request doesn't timeout
            )
        )

        update_firebase_call(phone, {
            "status": "Ringing...",
            "sip_participant_id": participant.participant_id,
            "dispatch_id": dispatch.id,
        })

        return {
            "success": True,
            "message": f"Calling {phone}",
            "room_name": room_name,
            "dispatch_id": dispatch.id,
            "agent_name": agent_name,
        }

    except api.TwirpError as e:
        logger.error(f"LiveKit TwirpError: {e.message}")
        update_firebase_call(phone, {
            "status": "Failed / Busy",
            "error": e.message,
        })
        raise HTTPException(status_code=500, detail=f"Call failed: {e.message}")
    except Exception as e:
        logger.error(f"Call trigger exception: {e}")
        update_firebase_call(phone, {
            "status": "Error",
            "error": str(e),
        })
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await lkapi.aclose()


@app.post("/api/webhook/summary")
async def call_summary_webhook(request: Request, bg_tasks: BackgroundTasks):
    """
    Endpoint configured in LiveKit Cloud Console:
    'Call ending' -> 'Summary endpoint URL'
    Receives call results / transcripts when a call finishes.
    """
    try:
        body = await request.json()
        logger.info(f"Received LiveKit call summary webhook: {body}")
    except Exception:
        body = {}

    # Extract room, transcript, or summary fields from LiveKit payload
    room_name = (
        body.get("room_name")
        or body.get("room", {}).get("name")
        or body.get("room_id")
        or body.get("room", {}).get("sid")
    )
    raw_summary = body.get("summary") or ""
    results = body.get("results") or {}
    transcript = body.get("transcript") or body.get("transcription") or raw_summary

    # Phone number can be inside metadata, results, or room name
    phone = (
        results.get("phone_number")
        or body.get("metadata", {}).get("phone_number")
        or body.get("phone_number")
    )

    # Process summary via OpenRouter asynchronously
    def process_and_save():
        analysis = generate_ai_summary(str(raw_summary or transcript or results))
        update_data = {
            "status": analysis.get("call_status", "Completed"),
            "ai_summary": analysis.get("ai_summary", raw_summary or "Call completed."),
            "sentiment": analysis.get("customer_sentiment", "Neutral"),
            "completed_at": {".sv": "timestamp"},
            "raw_results": results,
        }

        # 1. Look for existing call in Firebase by room_name or phone
        target_key = None
        try:
            resp = requests.get(f"{FIREBASE_DATABASE_URL}/calls.json", timeout=5)
            if resp.status_code == 200 and resp.json():
                all_calls = resp.json()
                for db_key, call_info in all_calls.items():
                    if isinstance(call_info, dict):
                        # Match by room_name
                        if room_name and call_info.get("room_name") == room_name:
                            target_key = db_key
                            break
                        # Match by phone number suffix
                        if phone and (db_key.endswith(phone[-10:]) or sanitize_key(phone) == db_key):
                            target_key = db_key
                            break
                        # Match if room_name contains this db_key
                        if room_name and db_key in room_name:
                            target_key = db_key
                            break
        except Exception as e:
            logger.error(f"Error searching Firebase for call: {e}")

        # Fallback: if not found via search, derive from room_name or phone
        if not target_key:
            if phone:
                target_key = sanitize_key(phone)
            elif room_name and room_name.startswith("call_"):
                parts = room_name.split("_")
                if len(parts) >= 3:
                    target_key = "_".join(parts[1:-1])
            elif room_name:
                target_key = sanitize_key(room_name)

        if target_key:
            logger.info(f"Updating Firebase call under key '{target_key}' with summary")
            url = f"{FIREBASE_DATABASE_URL}/calls/{target_key}.json"
            try:
                r = requests.patch(url, json=update_data, timeout=5)
                logger.info(f"Firebase patch status: {r.status_code}")
            except Exception as e:
                logger.error(f"Failed to update Firebase key {target_key}: {e}")
        else:
            logger.warning(f"Could not determine target key for summary payload: {body}")

    bg_tasks.add_task(process_and_save)

    return {"status": "ok", "message": "Summary received"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)

