import asyncio
import json
import logging
import os
import re
import subprocess
import sys
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
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("outbound-server")

app = FastAPI(title="LiveKit Vobiz Outbound Server", version="1.1.0")

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

# Background worker process handle
_worker_proc = None


@app.on_event("startup")
async def startup_event():
    """Start the local Sarvam AI agent worker in the background on Render."""
    global _worker_proc
    try:
        agent_script = Path(__file__).resolve().parent / "agent.py"
        if agent_script.exists():
            logger.info("Starting background Sarvam voice worker (agent.py start)...")
            _worker_proc = subprocess.Popen(
                [sys.executable, str(agent_script), "start"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info("Background Sarvam worker started with PID %s", _worker_proc.pid)
    except Exception as e:
        logger.warning("Could not auto-start background worker: %s", e)


@app.on_event("shutdown")
def shutdown_event():
    global _worker_proc
    if _worker_proc:
        try:
            _worker_proc.terminate()
            logger.info("Background worker terminated.")
        except Exception:
            pass


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
        return {"status": "No Speech / Unknown", "summary": "No transcript available.", "sentiment": "Neutral"}

    prompt = f"""You are an expert AI call analyzer. Analyze this phone call transcript and return a short JSON object.

Transcript:
\"\"\"
{transcript_text}
\"\"\"

Respond ONLY in valid JSON matching this structure:
{{
  "call_status": "Completed" or "Rejected / Busy" or "Wrong Number" or "Voicemail",
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
            cleaned = re.sub(r"^```(?:json)?|```$", "", content, flags=re.MULTILINE).strip()
            return json.loads(cleaned)
        else:
            logger.warning(f"OpenRouter returned {resp.status_code}: {resp.text}")
    except Exception as e:
        logger.error(f"AI summarization error: {e}")

    return {
        "call_status": "Completed",
        "ai_summary": transcript_text[:180] + ("..." if len(transcript_text) > 180 else ""),
        "customer_sentiment": "Neutral",
    }


class CallRequest(BaseModel):
    phone_number: str
    name: Optional[str] = "Customer"
    agent_name: Optional[str] = None
    custom_data: Optional[Dict[str, Any]] = None


class SummarizeRequest(BaseModel):
    phone_number: str
    name: Optional[str] = "Customer"
    agent_name: Optional[str] = DEFAULT_AGENT_NAME
    transcript: Optional[str] = None


@app.post("/api/summarize_call")
async def manual_summarize_call(req: SummarizeRequest):
    """Generate or refresh AI summary for a call on demand."""
    phone = req.phone_number.strip()
    clean_phone = sanitize_key(phone)

    # 1. Check if there's any transcript passed or in Firebase
    transcript_text = req.transcript or ""
    if not transcript_text:
        try:
            r = requests.get(f"{FIREBASE_DATABASE_URL}/calls/{clean_phone}.json", timeout=5)
            call_data = r.json() or {}
            transcript_text = (
                call_data.get("transcript")
                or call_data.get("ai_summary")
                or f"Call between AI Agent '{req.agent_name}' and customer '{req.name}' ({phone}). Call answered and conversation completed."
            )
        except Exception:
            transcript_text = f"Outbound AI call to {req.name} ({phone}) via {req.agent_name}."

    # 2. Call OpenRouter
    analysis = generate_ai_summary(transcript_text)
    update_data = {
        "status": "Completed",
        "ai_summary": analysis.get("ai_summary", "Call completed successfully."),
        "sentiment": analysis.get("customer_sentiment", "Neutral"),
        "completed_at": {".sv": "timestamp"},
    }

    # 3. Update Firebase
    update_firebase_call(phone, update_data)
    return {"success": True, "data": update_data}


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "LiveKit Vobiz Outbound Voice Caller",
        "default_agent": DEFAULT_AGENT_NAME,
        "trunk_id": SIP_OUTBOUND_TRUNK_ID,
    }


@app.post("/api/call")
async def trigger_call(req: CallRequest, bg_tasks: BackgroundTasks):
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
                wait_until_answered=False,
            )
        )

        update_firebase_call(phone, {
            "status": "Ringing...",
            "sip_participant_id": participant.participant_id,
            "dispatch_id": dispatch.id,
        })

        # Save room mapping and latest active call in Firebase
        try:
            # Query room to get the exact LiveKit room SID (RM_...)
            room_obj = await lkapi.room.get_room(api.GetRoomRequest(room=room_name))
            room_sid = room_obj.sid
            requests.patch(f"{FIREBASE_DATABASE_URL}/room_mappings.json", json={
                room_sid: phone,
                room_name: phone,
            }, timeout=3)
        except Exception:
            room_sid = None

        try:
            requests.patch(f"{FIREBASE_DATABASE_URL}/latest_active_call.json", json={
                "phone": phone,
                "room_name": room_name,
                "room_sid": room_sid or "",
                "customer_name": req.name or "Customer",
                "timestamp": {".sv": "timestamp"},
            }, timeout=3)
        except Exception:
            pass

        # 3. Start background room watcher to detect answer & call completion
        bg_tasks.add_task(
            monitor_call_lifecycle,
            room_name,
            phone,
            req.name or "Customer",
            agent_name
        )

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


async def monitor_call_lifecycle(room_name: str, phone: str, customer_name: str, agent_name: str):
    """
    Monitor the LiveKit room to detect when the call is answered and when it ends.
    Acts as a guaranteed fallback if the LiveKit summary webhook doesn't fire.
    """
    lk = api.LiveKitAPI()
    call_answered = False
    start_time = asyncio.get_event_loop().time()
    max_duration = 300  # 5 minutes max

    try:
        while (asyncio.get_event_loop().time() - start_time) < max_duration:
            await asyncio.sleep(2)
            try:
                res = await lk.room.list_participants(api.ListParticipantsRequest(room=room_name))
                identities = [p.identity for p in res.participants]

                if "phone-user" in identities:
                    if not call_answered:
                        call_answered = True
                        logger.info(f"Call answered by {phone} in {room_name}")
                        update_firebase_call(phone, {"status": "In Call / Speaking"})

                elif call_answered and "phone-user" not in identities:
                    logger.info(f"Call ended for {phone} in {room_name}")
                    break

            except api.TwirpError:
                if call_answered:
                    logger.info(f"Room {room_name} closed after call answered")
                    break
                else:
                    logger.info(f"Room {room_name} closed before answering")
                    update_firebase_call(phone, {
                        "status": "Not Answered / Busy",
                        "completed_at": {".sv": "timestamp"},
                        "ai_summary": f"Call to {customer_name} ({phone}) was not answered or rejected.",
                        "sentiment": "Neutral",
                    })
                    return
            except Exception as e:
                logger.warning(f"Error checking room {room_name}: {e}")

        # Wait 4 seconds for webhook or agent.py shutdown callback to deliver transcript
        await asyncio.sleep(4)

        clean_phone = sanitize_key(phone)
        try:
            r = requests.get(f"{FIREBASE_DATABASE_URL}/calls/{clean_phone}.json", timeout=5)
            existing = r.json() or {}
            existing_summary = existing.get("ai_summary") or existing.get("transcript")
            if existing_summary and existing_summary != "{}" and existing.get("status") == "Completed":
                return
        except Exception:
            pass

        logger.info(f"Generating fallback AI summary via OpenRouter for {phone}")
        analysis = generate_ai_summary(
            f"Phone call with customer '{customer_name}' ({phone}) via AI Agent '{agent_name}'. "
            "Call was answered and conversation completed."
        )

        update_firebase_call(phone, {
            "status": "Completed",
            "completed_at": {".sv": "timestamp"},
            "ai_summary": analysis.get("ai_summary", "Call completed successfully with customer."),
            "sentiment": analysis.get("customer_sentiment", "Positive"),
        })

    except Exception as e:
        logger.error(f"Error in monitor_call_lifecycle for {phone}: {e}")
    finally:
        await lk.aclose()


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

    room_id = (
        body.get("room_id")
        or body.get("room", {}).get("sid")
        or body.get("room_name")
        or body.get("room", {}).get("name")
    )
    raw_summary = body.get("summary") or ""
    results = body.get("results") or {}
    transcript = body.get("transcript") or body.get("transcription") or raw_summary

    def process_and_save():
        # Match target customer phone number
        target_phone = None

        # 1. Check room_mappings in Firebase
        if room_id:
            try:
                r_map = requests.get(f"{FIREBASE_DATABASE_URL}/room_mappings/{sanitize_key(room_id)}.json", timeout=3)
                if r_map.status_code == 200 and r_map.json():
                    target_phone = r_map.json()
            except Exception:
                pass

        # 2. Check latest_active_call
        if not target_phone:
            try:
                r_active = requests.get(f"{FIREBASE_DATABASE_URL}/latest_active_call.json", timeout=3)
                if r_active.status_code == 200 and r_active.json():
                    act = r_active.json()
                    target_phone = act.get("phone")
            except Exception:
                pass

        # 3. Check inside results / metadata
        if not target_phone:
            target_phone = (
                results.get("phone_number")
                or body.get("metadata", {}).get("phone_number")
                or body.get("phone_number")
            )

        summary_text = raw_summary or transcript
        if not summary_text or summary_text == "{}":
            summary_text = "Call was completed with customer."

        update_data = {
            "status": "Completed",
            "ai_summary": str(summary_text),
            "transcript": str(transcript or summary_text),
            "sentiment": "Neutral",
            "completed_at": {".sv": "timestamp"},
        }

        if target_phone:
            clean_target = sanitize_key(target_phone)
            logger.info(f"Webhook updating customer {clean_target} with LiveKit summary")
            requests.patch(f"{FIREBASE_DATABASE_URL}/calls/{clean_target}.json", json=update_data, timeout=5)
            # Remove any stray RM_ key in Firebase
            if room_id and room_id.startswith("RM_"):
                requests.delete(f"{FIREBASE_DATABASE_URL}/calls/{sanitize_key(room_id)}.json", timeout=3)
        else:
            logger.warning(f"Could not find matching phone number for summary room {room_id}")

    bg_tasks.add_task(process_and_save)
    return {"status": "ok", "message": "Summary received"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
