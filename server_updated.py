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
# Keep API credentials in Render Environment Variables; never hard-code secrets.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
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
    """Summarize the real call transcript using OpenRouter."""
    transcript_text = (transcript_text or "").strip()
    if not transcript_text:
        return {
            "call_status": "No Speech / Unknown",
            "ai_summary": "No transcript available.",
            "customer_sentiment": "Neutral",
        }

    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY is not configured in Render environment variables.")
        return {
            "call_status": "Completed",
            "ai_summary": transcript_text[:350] + ("..." if len(transcript_text) > 350 else ""),
            "customer_sentiment": "Neutral",
        }

    prompt = f"""You are an expert AI call analyst. Analyze ONLY the actual phone-call transcript below.
Do not invent facts, commitments, customer details, or outcomes that are not explicitly present.
Return a concise, accurate summary of the conversation and the customer's response.

TRANSCRIPT:
\"\"\"
{transcript_text}
\"\"\"

Respond ONLY with valid JSON using exactly this structure:
{{
  "call_status": "Answered" or "Rejected / Busy" or "Wrong Number" or "Voicemail" or "Completed",
  "ai_summary": "2-4 clear sentences describing the actual discussion, customer needs/objections, and agreed next step if any",
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
            "temperature": 0.1,
        }
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=body,
            timeout=30,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        cleaned = re.sub(r"^```(?:json)?\s*|```\s*$", "", content, flags=re.MULTILINE).strip()
        result = json.loads(cleaned)
        return {
            "call_status": result.get("call_status", "Completed"),
            "ai_summary": result.get("ai_summary", "Summary could not be generated."),
            "customer_sentiment": result.get("customer_sentiment", "Neutral"),
        }
    except Exception as e:
        logger.error(f"AI summarization error: {e}")

    # Fallback uses a short excerpt of the real transcript, never fabricated call details.
    return {
        "call_status": "Completed",
        "ai_summary": transcript_text[:350] + ("..." if len(transcript_text) > 350 else ""),
        "customer_sentiment": "Neutral",
    }


def _transcript_item_to_text(item: Any) -> str:
    """Convert common LiveKit transcript/conversation item shapes into readable text."""
    if isinstance(item, str):
        return item.strip()

    if isinstance(item, list):
        parts = [_transcript_item_to_text(part) for part in item]
        return "\n".join(part for part in parts if part)

    if not isinstance(item, dict):
        return ""

    text_value = (
        item.get("text")
        or item.get("transcript")
        or item.get("content")
        or item.get("utterance")
        or item.get("text_content")
        or item.get("textContent")
    )

    if not text_value and isinstance(item.get("message"), (str, dict, list)):
        text_value = item.get("message")

    if isinstance(text_value, (dict, list)):
        text_value = _transcript_item_to_text(text_value)

    if text_value:
        text_value = str(text_value).strip()
        speaker = (
            item.get("speaker")
            or item.get("role")
            or item.get("participant_identity")
            or item.get("identity")
            or item.get("name")
        )
        return f"{speaker}: {text_value}" if speaker else text_value

    for key in ("segments", "utterances", "messages", "items", "entries", "conversation", "turns"):
        nested = item.get(key)
        if isinstance(nested, (dict, list)):
            rendered = _transcript_item_to_text(nested)
            if rendered:
                return rendered

    return ""


def extract_transcript(payload: Any) -> str:
    """Extract an actual transcript field; a generic summary is never treated as a transcript."""
    transcript_keys = (
        "transcript",
        "transcription",
        "transcript_text",
        "full_transcript",
        "conversation_transcript",
        "chat_history",
    )

    def search(node: Any) -> str:
        if isinstance(node, dict):
            for key in transcript_keys:
                if key in node:
                    rendered = _transcript_item_to_text(node.get(key))
                    if rendered:
                        return rendered
            for value in node.values():
                if isinstance(value, (dict, list)):
                    rendered = search(value)
                    if rendered:
                        return rendered
        elif isinstance(node, list):
            for value in node:
                rendered = search(value)
                if rendered:
                    return rendered
        return ""

    return search(payload).strip()


def extract_webhook_phone(payload: dict) -> Optional[str]:
    """Find the phone number in common webhook metadata locations."""
    metadata = payload.get("metadata") or {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except Exception:
            metadata = {}

    room = payload.get("room") or {}
    if not isinstance(room, dict):
        room = {}

    results = payload.get("results") or {}
    if not isinstance(results, dict):
        results = {}

    candidates = [
        results.get("phone_number"),
        metadata.get("phone_number") if isinstance(metadata, dict) else None,
        room.get("metadata", {}).get("phone_number")
        if isinstance(room.get("metadata"), dict) else None,
        payload.get("phone_number"),
    ]
    for candidate in candidates:
        if candidate:
            return str(candidate).strip()
    return None


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


class TranscriptRequest(BaseModel):
    phone_number: str
    transcript: str
    room_name: Optional[str] = None
    source: Optional[str] = "livekit_agent_session"


@app.post("/api/transcript")
async def receive_call_transcript(req: TranscriptRequest):
    """Store the exact transcript collected by the LiveKit AgentSession."""
    phone = req.phone_number.strip()
    transcript_text = (req.transcript or "").strip()

    if not re.fullmatch(r"\+[1-9]\d{6,14}", phone):
        raise HTTPException(status_code=400, detail="Invalid E.164 phone format.")
    if not transcript_text:
        raise HTTPException(status_code=400, detail="Transcript cannot be empty.")

    update_data = {
        "phone": phone,
        "transcript": transcript_text,
        "transcript_source": req.source or "livekit_agent_session",
        "transcript_updated_at": {".sv": "timestamp"},
    }
    if req.room_name:
        update_data["room_name"] = req.room_name

    update_firebase_call(phone, update_data)
    return {"success": True, "message": "Transcript saved to Firebase."}


@app.post("/api/summarize_call")
async def manual_summarize_call(req: SummarizeRequest):
    """Generate or refresh an AI summary from the saved, exact transcript."""
    phone = req.phone_number.strip()
    clean_phone = sanitize_key(phone)

    transcript_text = (req.transcript or "").strip()
    if not transcript_text:
        try:
            r = requests.get(f"{FIREBASE_DATABASE_URL}/calls/{clean_phone}.json", timeout=10)
            r.raise_for_status()
            call_data = r.json() or {}
            transcript_text = (call_data.get("transcript") or "").strip()
        except Exception as e:
            logger.error(f"Could not load call transcript from Firebase for {phone}: {e}")
            raise HTTPException(status_code=502, detail="Could not read the call record from Firebase.")

    if not transcript_text:
        raise HTTPException(
            status_code=409,
            detail="Exact transcript is not available for this call yet. Wait for LiveKit transcript sync, then try again.",
        )

    analysis = generate_ai_summary(transcript_text)
    update_data = {
        "status": analysis.get("call_status", "Completed"),
        "ai_summary": analysis.get("ai_summary", "Summary could not be generated."),
        "sentiment": analysis.get("customer_sentiment", "Neutral"),
        "completed_at": {".sv": "timestamp"},
    }

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
                wait_until_answered=False,  # Return immediately so web request doesn't timeout
            )
        )

        update_firebase_call(phone, {
            "status": "Ringing...",
            "sip_participant_id": participant.participant_id,
            "dispatch_id": dispatch.id,
        })

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
        # Check every 3 seconds for up to 5 minutes
        while (asyncio.get_event_loop().time() - start_time) < max_duration:
            await asyncio.sleep(3)
            try:
                # Check room participants
                res = await lk.room.list_participants(api.ListParticipantsRequest(room=room_name))
                identities = [p.identity for p in res.participants]

                # If phone-user is in room, call is answered!
                if "phone-user" in identities:
                    if not call_answered:
                        call_answered = True
                        logger.info(f"Call answered by {phone} in {room_name}")
                        update_firebase_call(phone, {"status": "In Call / Speaking"})

                # If call was answered and now phone-user left (or room empty), call ended
                elif call_answered and "phone-user" not in identities:
                    logger.info(f"Call ended for {phone} in {room_name}")
                    break

            except api.TwirpError:
                # Room does not exist anymore (e.g. deleted by agent when ending call)
                if call_answered:
                    logger.info(f"Room {room_name} closed after call answered")
                    break
                else:
                    # Ringing timed out or call rejected
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

        # Once call has ended, wait 6 seconds to see if webhook delivers transcript
        await asyncio.sleep(6)

        # Check whether the webhook or AgentSession transcript sync already saved the call.
        clean_phone = sanitize_key(phone)
        try:
            r = requests.get(f"{FIREBASE_DATABASE_URL}/calls/{clean_phone}.json", timeout=10)
            r.raise_for_status()
            existing = r.json() or {}

            if existing.get("transcript"):
                if not existing.get("ai_summary"):
                    analysis = generate_ai_summary(existing["transcript"])
                    update_firebase_call(phone, {
                        "status": analysis.get("call_status", "Completed"),
                        "ai_summary": analysis.get("ai_summary", "Summary could not be generated."),
                        "sentiment": analysis.get("customer_sentiment", "Neutral"),
                        "completed_at": {".sv": "timestamp"},
                    })
                return

            # Do not fabricate a summary when no transcript was received.
            update_firebase_call(phone, {
                "status": "Completed - Transcript Pending",
                "completed_at": {".sv": "timestamp"},
            })
        except Exception as e:
            logger.warning(f"Could not verify saved transcript for {phone}: {e}")

    except Exception as e:
        logger.error(f"Error in monitor_call_lifecycle for {phone}: {e}")
    finally:
        await lk.aclose()


@app.post("/api/webhook/summary")
async def call_summary_webhook(request: Request, bg_tasks: BackgroundTasks):
    """
    Endpoint configured in LiveKit Cloud Console:
    'Call ending' -> 'Summary endpoint URL'.

    Saves an exact transcript when LiveKit includes one in the payload.
    """
    try:
        body = await request.json()
        if not isinstance(body, dict):
            body = {}
        logger.info("Received LiveKit call summary webhook.")
    except Exception:
        body = {}

    room = body.get("room") or {}
    if not isinstance(room, dict):
        room = {}

    room_name = (
        body.get("room_name")
        or room.get("name")
        or body.get("room_id")
        or room.get("sid")
    )
    results = body.get("results") or {}
    if not isinstance(results, dict):
        results = {}

    transcript_text = extract_transcript(body)
    raw_summary = body.get("summary") or results.get("summary") or ""
    phone = extract_webhook_phone(body)

    def process_and_save():
        target_key = None

        try:
            resp = requests.get(f"{FIREBASE_DATABASE_URL}/calls.json", timeout=10)
            resp.raise_for_status()
            all_calls = resp.json() or {}

            if isinstance(all_calls, dict):
                for db_key, call_info in all_calls.items():
                    if not isinstance(call_info, dict):
                        continue
                    if room_name and call_info.get("room_name") == room_name:
                        target_key = db_key
                        break

                if not target_key and phone:
                    target_key = sanitize_key(phone)
                    if target_key not in all_calls:
                        digits = re.sub(r"\D", "", phone)[-10:]
                        for db_key, call_info in all_calls.items():
                            stored_phone = re.sub(
                                r"\D", "", str((call_info or {}).get("phone", ""))
                            )[-10:]
                            if digits and (db_key.endswith(digits) or stored_phone == digits):
                                target_key = db_key
                                break
        except Exception as e:
            logger.error(f"Error searching Firebase for call: {e}")

        if not target_key:
            if phone:
                target_key = sanitize_key(phone)
            elif room_name and str(room_name).startswith("call_"):
                parts = str(room_name).split("_")
                if len(parts) >= 3:
                    target_key = "_".join(parts[1:-1])
            elif room_name:
                target_key = sanitize_key(str(room_name))

        if not target_key:
            logger.warning("Could not determine Firebase call record for LiveKit summary webhook.")
            return

        update_data = {
            "status": "Completed",
            "completed_at": {".sv": "timestamp"},
            "raw_results": results,
        }
        if room_name:
            update_data["room_name"] = room_name

        if transcript_text:
            update_data["transcript"] = transcript_text
            update_data["transcript_source"] = "livekit_summary_webhook"
            update_data["transcript_updated_at"] = {".sv": "timestamp"}

            analysis = generate_ai_summary(transcript_text)
            update_data.update({
                "status": analysis.get("call_status", "Completed"),
                "ai_summary": analysis.get("ai_summary", "Summary could not be generated."),
                "sentiment": analysis.get("customer_sentiment", "Neutral"),
            })
        elif raw_summary:
            # Keep LiveKit's summary, but never mislabel it as the exact transcript.
            update_data["ai_summary"] = str(raw_summary)
            update_data["sentiment"] = "Neutral"
            update_data["transcript_status"] = "not_in_webhook_payload"
        else:
            update_data["transcript_status"] = "not_in_webhook_payload"

        logger.info(f"Updating Firebase call record '{target_key}' from LiveKit webhook.")
        try:
            url = f"{FIREBASE_DATABASE_URL}/calls/{target_key}.json"
            r = requests.patch(url, json=update_data, timeout=10)
            if r.status_code >= 400:
                logger.error(f"Firebase webhook update failed: {r.status_code} {r.text}")
            else:
                logger.info(f"Firebase webhook update successful for '{target_key}'.")
        except Exception as e:
            logger.error(f"Failed to update Firebase key {target_key}: {e}")

    bg_tasks.add_task(process_and_save)
    return {"status": "ok", "message": "Summary webhook received"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)

