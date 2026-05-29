"""FastAPI entry point: serves the static client and the /ws/audio WebSocket.

WS protocol (client -> server):
  * binary frames  : raw Int16 little-endian PCM, mono, 16 kHz
  * JSON {"type":"start"}  - reset segmenter / interrupt any in-flight turn
  * JSON {"type":"stop"}   - flush the current utterance (best effort)

WS protocol (server -> client):
  * binary frames  : MP3 audio chunks from edge-tts, in order
  * JSON {"type":"partial_transcript", "text": ..., "asr_ms": ...}
  * JSON {"type":"translation",       "english": ..., "hindi": ..., "translate_ms": ...}
  * JSON {"type":"latency",           "asr_ms": ..., "translate_ms": ..., "tts_first_chunk_ms": ..., "total_ms": ...}
  * JSON {"type":"stop_playback",     "reason": ...}
  * JSON {"type":"error",             "stage": ..., "message": ...}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import configure_logging, settings
from .session import Session
from .vad import SpeechStart, Utterance, UtteranceSegmenter

configure_logging()
logger = logging.getLogger(__name__)

if not settings.openai_api_key:
    logger.warning(
        "OPENAI_API_KEY is not set; transcription and translation will fail. "
        "Copy .env.example to .env and add your key."
    )

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="EN->HI Voice Translator")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.websocket("/ws/audio")
async def ws_audio(websocket: WebSocket) -> None:
    await websocket.accept()
    logger.info("WS connected")

    session = Session(websocket, settings)
    await session.start()

    segmenter = UtteranceSegmenter(
        sample_rate=settings.sample_rate,
        aggressiveness=settings.vad_aggressiveness,
        silence_ms=settings.vad_silence_ms,
        min_speech_ms=settings.vad_min_speech_ms,
    )

    await session.send_event(
        {
            "type": "ready",
            "sample_rate": settings.sample_rate,
            "voice": settings.tts_voice,
        }
    )

    try:
        while True:
            msg = await websocket.receive()
            if msg["type"] == "websocket.disconnect":
                break

            if "bytes" in msg and msg["bytes"] is not None:
                for event in segmenter.feed(msg["bytes"]):
                    if isinstance(event, SpeechStart):
                        if _session_is_active(session):
                            await session.interrupt(reason="speech_start")
                    elif isinstance(event, Utterance):
                        utterance = session.new_utterance(event.pcm, event.duration_ms)
                        await session.offer(session.asr_queue, utterance)
                        logger.info(
                            "Enqueued utterance: %d ms (gen=%d)",
                            event.duration_ms,
                            utterance.gen,
                        )

            elif "text" in msg and msg["text"] is not None:
                try:
                    payload = json.loads(msg["text"])
                except json.JSONDecodeError:
                    logger.warning("Ignoring non-JSON text frame")
                    continue
                kind = payload.get("type")
                if kind == "start":
                    segmenter.reset()
                    await session.interrupt(reason="client_start")
                elif kind == "stop":
                    segmenter.reset()
                else:
                    logger.debug("Unknown control message: %r", payload)

    except WebSocketDisconnect:
        logger.info("WS disconnected by client")
    except Exception:
        logger.exception("WS handler crashed")
    finally:
        await session.close()
        logger.info("WS session closed")


def _session_is_active(session: Session) -> bool:
    return (
        not session.asr_queue.empty()
        or not session.translation_queue.empty()
        or not session.tts_queue.empty()
        or not session.playback_queue.empty()
    )
