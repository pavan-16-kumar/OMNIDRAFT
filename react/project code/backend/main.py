"""
OmniDraft – FastAPI Backend
════════════════════════════
High-performance async API for handwriting-to-text conversion,
RAG-powered chat, and multi-format document export.
"""

from __future__ import annotations

import os
import re
import shutil
import uuid
from pathlib import Path

# ── Environment ───────────────────────────────────────────────────────────────
from dotenv import load_dotenv
# Explicitly load .env before importing any local modules that may depend on it
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

import asyncio
import json
import logging
from datetime import datetime
from io import BytesIO
from typing import Optional

from pydantic import BaseModel
import edge_tts
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from fastapi.staticfiles import StaticFiles

from models.schemas import (
    ChatRequest,
    ChatResponse,
    ExportFormat,
    HealthResponse,
    NoteListItem,
    ProcessingStatus,
    TranscriptionResult,
)
from services.export_service import export_note
from services.image_processor import extract_images_from_pdf, preprocess_image
from services.ocr_agent import transcribe_image
from services.rag_service import chat_with_notes, delete_note_embeddings, ingest_note, get_chat_suggestions
from services.verifier_agent import verify_transcription

# ── Config ─────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-25s │ %(levelname)-7s │ %(message)s",
)
logger = logging.getLogger("omnidraft")

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", "20")) * 1024 * 1024  # bytes

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".heic", ".heif", ".pdf"}

# ── Simple JSON-based note store (Persists data) ─────────────────────────────

notes_db: dict[str, TranscriptionResult] = {}
DB_FILE = Path("./data/notes_db.json")

def load_notes_db():
    if DB_FILE.exists():
        try:
            with open(DB_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                for k, v in data.items():
                    notes_db[k] = TranscriptionResult(**v)
            logger.info("Loaded %d notes from %s", len(notes_db), DB_FILE)
        except Exception as e:
            logger.error("Failed to load notes_db.json: %s", e)

def save_notes_db():
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DB_FILE, "w", encoding="utf-8") as f:
        # Pydantic 2.x method .model_dump()
        json.dump({k: v.model_dump(mode="json") for k, v in notes_db.items()}, f, indent=2)

# Load existing notes on startup
load_notes_db()

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="OmniDraft API",
    description="Handwriting-to-Text + RAG Intelligence",
    version="1.0.0",
)

# CORS – allow all origins in dev for easier debugging
cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in cors_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

# Serve uploaded images statically
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
# Serve output text files statically
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check system health."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        llm_provider=os.getenv("LLM_PROVIDER", "openrouter"),
        chroma_status="ok",
    )


# ── Upload & Process ──────────────────────────────────────────────────────────

def _save_text_file(note_id: str, filename: str, text: str) -> str:
    """Save the extracted text to a .txt file in the output directory, ordered by page."""
    safe_name = re.sub(r"[^\w\-.]", "_", Path(filename).stem)
    output_path = OUTPUT_DIR / f"{safe_name}_{note_id[:8]}.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"{'=' * 70}\n")
        f.write(f"  OmniDraft — Extracted Text\n")
        f.write(f"  Source: {filename}\n")
        f.write(f"  Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
        f.write(f"  Note ID: {note_id}\n")
        f.write(f"{'=' * 70}\n\n")
        f.write(text)
        f.write(f"\n\n{'=' * 70}\n")
        f.write(f"  End of extracted text\n")
        f.write(f"{'=' * 70}\n")

    logger.info("📄 Saved extracted text to %s", output_path)
    return str(output_path)


@app.post("/upload", response_model=TranscriptionResult, tags=["Notes"])
async def upload_and_process(file: UploadFile = File(...)):
    """
    Upload a handwritten note (image or PDF) and process it through
    the multi-agent OCR pipeline.
    """
    # Validate file
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported file type: {ext}. Allowed: {ALLOWED_EXTENSIONS}",
        )

    # Read file
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(413, f"File too large. Max: {MAX_FILE_SIZE // (1024*1024)}MB")

    note_id = str(uuid.uuid4())
    original_filename = file.filename

    # Save original file
    save_path = UPLOAD_DIR / f"{note_id}{ext}"
    with open(save_path, "wb") as f:
        f.write(content)

    # Create initial note record
    note = TranscriptionResult(
        note_id=note_id,
        filename=original_filename,
        status=ProcessingStatus.PROCESSING,
        image_path=f"/uploads/{note_id}{ext}",
    )
    notes_db[note_id] = note
    save_notes_db()

    try:
        # ── Handle PDF vs Image ──
        if ext == ".pdf":
            page_images = await asyncio.to_thread(extract_images_from_pdf, save_path)
            if not page_images:
                raise HTTPException(400, "Could not extract any pages from PDF")

            total_pages = len(page_images)
            logger.info("📚 Starting PDF processing: %d pages", total_pages)

            # Process pages concurrently (higher limit for speed on large PDFs like 30+ pages)
            max_concurrency = int(os.getenv("MAX_CONCURRENT_PAGES", "15"))
            sem = asyncio.Semaphore(max_concurrency)

            async def process_page(i: int, page_bytes: bytes):
                async with sem:
                    logger.info("Processing PDF page %d/%d", i + 1, total_pages)
                    _, processed_bytes = await asyncio.to_thread(preprocess_image, page_bytes)

                    # Save processed page image
                    page_img_path = UPLOAD_DIR / f"{note_id}_page_{i}.jpg"
                    with open(page_img_path, "wb") as pf:
                        pf.write(processed_bytes)

                    # OCR Transcription only (skip verification for speed)
                    raw_md = await transcribe_image(processed_bytes)
                    return i, raw_md

            note.status = ProcessingStatus.VERIFYING
            tasks = [process_page(i, pb) for i, pb in enumerate(page_images)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle any failed pages gracefully
            all_raw_md = []
            failed_pages = []
            for result in results:
                if isinstance(result, Exception):
                    failed_pages.append(str(result))
                    continue
                i, raw_md = result
                all_raw_md.append((i, raw_md))

            all_raw_md.sort(key=lambda x: x[0])

            ordered_text_parts = []
            for i, raw_md in all_raw_md:
                ordered_text_parts.append(f"## Page {i + 1}\n\n{raw_md}")

            combined_text = "\n\n---\n\n".join(ordered_text_parts)
            note.raw_markdown = combined_text
            note.verified_markdown = combined_text
            note.flags = []
            note.confidence_score = 0.95

            if failed_pages:
                logger.warning("⚠️ %d pages failed: %s", len(failed_pages), failed_pages[:3])

        else:
            # Single image
            _, processed_bytes = await asyncio.to_thread(preprocess_image, content)

            # Save processed image
            processed_path = UPLOAD_DIR / f"{note_id}_processed.jpg"
            with open(processed_path, "wb") as pf:
                pf.write(processed_bytes)

            # Step A: OCR Transcription
            logger.info("Step A: Transcribing image for note %s", note_id)
            raw_md = await transcribe_image(processed_bytes)
            note.raw_markdown = raw_md

            # Step B: Verification (only for single images — fast enough)
            logger.info("Step B: Verifying transcription for note %s", note_id)
            note.status = ProcessingStatus.VERIFYING
            verified_md, confidence, flags = await verify_transcription(
                processed_bytes, raw_md
            )
            note.verified_markdown = verified_md
            note.confidence_score = confidence
            note.flags = flags

        # ── Save extracted text to file ──
        final_text = note.verified_markdown or note.raw_markdown
        if final_text:
            _save_text_file(note_id, original_filename, final_text)

        # ── RAG Ingestion (run in background, don't block response) ──
        if final_text:
            asyncio.create_task(_background_ingest(note_id, final_text, original_filename))

        note.status = ProcessingStatus.COMPLETED
        note.created_at = datetime.utcnow()
        logger.info(
            "✅ Note %s processed: confidence=%.2f, flags=%d",
            note_id,
            note.confidence_score,
            len(note.flags),
        )

    except HTTPException:
        raise
    except Exception as e:
        note.status = ProcessingStatus.FAILED
        save_notes_db()
        logger.error("❌ Processing failed for %s: %s", note_id, e, exc_info=True)
        
        # Provide user-friendly error messages for common API issues
        err_str = str(e).lower()
        if "api key" in err_str or "leaked" in err_str or "permission" in err_str:
            raise HTTPException(500, "API key error: Your API key is invalid or has been revoked. Please update your API key in the .env file.")
        elif "401" in err_str or "unauthorized" in err_str or "authentication" in err_str:
            raise HTTPException(500, "Authentication failed: Your API key is invalid. Please check your .env configuration.")
        elif "quota" in err_str or "rate" in err_str or "limit" in err_str:
            raise HTTPException(429, "Rate limit exceeded. Please wait a moment and try again.")
        else:
            raise HTTPException(500, f"Processing failed: {str(e)}")

    save_notes_db()
    return note


async def _background_ingest(note_id: str, text: str, filename: str):
    """Ingest into RAG in the background so upload response returns immediately."""
    try:
        await ingest_note(note_id, text, filename)
        logger.info("🔍 RAG ingestion complete for %s", note_id)
    except Exception as e:
        logger.error("RAG ingestion failed for %s: %s", note_id, e)


# ── List Notes ─────────────────────────────────────────────────────────────────

@app.get("/notes", response_model=list[NoteListItem], tags=["Notes"])
async def list_notes():
    """List all processed notes."""
    items = []
    for nid, note in sorted(notes_db.items(), key=lambda x: x[1].created_at, reverse=True):
        preview_text = (note.verified_markdown or note.raw_markdown)[:200]
        items.append(NoteListItem(
            note_id=nid,
            filename=note.filename,
            status=note.status,
            confidence_score=note.confidence_score,
            created_at=note.created_at,
            preview=preview_text,
        ))
    return items


# ── Get Single Note ───────────────────────────────────────────────────────────

@app.get("/notes/{note_id}", response_model=TranscriptionResult, tags=["Notes"])
async def get_note(note_id: str):
    """Get the full details of a single note."""
    note = notes_db.get(note_id)
    if not note:
        raise HTTPException(404, f"Note not found: {note_id}")
    return note


# ── Delete Note ───────────────────────────────────────────────────────────────

@app.delete("/notes/{note_id}", tags=["Notes"])
async def delete_note(note_id: str):
    """Delete a note and its associated data."""
    note = notes_db.pop(note_id, None)
    if not note:
        raise HTTPException(404, f"Note not found: {note_id}")

    # Delete from ChromaDB
    await delete_note_embeddings(note_id)

    # Delete uploaded files
    for f in UPLOAD_DIR.glob(f"{note_id}*"):
        f.unlink(missing_ok=True)

    save_notes_db()
    return {"status": "deleted", "note_id": note_id}


# ── Update Note (User corrections) ────────────────────────────────────────────

@app.patch("/notes/{note_id}", response_model=TranscriptionResult, tags=["Notes"])
async def update_note_text(note_id: str, updated_text: str = Query(...)):
    """Update the verified text after user corrections."""
    note = notes_db.get(note_id)
    if not note:
        raise HTTPException(404, f"Note not found: {note_id}")

    note.verified_markdown = updated_text
    note.confidence_score = 1.0  # User-verified
    note.flags = []

    # Re-ingest with updated text
    await delete_note_embeddings(note_id)
    await ingest_note(note_id, updated_text, note.filename)

    save_notes_db()
    return note


# ── Export / Download ─────────────────────────────────────────────────────────

@app.get("/download/{note_id}", tags=["Export"])
async def download_note(
    note_id: str,
    format: str = Query("pdf", description="Export format: pdf, docx, md, txt"),
):
    """Export and download a note in the specified format."""
    note = notes_db.get(note_id)
    if not note:
        raise HTTPException(404, f"Note not found: {note_id}")

    text = note.verified_markdown or note.raw_markdown
    if not text:
        raise HTTPException(400, "Note has no transcribed text to export")

    title = Path(note.filename).stem or "OmniDraft Note"

    # Normalize format string
    fmt = format.lower().strip().strip(".")

    try:
        file_bytes, content_type, ext = export_note(text, fmt, title)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error("Export failed: %s", e)
        raise HTTPException(500, f"Export failed: {str(e)}")

    safe_name = re.sub(r"[^\w\-.]", "_", title)

    return Response(
        content=file_bytes,
        media_type=content_type,
        headers={
            "Content-Disposition": f'attachment; filename="{safe_name}{ext}"',
        },
    )


# ── Chat (RAG) ────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse, tags=["RAG"])
async def chat(request: ChatRequest):
    """Chat with your notes using RAG-powered search."""
    try:
        answer, sources = await chat_with_notes(
            query=request.query,
            note_id=request.note_id,
        )
        return ChatResponse(answer=answer, sources=sources)
    except Exception as e:
        logger.error("Chat failed: %s", e, exc_info=True)
        raise HTTPException(500, f"Chat failed: {str(e)}")


@app.get("/chat/suggestions", tags=["RAG"])
async def chat_suggestions(note_id: Optional[str] = Query(None)):
    """Get smart suggestions for questions to ask the chat assistant."""
    try:
        suggestions = await get_chat_suggestions(note_id)
        return {"suggestions": suggestions}
    except Exception as e:
        logger.error("Suggestions failed: %s", e)
        return {"suggestions": []}


# ── Text-to-Speech (edge-tts — Microsoft Neural, FREE & UNLIMITED) ────────────

# Best neural voice for every language (Microsoft Edge TTS voices)
_TTS_VOICES: dict[str, str] = {
    # Indian languages
    "te-IN": "te-IN-ShrutiNeural",   # Telugu  — Female, very clear
    "ta-IN": "ta-IN-PallaviNeural",  # Tamil   — Female
    "hi-IN": "hi-IN-SwaraNeural",    # Hindi   — Female
    "kn-IN": "kn-IN-SapnaNeural",    # Kannada — Female
    "ml-IN": "ml-IN-SobhanaNeural",  # Malayalam — Female
    "bn-IN": "bn-IN-TanishaaNeural", # Bengali — Female
    "gu-IN": "gu-IN-DhwaniNeural",   # Gujarati — Female
    "mr-IN": "mr-IN-AarohiNeural",   # Marathi — Female
    "or-IN": "or-IN-SubhasiniNeural",# Odia — Female
    "pa-IN": "pa-IN-OjasNeural",     # Punjabi — Male (only option)
    "ur-PK": "ur-PK-AsadNeural",     # Urdu
    "si-LK": "si-LK-ThiliniNeural",  # Sinhala
    # English
    "en-US": "en-US-AriaNeural",     # Best US English — very natural
    "en-IN": "en-IN-NeerjaNeural",   # Indian English
    "en-GB": "en-GB-SoniaNeural",    # British English
    # East Asian
    "zh-CN": "zh-CN-XiaoxiaoNeural",
    "zh-TW": "zh-TW-HsiaoChenNeural",
    "ja-JP": "ja-JP-NanamiNeural",
    "ko-KR": "ko-KR-SunHiNeural",
    # Middle East / Africa
    "ar-SA": "ar-SA-ZariyahNeural",
    "he-IL": "he-IL-HilaNeural",
    # Southeast Asia
    "th-TH": "th-TH-PremwadeeNeural",
    "id-ID": "id-ID-GadisNeural",
    "ms-MY": "ms-MY-YasminNeural",
    "vi-VN": "vi-VN-HoaiMyNeural",
    # European
    "ru-RU": "ru-RU-SvetlanaNeural",
    "fr-FR": "fr-FR-DeniseNeural",
    "de-DE": "de-DE-KatjaNeural",
    "es-ES": "es-ES-ElviraNeural",
    "pt-BR": "pt-BR-FranciscaNeural",
    "it-IT": "it-IT-ElsaNeural",
    "nl-NL": "nl-NL-ColetteNeural",
    "pl-PL": "pl-PL-ZofiaNeural",
    "el-GR": "el-GR-AthinaNeural",
    "tr-TR": "tr-TR-EmelNeural",
    "uk-UA": "uk-UA-OstapNeural",
    "ka-GE": "ka-GE-EkaNeural",
}


def _rate_to_edge(rate: float) -> str:
    """Convert 0.5–2.0 rate float to edge-tts '+/-N%' format."""
    pct = int(round((rate - 1.0) * 100))
    return f"+{pct}%" if pct >= 0 else f"{pct}%"


class TTSRequest(BaseModel):
    text: str
    lang: str = "en-US"   # BCP-47 language code
    rate: float = 0.92    # 0.5 – 2.0; 0.92 = clearest
    voice: Optional[str] = None  # override voice name


@app.post("/tts", tags=["TTS"])
async def text_to_speech(req: TTSRequest):
    """
    Convert text to speech using Microsoft Edge TTS neural voices.
    Returns MP3 audio bytes. FREE and UNLIMITED.
    """
    raw = req.text.strip()
    if not raw:
        raise HTTPException(400, "Text is empty")

    # Clean markdown for natural spoken reading
    clean = re.sub(r"#{1,6}\s+", "", raw)
    clean = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", clean)
    clean = re.sub(r"`{1,3}[^`]*`{1,3}", "", clean)
    clean = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", clean)
    clean = re.sub(r"---+", "", clean)
    clean = re.sub(r"\n{2,}", ". ", clean)
    clean = re.sub(r"\n", " ", clean)
    clean = clean.strip()[:12000]  # cap at 12k chars to avoid timeout

    voice = req.voice or _TTS_VOICES.get(req.lang, _TTS_VOICES["en-US"])
    rate_str = _rate_to_edge(req.rate)

    logger.info("🔊 TTS: lang=%s voice=%s rate=%s len=%d", req.lang, voice, rate_str, len(clean))

    try:
        communicate = edge_tts.Communicate(clean, voice, rate=rate_str)
        buf = BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        buf.seek(0)
        content = buf.read()

        if not content:
            raise HTTPException(500, "edge-tts returned no audio — check internet connection")

        return Response(
            content=content,
            media_type="audio/mpeg",
            headers={"Cache-Control": "no-cache", "Content-Disposition": "inline"},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("TTS error: %s", e)
        raise HTTPException(500, f"TTS failed: {e}")


# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=True)
