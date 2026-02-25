"""
Pydantic models / schemas for OmniDraft API.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────────────

class ExportFormat(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    MARKDOWN = "md"
    TXT = "txt"


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"


class LLMProvider(str, Enum):
    GEMINI = "gemini"
    OPENAI = "openai"


# ── Request Models ─────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User's question about their notes")
    note_id: Optional[str] = Field(None, description="Specific note to query against; if None search all notes")


class ExportRequest(BaseModel):
    note_id: str = Field(..., description="ID of the note to export")
    format: ExportFormat = Field(..., description="Desired export format")


# ── Response Models ────────────────────────────────────────────────────────────

class VerificationFlag(BaseModel):
    word: str = Field(..., description="The flagged word/phrase")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score 0-1")
    suggestion: Optional[str] = Field(None, description="Suggested correction")
    context: str = Field("", description="Surrounding text for context")


class TranscriptionResult(BaseModel):
    note_id: str
    filename: str
    raw_markdown: str = Field("", description="Raw Markdown from OCR")
    verified_markdown: str = Field("", description="Post-verification Markdown")
    confidence_score: float = Field(0.0, ge=0, le=1)
    flags: list[VerificationFlag] = Field(default_factory=list)
    status: ProcessingStatus = ProcessingStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    image_path: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)


class NoteListItem(BaseModel):
    note_id: str
    filename: str
    status: ProcessingStatus
    confidence_score: float
    created_at: datetime
    preview: str = Field("", description="First ~200 chars of the transcription")


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    llm_provider: str = ""
    chroma_status: str = "ok"
