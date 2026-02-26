"""
RAG Service – Retrieval-Augmented Generation over Notes
────────────────────────────────────────────────────────
Uses LangChain to chunk text, ChromaDB's built-in embeddings for
vector storage, and OpenRouter/Gemini/OpenAI for answering questions.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# Load .env explicitly for any isolated service execution
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

# ── Globals (lazy init) ───────────────────────────────────────────────────────

_chroma_client: Optional[chromadb.ClientAPI] = None
_collection: Optional[chromadb.Collection] = None

COLLECTION_NAME = "omnidraft_notes"


def _get_chroma():
    """Lazy-init ChromaDB persistent client with built-in embeddings."""
    global _chroma_client, _collection
    if _chroma_client is None:
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        _chroma_client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        # Use ChromaDB's default embedding function (all-MiniLM-L6-v2)
        # This runs locally — no API key needed for embeddings
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB initialised at %s (using built-in embeddings)", persist_dir)
    return _collection


def _get_llm():
    """Get the chat LLM for the configured provider."""
    provider = os.getenv("LLM_PROVIDER", "openrouter").lower()

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENROUTER_CHAT_MODEL", "google/gemini-2.0-flash-001"),
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            temperature=0.3,
            default_headers={
                "HTTP-Referer": "https://omnidraft.app",
                "X-Title": "OmniDraft",
            },
        )
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.3,
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.3,
        )


# ── Ingestion ──────────────────────────────────────────────────────────────────

async def ingest_note(note_id: str, text: str, filename: str) -> int:
    """
    Chunk the transcribed text and store in ChromaDB.
    Uses ChromaDB's built-in embedding function (no external API needed).

    Returns the number of chunks created.
    """
    collection = _get_chroma()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n## ", "\n### ", "\n- ", "\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_text(text)
    if not chunks:
        logger.warning("No chunks produced for note %s", note_id)
        return 0

    # Store in ChromaDB — it handles embedding automatically
    ids = [f"{note_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {"note_id": note_id, "filename": filename, "chunk_index": i}
        for i in range(len(chunks))
    ]

    collection.upsert(
        ids=ids,
        documents=chunks,
        metadatas=metadatas,
    )

    logger.info("Ingested %d chunks for note %s", len(chunks), note_id)
    return len(chunks)


# ── Querying / Chat ───────────────────────────────────────────────────────────

async def chat_with_notes(
    query: str,
    note_id: Optional[str] = None,
    n_results: int = 5,
) -> tuple[str, list[str]]:
    """
    Answer a user question using RAG over their stored notes.

    Returns (answer, list_of_source_note_ids).
    """
    collection = _get_chroma()
    provider = os.getenv("LLM_PROVIDER", "local").lower()

    # Build filter
    where_filter = {"note_id": note_id} if note_id else None

    # ChromaDB handles embedding the query automatically
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter,
        include=["documents", "metadatas"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not documents:
        return "I couldn't find any relevant information in your notes. Try uploading more notes or rephrasing your question.", []

    # Build context
    context = "\n\n---\n\n".join(documents)
    source_ids = list({m.get("note_id", "") for m in metadatas if m})

    chat_prompt = """You are a professional 2026-grade AI Analyst for OmniDraft. Your objective is to answer the user's question with surgical precision using the provided context.

**Operational Guidelines:**
1. **Primary Source:** Rely on the "Context from notes" below. If the answer is found there, cite it effectively.
2. **Supplemental Intelligence:** If the notes are insufficient, you MAY use your internal pre-trained knowledge to provide a "Contextual Extension". Clearly distinguish this by starting the extension with "[External Knowledge Integration]".
3. **Multilingual Mode:** Always mirror the user's language. If they ask in Telugu, respond in Telugu.
4. **Formatting:** Use rich Markdown (bolding, lists, tables) for maximum clarity and visual impact.

**Context from notes:**
{context}

**User's Question:** {query}

**Answer:**""".replace("{context}", context).replace("{query}", query)

    # ── Cloud mode: use LangChain LLM to generate an answer ──
    llm = _get_llm()

    response = llm.invoke(chat_prompt)
    answer = response.content if hasattr(response, "content") else str(response)

    return answer.strip(), source_ids


async def get_chat_suggestions(note_id: Optional[str] = None) -> list[str]:
    """
    Generate 3 smart, context-aware questions for the user to ask.
    If note_id is provided, suggestions are specific to that note.
    Otherwise, they are general/global.
    """
    collection = _get_chroma()
    
    # Get a bit of context to seed the suggestions
    where_filter = {"note_id": note_id} if note_id else None
    
    # We just want a sample of the notes
    results = collection.get(
        where=where_filter,
        limit=5,
        include=["documents"]
    )
    
    documents = results.get("documents", [])
    context_snippet = "\n".join(documents)[:2000] if documents else "No notes uploaded yet."

    suggestion_prompt = """You are a creative brainstorming assistant for OmniDraft. 
Based on the following context from the user's notes, suggest 3 short, intriguing, and highly useful questions they might want to ask you about these notes.

**Rules:**
1. Return ONLY a JSON list of strings (e.g., ["Question 1?", "Question 2?", "Question 3?"]).
2. No commentary, no numbering, no preamble.
3. Keep questions under 12 words.
4. If there is no context, suggest general productivity/getting started questions.

**Context:**
{context_snippet}

**Suggestions (JSON list):**""".replace("{context_snippet}", context_snippet)

    llm = _get_llm()
    try:
        response = llm.invoke(suggestion_prompt)
        text = response.content if hasattr(response, "content") else str(response)
        
        # Simple JSON extraction
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            suggestions = json.loads(match.group(0))
            return suggestions[:3]
    except Exception as e:
        logger.error("Failed to generate suggestions: %s", e)
    
    # Fallback suggestions
    return [
        "What are the key takeaways from my notes?",
        "Can you summarize my most recent uploads?",
        "How can I better organize these documents?"
    ]


# ── Deletion ──────────────────────────────────────────────────────────────────

async def delete_note_embeddings(note_id: str) -> bool:
    """Remove all chunks for a given note from ChromaDB."""
    collection = _get_chroma()
    try:
        results = collection.get(
            where={"note_id": note_id},
            include=[],
        )
        if results["ids"]:
            collection.delete(ids=results["ids"])
            logger.info("Deleted %d chunks for note %s", len(results["ids"]), note_id)
        return True
    except Exception as e:
        logger.error("Failed to delete embeddings for %s: %s", note_id, e)
        return False
