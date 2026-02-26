"""
OCR Agent – Vision LLM + Local EasyOCR
──────────────────────────────────────────────────────────────
Step A of the Agentic Transcription pipeline.
Supports:
  - "local"      → EasyOCR (FREE, UNLIMITED, all Indian languages)
  - "openrouter" → OpenRouter Vision LLM
  - "gemini"     → Google Gemini Vision
  - "openai"     → OpenAI GPT-4o Vision
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Supported Indian languages for EasyOCR ─────────────────────────────────────
# Map of human-readable name → EasyOCR code
INDIAN_LANGUAGES = {
    "english": "en",
    "telugu": "te",
    "hindi": "hi",
    "tamil": "ta",
    "kannada": "kn",
    "bengali": "bn",
    "marathi": "mr",
    "gujarati": "gu",
    "malayalam": "ml",
    "punjabi": "pa",
    "urdu": "ur",
    "nepali": "ne",
    "assamese": "as",
}

# ── EasyOCR singleton (lazy-loaded, heavy on first init) ───────────────────────

_easyocr_reader = None


# ── EasyOCR language compatibility groups ──────────────────────────────────────
# EasyOCR requires that non-Latin scripts are only combined with English.
# Different scripts CANNOT be mixed in a single reader.
# So we create separate readers per script family.

SCRIPT_GROUPS = {
    # Devanagari-based scripts (can be combined together + English)
    "devanagari": ["hi", "mr", "ne"],
    # Each of these is its own script family (only + English)
    "telugu": ["te"],
    "tamil": ["ta"],
    "kannada": ["kn"],
    "bengali": ["bn", "as"],
    "gujarati": ["gu"],
    "malayalam": ["ml"],
    "punjabi": ["pa"],
    "urdu": ["ur"],
}

_easyocr_readers: dict[str, object] = {}
_readers_initialized = False


def _get_easyocr_readers() -> list:
    """
    Initialise one EasyOCR Reader per script family needed.
    Downloads models on first run (~100-300 MB per language pack).
    Returns a list of (reader, lang_list) tuples.
    """
    global _easyocr_readers, _readers_initialized
    if _readers_initialized:
        return list(_easyocr_readers.values())

    import easyocr

    # Read languages from env, default: English + Telugu
    lang_str = os.getenv("LOCAL_OCR_LANGUAGES", "en,te")
    requested_langs = [l.strip() for l in lang_str.split(",") if l.strip()]

    # Validate
    valid_codes = set(INDIAN_LANGUAGES.values())
    requested_langs = [l for l in requested_langs if l in valid_codes]
    if not requested_langs:
        requested_langs = ["en"]

    # Ensure English is always included
    if "en" not in requested_langs:
        requested_langs.append("en")

    use_gpu = os.getenv("LOCAL_OCR_GPU", "false").lower() == "true"

    # Group requested languages by script family
    groups_needed: dict[str, list[str]] = {}
    english_only = True

    for lang in requested_langs:
        if lang == "en":
            continue  # English is added to every group
        english_only = False

        # Find which group this language belongs to
        found = False
        for group_name, group_langs in SCRIPT_GROUPS.items():
            if lang in group_langs:
                if group_name not in groups_needed:
                    groups_needed[group_name] = []
                groups_needed[group_name].append(lang)
                found = True
                break
        if not found:
            logger.warning("Language '%s' not in any known script group, skipping", lang)

    # If only English was requested, create a single English reader
    if english_only or not groups_needed:
        logger.info("🔤 Initialising EasyOCR with: ['en']")
        reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
        _easyocr_readers["english"] = (reader, ["en"])
    else:
        # Create one reader per script group
        for group_name, langs in groups_needed.items():
            lang_list = langs + ["en"]  # Always include English
            logger.info("🔤 Initialising EasyOCR reader [%s] with: %s", group_name, lang_list)
            try:
                reader = easyocr.Reader(lang_list, gpu=use_gpu, verbose=False)
                _easyocr_readers[group_name] = (reader, lang_list)
                logger.info("✅ EasyOCR [%s] ready", group_name)
            except Exception as e:
                logger.error("❌ Failed to init EasyOCR [%s]: %s", group_name, e)

    if not _easyocr_readers:
        # Fallback: English only
        logger.warning("⚠️ No readers initialized, falling back to English only")
        reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
        _easyocr_readers["english"] = (reader, ["en"])

    _readers_initialized = True
    total_langs = set()
    for _, lang_list in _easyocr_readers.values():
        total_langs.update(lang_list)
    logger.info("✅ EasyOCR fully ready: %d readers, languages: %s", len(_easyocr_readers), sorted(total_langs))

    return list(_easyocr_readers.values())


# ── Smart Markdown formatter for raw OCR text ──────────────────────────────────

def _format_ocr_to_markdown(detections: list) -> str:
    """
    Convert EasyOCR detections into structured Markdown.

    Each detection is (bbox, text, confidence).
    We group by vertical position, detect structure and
    produce clean Markdown.
    """
    if not detections:
        return "*No text detected in image.*"

    # Sort by vertical position (top of bounding box), then horizontal
    def _sort_key(det):
        bbox = det[0]
        y_top = min(p[1] for p in bbox)
        x_left = min(p[0] for p in bbox)
        return (y_top, x_left)

    detections.sort(key=_sort_key)

    # ── Group detections into lines ──────────────────────────────
    lines: list[dict] = []       # {texts: [...], y: ..., avg_h: ..., confidences: [...]}
    current_line_y = None
    line_threshold = 15  # pixels vertical tolerance for same line

    for det in detections:
        bbox, text, conf = det
        text = text.strip()
        if not text:
            continue

        y_top = min(p[1] for p in bbox)
        y_bot = max(p[1] for p in bbox)
        x_left = min(p[0] for p in bbox)
        char_height = y_bot - y_top

        # Adaptive threshold based on character height
        adaptive_threshold = max(line_threshold, char_height * 0.5)

        if current_line_y is None or abs(y_top - current_line_y) > adaptive_threshold:
            # New line
            lines.append({
                "texts": [text],
                "y": y_top,
                "avg_h": char_height,
                "confidences": [conf],
                "x_left": x_left,
            })
            current_line_y = y_top
        else:
            # Same line – append
            lines[-1]["texts"].append(text)
            lines[-1]["confidences"].append(conf)

    if not lines:
        return "*No text detected in image.*"

    # ── Calculate average character height for heading detection ───
    all_heights = [l["avg_h"] for l in lines if l["avg_h"] > 0]
    median_height = sorted(all_heights)[len(all_heights) // 2] if all_heights else 20

    # ── Build Markdown ────────────────────────────────────────────
    md_parts: list[str] = []
    prev_was_blank = True  # to avoid double-spacing at start

    for i, line in enumerate(lines):
        full_text = " ".join(line["texts"]).strip()
        if not full_text:
            continue

        avg_conf = sum(line["confidences"]) / len(line["confidences"])
        h = line["avg_h"]
        x = line.get("x_left", 0)

        # Mark low-confidence words
        if avg_conf < 0.4:
            full_text = f"⚠️{full_text}⚠️"

        # ── Detect headings (significantly larger text) ───────
        if h > median_height * 1.6 and len(full_text.split()) <= 12:
            md_parts.append(f"\n# {full_text}\n")
            prev_was_blank = True
            continue
        elif h > median_height * 1.3 and len(full_text.split()) <= 15:
            md_parts.append(f"\n## {full_text}\n")
            prev_was_blank = True
            continue

        # ── Detect bullet points (indented or starts with -, *, •) ──
        bullet_match = re.match(r'^[\-\*\•\·]\s*(.*)', full_text)
        num_match = re.match(r'^(\d+)[\.\)]\s*(.*)', full_text)
        if bullet_match:
            md_parts.append(f"- {bullet_match.group(1)}")
            prev_was_blank = False
            continue
        elif num_match:
            md_parts.append(f"{num_match.group(1)}. {num_match.group(2)}")
            prev_was_blank = False
            continue
        elif x > 50 and len(full_text.split()) <= 20:
            # Indented text that's likely a list item
            md_parts.append(f"- {full_text}")
            prev_was_blank = False
            continue

        # ── Regular paragraph text ────────────────────────────
        # Check for significant vertical gap (new paragraph)
        if i > 0:
            prev_y = lines[i - 1]["y"] + lines[i - 1]["avg_h"]
            gap = line["y"] - prev_y
            if gap > median_height * 1.5 and not prev_was_blank:
                md_parts.append("")  # blank line = new paragraph
                prev_was_blank = True

        md_parts.append(full_text)
        prev_was_blank = False

    # Join and clean up excessive whitespace
    raw_md = "\n".join(md_parts)
    raw_md = re.sub(r'\n{3,}', '\n\n', raw_md).strip()

    # ── Add overall confidence summary at the bottom ──────────
    all_confs = []
    for line in lines:
        all_confs.extend(line["confidences"])
    overall_conf = sum(all_confs) / len(all_confs) if all_confs else 0

    raw_md += f"\n\n---\n\n> **OCR Confidence:** {overall_conf:.0%} (Local EasyOCR Engine)"

    return raw_md


# ── Provider: Local EasyOCR ────────────────────────────────────────────────────

# Removed _refine_with_ollama


async def transcribe_with_local(image_bytes: bytes, prompt: str = "") -> str:
    """
    FREE, UNLIMITED local OCR using EasyOCR + Ollama.
    Pipeline: EasyOCR extracts text → Ollama phi3 refines into beautiful Markdown.
    No API key required. Runs entirely on your machine.
    """
    import asyncio
    import cv2

    readers = _get_easyocr_readers()

    def _run_ocr():
        # Decode image
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            from PIL import Image
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img = np.array(pil_img)[:, :, ::-1]  # RGB → BGR

        # Run ALL readers and merge results
        all_detections = []
        for reader, lang_list in readers:
            try:
                results = reader.readtext(img, paragraph=False, detail=1)
                all_detections.extend(results)
                logger.info("  Reader %s found %d regions", lang_list, len(results))
            except Exception as e:
                logger.error("  Reader %s failed: %s", lang_list, e)

        # Deduplicate overlapping detections — keep highest confidence
        if len(readers) > 1:
            all_detections = _deduplicate_detections(all_detections)

        return all_detections

    logger.info("🔍 Running local EasyOCR transcription (%d readers)...", len(readers))
    detections = await asyncio.to_thread(_run_ocr)
    logger.info("📝 EasyOCR found %d text regions (after dedup)", len(detections))

    # Step 1: Format raw OCR into basic markdown
    raw_markdown = _format_ocr_to_markdown(detections)

    return raw_markdown


def _deduplicate_detections(detections: list) -> list:
    """Remove duplicate/overlapping detections, keeping highest confidence."""
    if not detections:
        return []

    # Sort by confidence descending
    detections.sort(key=lambda d: d[2], reverse=True)

    kept = []
    for det in detections:
        bbox, text, conf = det
        text = text.strip()
        if not text:
            continue

        # Check if this overlaps significantly with any already-kept detection
        center_y = sum(p[1] for p in bbox) / len(bbox)
        center_x = sum(p[0] for p in bbox) / len(bbox)

        is_duplicate = False
        for kept_det in kept:
            k_bbox = kept_det[0]
            k_center_y = sum(p[1] for p in k_bbox) / len(k_bbox)
            k_center_x = sum(p[0] for p in k_bbox) / len(k_bbox)

            # If centers are very close, consider it a duplicate
            if abs(center_y - k_center_y) < 15 and abs(center_x - k_center_x) < 30:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(det)

    return kept


# ── Provider: Ollama Vision ────────────────────────────────────────────────────

# Removed transcribe_with_ollama

# ── Provider: OpenRouter ───────────────────────────────────────────────────────

TRANSCRIPTION_PROMPT = """You are an expert multilingual handwriting-to-text transcription engine and a professional Document Designer acting in 2026.

**Your Task:** Convert the handwriting in this image into clean, structured Markdown with the highest possible accuracy. The handwriting may contain multiple languages, including English and Telugu. Format it as a highly structured, visually modern report following 2026 design standards.

**Rules for Formatting & Transcription:**
1. **Multilingual Support:** Accurately transcribe and maintain the original script of any language found (e.g., Telugu, Hindi, Tamil, Kannada, English). Do NOT translate; transcribe as written.
2. **Structural Hierarchy:** Add a high-contrast Title (#) and subtitle (##) if appropriate. Use logical heading flow (H1 -> H2 -> H3).
3. **Bento Grid & Call-outs:** Use Blockquotes (`>`) for Call-out Blocks like "Pro-Tips" or "Warnings" (Glassmorphism layout).
4. **Variable Typography:** Liberally use **Bold text** for emphasis, and *Italicized text* for quotes.
5. **Data Tools:** Extract tabular data into clean Markdown tables. If a list of metrics is present *in the image*, format it as a horizontal KPI ribbon separated by pipes.
6. **Visual Aids & Dividers:** Use `---` for modern thin dividers between major sections. Use `- [ ]` or `- [x]` for checklists.
7. **Accuracy & Context:** If the input image is blurry or has missing letters, predict the letters based on context. Mark totally illegible words with `⚠️unclear_word⚠️`.
8. **Purity:** Do NOT add any commentary, preamble, or explanations. 
9. **CRITICAL ANTI-HALLUCINATION RULE:** DO NOT invent, hallucinate, or insert ANY text, metrics, examples, or stats that are NOT explicitly written in the provided image. ONLY transcribe and format the actual handwriting!

Begin transcription and modern formatting now:"""


async def transcribe_with_openrouter(image_bytes: bytes, prompt: str = TRANSCRIPTION_PROMPT) -> str:
    """Use OpenRouter (OpenAI-compatible API) for vision-based OCR."""
    from openai import AsyncOpenAI

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY not set in environment")

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://omnidraft.app",
            "X-Title": "OmniDraft",
        },
    )

    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    model = os.getenv("OPENROUTER_VISION_MODEL", "google/gemini-2.0-flash-001")

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_image}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
        temperature=0.1,
        max_tokens=8192,
    )

    text = response.choices[0].message.content
    if not text:
        raise RuntimeError("OpenRouter returned empty response")
    return text.strip()


async def transcribe_with_gemini(image_bytes: bytes, prompt: str = TRANSCRIPTION_PROMPT) -> str:
    """Use Google Gemini directly for vision-based OCR."""
    import asyncio
    import google.generativeai as genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set in environment")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    image_part = {
        "mime_type": "image/jpeg",
        "data": image_bytes,
    }

    def _call_gemini():
        return model.generate_content(
            [prompt, image_part],
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=8192,
            ),
        )

    response = await asyncio.to_thread(_call_gemini)

    if not response.text:
        raise RuntimeError("Gemini returned empty response")
    return response.text.strip()


async def transcribe_with_openai(image_bytes: bytes, prompt: str = TRANSCRIPTION_PROMPT) -> str:
    """Use OpenAI GPT-4o directly for vision-based OCR."""
    from openai import AsyncOpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set in environment")

    client = AsyncOpenAI(api_key=api_key)
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_image}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
        temperature=0.1,
        max_tokens=8192,
    )

    text = response.choices[0].message.content
    if not text:
        raise RuntimeError("OpenAI returned empty response")
    return text.strip()


# ── Main Entry Point ───────────────────────────────────────────────────────────

def _get_provider() -> str:
    return os.getenv("LLM_PROVIDER", "local").lower()


async def transcribe_image(image_bytes: bytes, prompt: Optional[str] = None) -> str:
    """
    Main entry point – dispatches to the configured provider.

    Parameters
    ----------
    image_bytes : JPEG bytes of the preprocessed image.
    prompt      : Override the default transcription prompt (ignored for local).

    Returns
    -------
    Markdown string of the transcription.
    """
    _prompt = prompt or TRANSCRIPTION_PROMPT
    provider = _get_provider()

    logger.info("Transcribing image with provider=%s", provider)

    if provider == "local":
        return await transcribe_with_local(image_bytes, _prompt)
    elif provider == "openrouter":
        return await transcribe_with_openrouter(image_bytes, _prompt)
    elif provider == "gemini":
        return await transcribe_with_gemini(image_bytes, _prompt)
    elif provider == "openai":
        return await transcribe_with_openai(image_bytes, _prompt)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}. Use 'local', 'ollama', 'openrouter', 'gemini', or 'openai'.")
