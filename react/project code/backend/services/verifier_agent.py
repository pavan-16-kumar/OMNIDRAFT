"""
Verifier Agent – Cross-check Transcription Against Image
─────────────────────────────────────────────────────────
Step B of the Agentic Transcription pipeline.
Supports:
  - "local"      → Confidence-based verification (FREE, no API)
  - "openrouter" → OpenRouter Vision LLM
  - "gemini"     → Google Gemini Vision
  - "openai"     → OpenAI GPT-4o Vision
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from typing import Optional

from models.schemas import VerificationFlag

logger = logging.getLogger(__name__)

VERIFICATION_PROMPT = """You are an expert proofreader and verification agent for handwriting transcription with advanced capabilities in deducing missing information across multiple languages.

You will receive:
1. An IMAGE of handwritten text (may contain English, Telugu, Hindi, Tamil, and other Indian languages).
2. A TRANSCRIPTION of that image in Markdown format.

**Your Task:**
Carefully compare the transcription against the original handwriting in the image. For each word or phrase, assess whether the transcription is accurate, especially for complex scripts like Telugu.

**Output Format — return ONLY a JSON object with this exact structure:**

```json
{{
  "verified_text": "<the corrected/verified Markdown transcription>",
  "confidence_score": <float between 0.0 and 1.0>,
  "flags": [
    {{
      "word": "<the problematic word>",
      "confidence": <float 0-1>,
      "suggestion": "<your suggested correction or null>",
      "context": "<surrounding ~10 words for context>"
    }}
  ]
}}
```

**Rules:**
1. **Multilingual Accuracy:** Fix OCR errors in all languages. If the input is in Telugu/Hindi/Tamil, ensure the script is preserved correctly.
2. Fix obvious OCR errors. If the input image is blurry, has missing letters, predict and deduce the correct letters based on context.
3. Only flag words with confidence < 0.85.
4. **Format Enforcement:** Ensure Headings use logical H1/H2 flow. Use bold and italics for aesthetic hierarchy.
5. **CRITICAL ANTI-HALLUCINATION RULE:** DO NOT invent or insert ANY text not in the provided image.
6. If the transcription is perfect, return it with confidence 1.0 and an empty flags array.
7. Return ONLY the final JSON object — no markdown code fences, no preamble.

**TRANSCRIPTION TO VERIFY:**
{transcription}
"""


def _parse_verification_response(raw: str) -> dict:
    """Parse the LLM's JSON response, handling common formatting issues."""
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        logger.error("Failed to parse verification JSON: %s", cleaned[:500])
        return {
            "verified_text": "",
            "confidence_score": 0.5,
            "flags": [],
        }


# ── Local Verification (FREE, no API) ─────────────────────────────────────────

async def verify_with_local(
    image_bytes: bytes,
    transcription: str,
) -> dict:
    """
    Local verification using EasyOCR confidence scores.
    Runs a second-pass OCR and compares results to find low-confidence regions.
    Completely FREE and UNLIMITED.
    """
    import asyncio

    from services.ocr_agent import _get_easyocr_readers

    def _run_verification():
        import cv2
        import numpy as np

        readers = _get_easyocr_readers()

        # Decode image
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            from PIL import Image
            import io
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img = np.array(pil_img)[:, :, ::-1]

        # Run OCR for confidence analysis (use first reader)
        reader, lang_list = readers[0]
        results = reader.readtext(img, paragraph=False, detail=1)

        # Collect low-confidence words
        flags = []
        all_confidences = []
        for bbox, text, conf in results:
            text = text.strip()
            if not text:
                continue
            all_confidences.append(conf)
            if conf < 0.85:
                flags.append({
                    "word": text,
                    "confidence": round(conf, 2),
                    "suggestion": None,
                    "context": text,
                })

        overall_conf = sum(all_confidences) / len(all_confidences) if all_confidences else 0.5

        return {
            "verified_text": transcription,  # keep original (already best effort)
            "confidence_score": round(overall_conf, 2),
            "flags": flags[:20],  # limit to 20 flags
        }

    logger.info("🔍 Running local verification pass...")
    result = await asyncio.to_thread(_run_verification)
    logger.info("✅ Local verification: confidence=%.2f, flags=%d",
                result["confidence_score"], len(result["flags"]))
    return result


# Removed verify_with_ollama

# ── Cloud Providers ────────────────────────────────────────────────────────────

async def verify_with_openrouter(
    image_bytes: bytes,
    transcription: str,
) -> dict:
    """Use OpenRouter (OpenAI-compatible) to verify the transcription."""
    from openai import AsyncOpenAI

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY not set")

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://omnidraft.app",
            "X-Title": "OmniDraft",
        },
    )

    prompt = VERIFICATION_PROMPT.format(transcription=transcription)
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
        temperature=0.05,
        max_tokens=8192,
    )

    return _parse_verification_response(response.choices[0].message.content or "{}")


async def verify_with_gemini(
    image_bytes: bytes,
    transcription: str,
) -> dict:
    """Use Gemini directly to verify the transcription against the image."""
    import asyncio
    import google.generativeai as genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    prompt = VERIFICATION_PROMPT.format(transcription=transcription)
    image_part = {"mime_type": "image/jpeg", "data": image_bytes}

    def _call_gemini():
        return model.generate_content(
            [prompt, image_part],
            generation_config=genai.GenerationConfig(
                temperature=0.05,
                max_output_tokens=8192,
            ),
        )

    response = await asyncio.to_thread(_call_gemini)

    return _parse_verification_response(response.text or "{}")


async def verify_with_openai(
    image_bytes: bytes,
    transcription: str,
) -> dict:
    """Use GPT-4o directly to verify the transcription against the image."""
    from openai import AsyncOpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set")

    client = AsyncOpenAI(api_key=api_key)
    prompt = VERIFICATION_PROMPT.format(transcription=transcription)
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
        temperature=0.05,
        max_tokens=8192,
    )

    return _parse_verification_response(response.choices[0].message.content or "{}")


# ── Main Entry Point ──────────────────────────────────────────────────────────

async def verify_transcription(
    image_bytes: bytes,
    transcription: str,
    provider: Optional[str] = None,
) -> tuple[str, float, list[VerificationFlag]]:
    """
    Main verification entry point.

    Parameters
    ----------
    image_bytes   : JPEG bytes of the original (preprocessed) image.
    transcription : The raw Markdown from the OCR agent (Step A).
    provider      : Override the default LLM_PROVIDER env variable.

    Returns
    -------
    (verified_markdown, confidence_score, flags)
    """
    prov = provider or os.getenv("LLM_PROVIDER", "local").lower()
    logger.info("Verifying transcription with provider=%s", prov)

    if prov == "local":
        result = await verify_with_local(image_bytes, transcription)
    elif prov == "openrouter":
        result = await verify_with_openrouter(image_bytes, transcription)
    elif prov == "gemini":
        result = await verify_with_gemini(image_bytes, transcription)
    elif prov == "openai":
        result = await verify_with_openai(image_bytes, transcription)
    else:
        raise ValueError(f"Unknown provider: {prov}")

    verified_text = result.get("verified_text", "") or transcription
    confidence = float(result.get("confidence_score", 0.5))
    raw_flags = result.get("flags", [])

    flags = []
    for f in raw_flags:
        try:
            flags.append(VerificationFlag(
                word=f.get("word", ""),
                confidence=float(f.get("confidence", 0.5)),
                suggestion=f.get("suggestion"),
                context=f.get("context", ""),
            ))
        except Exception as e:
            logger.warning("Skipping malformed flag: %s", e)

    logger.info(
        "Verification complete: confidence=%.2f, flags=%d",
        confidence,
        len(flags),
    )
    return verified_text, confidence, flags
