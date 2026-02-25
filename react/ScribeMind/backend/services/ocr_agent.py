"""
OCR Agent – Vision LLM Transcription
──────────────────────────────────────
Step A of the Agentic Transcription pipeline.
Sends the preprocessed handwriting image to a Vision LLM
(via OpenRouter, Gemini, or OpenAI) and returns structured Markdown.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# ── Provider helpers ───────────────────────────────────────────────────────────

TRANSCRIPTION_PROMPT = """You are an expert handwriting-to-text transcription engine and a professional Document Designer acting in 2026.

**Your Task:** Convert the handwriting in this image into clean, structured Markdown with the highest possible accuracy, formatting it as a highly structured, visually modern report following 2026 design standards.

**Rules for Formatting & Transcription:**
1. **Structural Hierarchy:** Add a high-contrast Title (#) and subtitle (##) if appropriate. Use logical heading flow (H1 -> H2 -> H3).
2. **Bento Grid & Call-outs:** Use Blockquotes (`>`) for Call-out Blocks like "Pro-Tips" or "Warnings" (Glassmorphism layout).
3. **Variable Typography:** Liberally use **Bold text** for emphasis, and *Italicized text* for quotes.
4. **Data Tools:** Extract tabular data into clean Markdown tables. If a list of metrics is present *in the image*, format it as a horizontal KPI ribbon separated by pipes.
5. **Visual Aids & Dividers:** Use `---` for modern thin dividers between major sections. Use `- [ ]` or `- [x]` for checklists.
6. **Accuracy & Context:** If the input image is blurry or has missing letters, predict the letters based on context. Mark totally illegible words with `⚠️unclear_word⚠️`.
7. **Purity:** Do NOT add any commentary, preamble, or explanations. 
8. **CRITICAL ANTI-HALLUCINATION RULE:** DO NOT invent, hallucinate, or insert ANY text, metrics, examples, or stats (e.g. "Material Cost", "Wage", etc.) that are NOT explicitly written in the provided image. ONLY transcribe and format the actual handwriting!

Begin transcription and modern formatting now:"""


def _get_provider() -> str:
    return os.getenv("LLM_PROVIDER", "openrouter").lower()


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

    response = model.generate_content(
        [prompt, image_part],
        generation_config=genai.GenerationConfig(
            temperature=0.1,
            max_output_tokens=8192,
        ),
    )

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


async def transcribe_image(image_bytes: bytes, prompt: Optional[str] = None) -> str:
    """
    Main entry point – dispatches to the configured LLM provider.

    Parameters
    ----------
    image_bytes : JPEG bytes of the preprocessed image.
    prompt      : Override the default transcription prompt.

    Returns
    -------
    Markdown string of the transcription.
    """
    _prompt = prompt or TRANSCRIPTION_PROMPT
    provider = _get_provider()

    logger.info("Transcribing image with provider=%s", provider)

    if provider == "openrouter":
        return await transcribe_with_openrouter(image_bytes, _prompt)
    elif provider == "gemini":
        return await transcribe_with_gemini(image_bytes, _prompt)
    elif provider == "openai":
        return await transcribe_with_openai(image_bytes, _prompt)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}. Use 'openrouter', 'gemini', or 'openai'.")
