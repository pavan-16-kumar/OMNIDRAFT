"""
Image Pre-Processing Service
─────────────────────────────
Uses OpenCV to deskew, denoise, and normalise handwriting images
before they are sent to the Vision LLM.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Try importing HEIC support (optional)
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIF_SUPPORT = True
except ImportError:
    HEIF_SUPPORT = False
    logger.warning("pillow-heif not installed – HEIC files won't be supported.")


def load_image(source: Union[str, Path, bytes]) -> np.ndarray:
    """Load an image from path or raw bytes into a BGR numpy array."""
    if isinstance(source, (str, Path)):
        path = str(source)
        if path.lower().endswith((".heic", ".heif")):
            pil_img = Image.open(path).convert("RGB")
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not read image at {path}")
        return img
    else:
        # bytes
        arr = np.frombuffer(source, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            # Try via PIL (handles HEIC bytes)
            pil_img = Image.open(io.BytesIO(source)).convert("RGB")
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return img


def _deskew(image: np.ndarray) -> np.ndarray:
    """Correct rotation / skew using Hough line detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 10:
        return image

    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Only correct if skew is small (< 15°); large angles may be intentional
    if abs(angle) > 15 or abs(angle) < 0.5:
        return image

    h, w = image.shape[:2]
    centre = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
    rotated = cv2.warpAffine(
        image, matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    logger.info("Deskewed image by %.2f°", angle)
    return rotated


def _denoise(image: np.ndarray) -> np.ndarray:
    """Remove noise while preserving text edges (using fast median blur)."""
    return cv2.medianBlur(image, 3)


def _enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE for local contrast enhancement."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge([l_channel, a_channel, b_channel])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _sharpen(image: np.ndarray) -> np.ndarray:
    """Apply an unsharp mask to clarify blurry text edges and characters."""
    gaussian_blur = cv2.GaussianBlur(image, (0, 0), 2.0)
    # 1.5 * image - 0.5 * blurred = sharpened image
    return cv2.addWeighted(image, 1.5, gaussian_blur, -0.5, 0)


def _resize_for_llm(image: np.ndarray, max_dim: int = 1600) -> np.ndarray:
    """Resize so the longest edge is at most max_dim pixels."""
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def preprocess_image(source: Union[str, Path, bytes]) -> tuple[np.ndarray, bytes]:
    """
    Full preprocessing pipeline.

    Returns
    -------
    (processed_cv_image, jpeg_bytes)
    """
    img = load_image(source)
    img = _deskew(img)
    img = _denoise(img)
    img = _enhance_contrast(img)
    img = _sharpen(img)  # recover details from blur
    img = _resize_for_llm(img)

    # Encode as high-quality JPEG for the LLM
    success, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not success:
        raise RuntimeError("Failed to encode processed image to JPEG")

    return img, buf.tobytes()


def extract_images_from_pdf(pdf_path: str | Path) -> list[bytes]:
    """Extract page images from a PDF file using PyMuPDF."""
    import fitz  # PyMuPDF

    images: list[bytes] = []
    doc = fitz.open(str(pdf_path))
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Extract at 150 DPI for faster processing (standard A4 is ~1750px high at 150 DPI, ideal for LLM)
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("jpeg", 85) # slightly compress to save memory/time
        images.append(img_bytes)
    doc.close()
    logger.info("Extracted %d page images from PDF", len(images))
    return images
