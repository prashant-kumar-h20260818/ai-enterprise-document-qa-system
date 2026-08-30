from __future__ import annotations

import base64
import io
import mimetypes
from dataclasses import dataclass
from typing import Optional, Tuple

from groq import Groq
from PIL import Image


VISION_PROMPT = """You are extracting evidence from a document image for a RAG system.

Extract ALL meaningful information visible in the image. Preserve structure instead of merely
summarizing it.

Rules:
- Transcribe important text faithfully.
- Convert tables to Markdown tables when possible.
- For charts/plots, state the chart title, axes, legend/series, visible values/trends, and labels.
- For diagrams/flowcharts/architecture drawings, describe nodes, labels, arrows, connections,
  direction, hierarchy, and spatial relationships.
- For screenshots/forms, preserve field labels and values.
- For formulas, write the formula in readable text/LaTeX when possible.
- Describe meaningful images/figures and their captions.
- Do not invent values that are unreadable; mark them as unreadable/unclear.
- Return only extracted document evidence in concise Markdown.
"""


def _prepare_image(image_bytes: bytes, mime_type: str) -> Tuple[bytes, str]:
    """Keep document details readable while staying below common API image limits."""
    if len(image_bytes) <= 12 * 1024 * 1024:
        return image_bytes, mime_type

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.thumbnail((2400, 2400))
        if image.mode not in {"RGB", "L"}:
            image = image.convert("RGB")
        out = io.BytesIO()
        image.save(out, format="JPEG", quality=90, optimize=True)
        return out.getvalue(), "image/jpeg"
    except Exception:
        return image_bytes, mime_type


@dataclass
class GroqVisionExtractor:
    """Best-effort multimodal extractor for images, diagrams and scanned pages.

    The vision model is deliberately configurable because provider model names can change.
    """

    api_key: str
    model_name: str
    max_tokens: int = 1600

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is required for visual extraction")
        self.client = Groq(api_key=self.api_key)

    def describe_image_bytes(
        self,
        image_bytes: bytes,
        mime_type: Optional[str] = None,
        extra_instruction: str = "",
    ) -> str:
        if not image_bytes:
            return ""
        mime = mime_type or "image/png"
        image_bytes, mime = _prepare_image(image_bytes, mime)
        encoded = base64.b64encode(image_bytes).decode("ascii")
        prompt = VISION_PROMPT
        if extra_instruction:
            prompt += f"\nAdditional context:\n{extra_instruction.strip()}\n"

        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=0.0,
            max_completion_tokens=self.max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{encoded}"},
                        },
                    ],
                }
            ],
        )
        return (response.choices[0].message.content or "").strip()


def guess_mime_type(filename: str) -> str:
    mime, _ = mimetypes.guess_type(filename)
    return mime or "image/png"
