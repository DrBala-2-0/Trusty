import base64
import os

from config.settings import settings
from openai import OpenAI
from config.settings import settings

client = OpenAI(api_key=settings.GROQ_API_KEY, base_url=settings.GROQ_BASE_URL)

def ask_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model=settings.MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500,
    )
    return response.choices[0].message.content


_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def ask_vision(image_path: str, prompt: str = "Describe this image in detail, including any visible text.") -> str:
    """Caption an image via the vision-capable model (settings.VISION_MODEL_ID),
    kept deliberately separate from ask_llm()'s text model — see blueprint §1.1.

    Deliberately does not catch API errors (rate limits, model unavailability,
    etc.) — they propagate to the caller so failures are loud, not silent.
    No retry/backoff here; that's Ch12's scope, not this loader's."""
    ext = os.path.splitext(image_path)[1].lower()
    mime_type = _MIME_TYPES.get(ext, "image/png")

    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")

    response = client.chat.completions.create(
        model=settings.VISION_MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}},
                ],
            }
        ],
        temperature=0.3,
        max_tokens=500,
    )
    return response.choices[0].message.content


def ask_audio(audio_path: str) -> str:
    """Transcribe an audio file via settings.AUDIO_MODEL_ID.

    Whole-file transcript only — no per-segment timestamps yet (named gap,
    see chapter-9.md: would need the loader contract widened to return
    segments, not just a string). Deliberately does not catch API errors;
    they propagate so failures are loud, matching ask_vision()."""
    with open(audio_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            file=f,
            model=settings.AUDIO_MODEL_ID,
        )
    return transcription.text