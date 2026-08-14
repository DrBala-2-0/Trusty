import base64
import os

from openai import OpenAI

from config.settings import settings
from utils.retry import with_retry

client = OpenAI(api_key=settings.GROQ_API_KEY, base_url=settings.GROQ_BASE_URL)


# ---------------------------------------------------------------------------
# Internal retryable call — wrapped separately from ask_llm() so the
# retry decorator only covers the network call, not any pre/post processing
# that ask_llm() might do in future.
# ---------------------------------------------------------------------------

@with_retry(
    max_attempts=3,
    initial_delay_s=2.0,
    backoff_factor=2.0,
    max_delay_s=16.0,
)
def _call_llm(prompt: str) -> str:
    """Make one LLM text completion call with a hard timeout.

    Separated from ask_llm() so @with_retry wraps only the network call.
    Not intended to be called directly outside this module.
    """
    response = client.chat.completions.create(
        model=settings.MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500,
        timeout=30,          # seconds — hard ceiling per attempt
    )
    return response.choices[0].message.content


def ask_llm(prompt: str) -> str:
    """Submit a text prompt and return the model's response.

    Retries up to 3 times with exponential backoff on transient errors
    (429 rate limits, 5xx server errors). Hard timeout of 30s per attempt.
    Raises on non-retryable errors (400 bad request, 401 auth failure).
    """
    return _call_llm(prompt)


# ---------------------------------------------------------------------------
# Vision and audio — timeout added, retry deliberately omitted.
# See reasoning in design decisions below.
# ---------------------------------------------------------------------------

_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def ask_vision(
    image_path: str,
    prompt: str = "Describe this image in detail, including any visible text.",
) -> str:
    """Caption an image via the vision-capable model (settings.VISION_MODEL_ID).

    Deliberately kept separate from ask_llm()'s text model — see blueprint §1.1
    and §2.9. No retry here: the vision model is Groq preview-tier and may be
    discontinued at short notice — retrying a model that's gone is pointless.
    Failures propagate to the caller so they are loud, not silent.
    """
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
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
        temperature=0.3,
        max_tokens=500,
        timeout=60,          # vision calls are slower than text — wider ceiling
    )
    return response.choices[0].message.content


def ask_audio(audio_path: str) -> str:
    """Transcribe an audio file via settings.AUDIO_MODEL_ID.

    Whole-file transcript only — no per-segment timestamps yet (named gap,
    see chapter-9.md). No retry: audio transcription is idempotent but
    uploading the same file bytes multiple times on a transient failure
    wastes bandwidth; better to surface the error fast and let the user
    re-upload. Failures propagate so they are loud, not silent.
    """
    with open(audio_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            file=f,
            model=settings.AUDIO_MODEL_ID,
            timeout=120,     # large audio files can take time — wider ceiling
        )
    return transcription.text