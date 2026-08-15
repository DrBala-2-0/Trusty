import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # LLM providers
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    MODEL_ID: str = os.getenv("MODEL_ID", "openai/gpt-oss-120b")
    VISION_MODEL_ID: str = os.getenv("VISION_MODEL_ID", "qwen/qwen3-32b")
    AUDIO_MODEL_ID: str = os.getenv("AUDIO_MODEL_ID", "whisper-large-v3-turbo")

    # Embeddings
    HF_API_KEY: str = os.getenv("HF_API_KEY", "")
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Retriever
    CHROMA_DB_DIR: str = os.getenv("CHROMA_DB_DIR", ".cache/chroma_db")
    CHROMA_COLLECTION_NAME: str = "trusty_docs"
    VECTOR_SEARCH_K: int = int(os.getenv("VECTOR_SEARCH_K", "5"))
    HYBRID_RETRIEVER_WEIGHTS: list = field(default_factory=lambda: [0.4, 0.6])

    # Session budget
    SESSION_LLM_CALL_LIMIT: int = int(os.getenv("SESSION_LLM_CALL_LIMIT", "20"))

    # Cache
    RESPONSE_CACHE_MAX_SIZE: int = int(os.getenv("RESPONSE_CACHE_MAX_SIZE", "500"))


settings = Settings()