import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    MODEL_ID: str = os.getenv("MODEL_ID", "openai/gpt-oss-120b")
    VISION_MODEL_ID: str = os.getenv("VISION_MODEL_ID", "qwen/qwen3.6-27b")
    AUDIO_MODEL_ID: str = os.getenv("AUDIO_MODEL_ID", "whisper-large-v3-turbo")

    HF_API_KEY: str = os.getenv("HF_API_KEY", "")
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    CHROMA_DB_DIR: str = os.getenv("CHROMA_DB_DIR", ".cache/chroma_db")
    CHROMA_COLLECTION_NAME: str = "trusty_docs"
    VECTOR_SEARCH_K: int = 5
    HYBRID_RETRIEVER_WEIGHTS: list = None

    def __post_init__(self):
        if self.HYBRID_RETRIEVER_WEIGHTS is None:
            self.HYBRID_RETRIEVER_WEIGHTS = [0.4, 0.6]  # [BM25, vector]

settings = Settings()