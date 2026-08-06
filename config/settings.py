import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    MODEL_ID: str = os.getenv("MODEL_ID", "llama-3.3-70b-versatile") #specid

settings = Settings()