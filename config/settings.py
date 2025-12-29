import os
from pydantic_settings import BaseSettings
import dotenv
dotenv.load_dotenv()

class Settings(BaseSettings):
    # API Configuration
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    MODEL_PROVIDER: str = os.getenv("MODEL_PROVIDER", "gemini")

    # Model Selection (Gemini 2.5 Pro for speed/cost/quality)
    EXTRACTION_MODEL: str = os.getenv("EXTRACTION_MODEL", "gemini-2.5-flash")
    EMBEDDING_MODEL_ID: str = os.getenv("EMBEDDING_MODEL_ID", "gemini-embedding-001")
    RERANK_MODEL: str = os.getenv("RERANK_MODEL", "gemini-2.5-pro")

    # Legacy fallbacks
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-2.5-flash")

    # Retrieval Config
    SEMANTIC_WEIGHT: float = 0.7
    LEXICAL_WEIGHT: float = 0.3
    TOP_N_CANDIDATES: int = 5
    MIN_SCORE_THRESHOLD: float = 0.4

    # Paths
    INDEX_PATH: str = "data/index"
    PROCESSED_DATA_PATH: str = "data/processed"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
