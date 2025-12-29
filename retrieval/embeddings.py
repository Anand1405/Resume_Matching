
import os
import numpy as np
from google import genai
from google.genai import types
from config.settings import settings

class GeminiEmbeddingGenerator:
    def __init__(self):
        print(f"Loading Gemini embedding model: {settings.EMBEDDING_MODEL_ID}...")
        api_key = settings.GOOGLE_API_KEY
        if not api_key:
            # Fallback for initialization, will fail on generation if not set later
            print("Warning: GOOGLE_API_KEY not set in settings.")
            self.client = None
        else:
            self.client = genai.Client(api_key=api_key)
        self.model_name = settings.EMBEDDING_MODEL_ID

    def generate(self, text: str) -> np.ndarray:
        if not self.client:
             # Try to re-init if key is now available (e.g. set via UI)
            if settings.GOOGLE_API_KEY:
                self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)
            else:
                raise ValueError("GOOGLE_API_KEY is missing")

        try:
            # text-embedding-005 supports 768 dimensions usually
            # We use 'text-embedding-004' in some docs, but 005 was in user plan/search
            # Let's ensure we match the config.
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=text,
                config=types.EmbedContentConfig(output_dimensionality=768)
            )
            return np.array(result.embeddings[0].values)
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Return zero vector fallback
            return np.zeros(768)

    def cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))
