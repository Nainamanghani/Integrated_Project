from typing import List
from sentence_transformers import SentenceTransformer

# Free local embedding model — no API key needed
model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    return model.encode(texts).tolist()


def embed_query(query: str) -> List[float]:
    return model.encode(query).tolist()
