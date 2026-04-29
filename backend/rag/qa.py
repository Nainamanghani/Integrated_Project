from typing import Any, Dict, List

from .config import settings
from .vector_store import VectorStore


def answer_question(question: str, project: str = None, top_k: int = 5) -> Dict[str, Any]:
    """Retrieve relevant chunks and return them as a structured answer payload."""
    project_name = project or settings.default_project
    store = VectorStore(project_name)
    retrieved = store.query(question, top_k=top_k)

    if not retrieved:
        return {
            "answer": "No relevant document content was found for this question.",
            "sources": [],
            "retrieval_count": 0,
        }

    answer_parts = [doc.get("content", "") for doc in retrieved if doc.get("content")]
    answer = "\n\n".join(answer_parts)

    sources = [
        {
            "source": doc["metadata"].get("source"),
            "chunk_index": doc["metadata"].get("chunk_index"),
            "distance": doc.get("distance"),
        }
        for doc in retrieved
    ]

    return {
        "answer": answer,
        "sources": sources,
        "retrieval_count": len(retrieved),
    }


def retrieve_context(topic: str, project: str = None, top_k: int = 5) -> str:
    """
    Lightweight helper used by the LangGraph RAG node.
    Returns a single string of retrieved context, or empty string if nothing found.
    """
    payload = answer_question(topic, project=project, top_k=top_k)
    if payload["retrieval_count"] == 0:
        return ""
    return payload["answer"]
