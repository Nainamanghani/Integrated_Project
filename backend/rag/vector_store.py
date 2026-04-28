import uuid
from typing import Any, Dict, List

import chromadb

from .config import settings
from .embeddings import embed_query, embed_texts


class VectorStore:
    def __init__(self, namespace: str):
        self.namespace = namespace or settings.default_project
        self.client = chromadb.PersistentClient(
    path=settings.chroma_persist_directory
)
        self.collection = self.client.get_or_create_collection(name=self.namespace)

    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        if not chunks:
            return

        ids = [chunk.get("id", str(uuid.uuid4())) for chunk in chunks]
        documents = [chunk["content"] for chunk in chunks]
        metadatas = [
            {
                "source": chunk.get("source", "unknown"),
                "chunk_index": chunk.get("chunk_index", 0),
            }
            for chunk in chunks
        ]
        embeddings = embed_texts(documents)
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        embedding = embed_query(query_text)
        if not embedding:
            return []

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
        )

        hits = []
        if results and results.get("documents"):
            for i in range(len(results["documents"][0])):
                hits.append(
                    {
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                    }
                )
        return hits

    def delete_collection(self) -> None:
        self.client.delete_collection(name=self.namespace)
