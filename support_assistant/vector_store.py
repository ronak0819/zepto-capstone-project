from __future__ import annotations

from functools import lru_cache
from typing import Any

import chromadb
from sklearn.feature_extraction.text import HashingVectorizer

from support_assistant.config import CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL_NAME


@lru_cache(maxsize=1)
def get_embedding_model():
    """Load MiniLM, with a deterministic fallback for constrained offline hosts."""
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(EMBEDDING_MODEL_NAME)
    except (ImportError, OSError, RuntimeError) as error:
        print(f"Sentence Transformers unavailable ({error}); using offline hashing embeddings.")
        return HashingVectorizer(
            n_features=384,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
            stop_words="english",
        )


@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.PersistentClient:
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_PATH))


def get_collection():
    return get_chroma_client().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "description": "Zepto support policy corpus"},
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    if isinstance(model, HashingVectorizer):
        return model.transform(texts).toarray().tolist()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return embeddings.tolist()


def query_collection(query: str, n_results: int = 3) -> list[dict[str, Any]]:
    collection = get_collection()
    if collection.count() == 0:
        raise RuntimeError("The Chroma collection is empty. Run support_assistant/ingest.py.")
    results = collection.query(
        query_embeddings=[embed_texts([query])[0]],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "id": chunk_id,
            "document": document,
            "metadata": metadata or {},
            "distance": float(distance),
        }
        for chunk_id, document, metadata, distance in zip(
            results.get("ids", [[]])[0],
            results.get("documents", [[]])[0],
            results.get("metadatas", [[]])[0],
            results.get("distances", [[]])[0],
            strict=True,
        )
    ]
