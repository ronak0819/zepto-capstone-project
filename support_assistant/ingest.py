from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from support_assistant.config import DOCS_DIR
from support_assistant.vector_store import embed_texts, get_collection


def load_documents() -> list[dict[str, str]]:
    documents = []
    for path in sorted(DOCS_DIR.glob("doc_*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Empty document: {path.name}")
        documents.append(
            {"document_id": path.stem, "chunk_id": f"{path.stem}_chunk_01", "text": text}
        )
    if len(documents) != 8:
        raise ValueError(f"Exactly eight policy documents are required; found {len(documents)}.")
    return documents


def ingest_documents() -> None:
    documents = load_documents()
    collection = get_collection()
    existing_ids = collection.get(include=[]).get("ids", [])
    if existing_ids:
        collection.delete(ids=existing_ids)
    texts = [document["text"] for document in documents]
    collection.add(
        ids=[document["chunk_id"] for document in documents],
        documents=texts,
        embeddings=embed_texts(texts),
        metadatas=[
            {
                "document_id": document["document_id"],
                "filename": f"{document['document_id']}.txt",
            }
            for document in documents
        ],
    )
    print(f"Ingested {collection.count()} chunks into ChromaDB.")


if __name__ == "__main__":
    ingest_documents()
