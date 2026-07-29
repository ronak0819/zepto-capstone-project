# Module 3 — Zepto Support Assistant

This offline-first RAG service uses `all-MiniLM-L6-v2` embeddings, a persistent cosine-distance ChromaDB collection, LangGraph routing, Pydantic response validation, FastAPI, and Docker. The graded/default `MOCK_LLM=1` mode never contacts an external LLM.

If PyTorch or the model cache is unavailable on a constrained offline host, retrieval uses a deterministic 384-feature hashing fallback. A normal installation uses the required MiniLM model.

```powershell
python support_assistant\ingest.py
$env:MOCK_LLM="1"
uvicorn support_assistant.main:app --reload
pytest support_assistant\tests -v
```

The eight policy files produce deterministic chunk IDs. Policy queries retrieve the nearest three chunks; general questions return a fixed scope message. The complete response schema is:

```json
{"answer": "string", "sources": ["chunk IDs"], "confidence": 1.0}
```

Docker:

```powershell
docker build -t zepto-support-assistant -f support_assistant\Dockerfile .
docker run --rm -p 7860:7860 -e MOCK_LLM=1 zepto-support-assistant
```

<!-- RESPONSE_START -->
Exact local mock response for `What is the delivery fee?`:

```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del...",
  "sources": [
    "doc_01_chunk_01",
    "doc_04_chunk_01",
    "doc_05_chunk_01"
  ],
  "confidence": 1.0
}
```
<!-- RESPONSE_END -->
