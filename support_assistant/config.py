from __future__ import annotations

import os
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
DOCS_DIR = MODULE_DIR / "docs"
CHROMA_PATH = MODULE_DIR / "chroma_db"
COLLECTION_NAME = "zepto_policy_documents"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MOCK_LLM = os.getenv("MOCK_LLM", "1").strip() != "0"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
