# Zepto Data and AI Project

This repository contains three connected modules:

1. `data_pipeline` — web scraping, cleaning, currency conversion, SQLite, SQL, and pandas analysis.
2. `analytics` — Titanic profiling, EDA, classification, imbalance handling, tuning, and regression.
3. `support_assistant` — an offline-first RAG support API using Sentence Transformers, ChromaDB, LangGraph, Pydantic, and FastAPI.

## Setup

Python 3.11 is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
python data_pipeline\main.py
python analytics\01_eda.py
python analytics\02_modeling.py
python support_assistant\ingest.py
pytest support_assistant\tests -v
uvicorn support_assistant.main:app --reload
```

See each module's README for its methodology, generated outputs, and interpretation.
