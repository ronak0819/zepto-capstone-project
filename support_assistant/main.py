from fastapi import FastAPI

from support_assistant.config import MOCK_LLM
from support_assistant.graph import ask_support_assistant
from support_assistant.schemas import AskRequest, AssistantResponse

app = FastAPI(
    title="Zepto Support Assistant",
    description="Offline-first RAG support service for Zepto policy questions.",
    version="1.0.0",
)


@app.get("/")
def root() -> dict[str, object]:
    return {
        "service": "Zepto Support Assistant",
        "mock_llm": MOCK_LLM,
        "endpoint": "/ask",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "healthy", "mock_llm": MOCK_LLM}


@app.post("/ask", response_model=AssistantResponse)
def ask(request: AskRequest) -> AssistantResponse:
    return ask_support_assistant(request.query)
