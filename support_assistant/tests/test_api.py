from fastapi.testclient import TestClient

from support_assistant.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "mock_llm": True}


def test_policy_question() -> None:
    response = client.post("/ask", json={"query": "What is the delivery fee?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("Based on the retrieved context:")
    assert len(body["sources"]) == 3
    assert body["confidence"] == 1.0


def test_general_question() -> None:
    response = client.post("/ask", json={"query": "Who invented Python?"})
    assert response.status_code == 200
    assert response.json() == {
        "answer": "I can only answer questions about Zepto policies right now.",
        "sources": [],
        "confidence": 1.0,
    }


def test_invalid_empty_query() -> None:
    assert client.post("/ask", json={"query": ""}).status_code == 422
