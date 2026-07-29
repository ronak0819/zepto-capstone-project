from __future__ import annotations

import json
from typing import Literal, TypedDict

import requests
from langgraph.graph import END, StateGraph
from pydantic import ValidationError

from support_assistant.config import GROQ_API_KEY, GROQ_MODEL, MOCK_LLM
from support_assistant.prompt_template import SUPPORT_PROMPT_TEMPLATE
from support_assistant.schemas import AssistantResponse
from support_assistant.vector_store import query_collection

Intent = Literal["policy_question", "general_question"]
POLICY_KEYWORDS = {
    "delivery",
    "fee",
    "return",
    "refund",
    "membership",
    "pass",
    "track",
    "rider",
    "cancel",
    "damaged",
    "spoiled",
    "missing",
    "gift card",
    "support",
    "order",
}


class SupportState(TypedDict, total=False):
    query: str
    intent: Intent
    retrieved_chunks: list[dict]
    answer: str
    sources: list[str]
    confidence: float
    response: dict


def call_real_llm(prompt: str) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is required when MOCK_LLM=0.")
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def validate_real_output(prompt: str) -> AssistantResponse:
    current = prompt
    for _ in range(3):
        try:
            return AssistantResponse.model_validate(json.loads(call_real_llm(current)))
        except (json.JSONDecodeError, ValidationError):
            current = (
                prompt
                + "\nReturn only valid JSON matching answer:string, sources:list[str], "
                "confidence:number from 0 to 1."
            )
    return AssistantResponse(
        answer="ERROR: The real LLM response failed schema validation after three attempts.",
        sources=[],
        confidence=0.0,
    )


def classify_intent(state: SupportState) -> SupportState:
    query = state["query"].strip()
    if MOCK_LLM:
        intent: Intent = (
            "policy_question"
            if any(keyword in query.lower() for keyword in POLICY_KEYWORDS)
            else "general_question"
        )
    else:
        label = call_real_llm(
            "Classify as exactly policy_question or general_question:\n" + query
        ).lower()
        intent = "policy_question" if "policy_question" in label else "general_question"
    return {**state, "intent": intent}


def retrieve_and_answer(state: SupportState) -> SupportState:
    retrieved = query_collection(state["query"], n_results=3)
    sources = [chunk["id"] for chunk in retrieved]
    if MOCK_LLM:
        text = retrieved[0]["document"].strip()
        snippet = text[:200] + ("..." if len(text) > 200 else "")
        return {
            **state,
            "retrieved_chunks": retrieved,
            "answer": f"Based on the retrieved context: {snippet}",
            "sources": sources,
            "confidence": 1.0,
        }
    context = "\n\n".join(
        f"Source: {chunk['id']}\n{chunk['document']}" for chunk in retrieved
    )
    validated = validate_real_output(
        SUPPORT_PROMPT_TEMPLATE.format(context=context, query=state["query"])
    )
    return {
        **state,
        "retrieved_chunks": retrieved,
        "answer": validated.answer,
        "sources": validated.sources,
        "confidence": validated.confidence,
    }


def direct_answer(state: SupportState) -> SupportState:
    if MOCK_LLM:
        return {
            **state,
            "answer": "I can only answer questions about Zepto policies right now.",
            "sources": [],
            "confidence": 1.0,
        }
    validated = validate_real_output(
        f"Answer this question in JSON with answer, sources:[], confidence:\n{state['query']}"
    )
    return {**state, **validated.model_dump(), "sources": []}


def finalize_response(state: SupportState) -> SupportState:
    validated = AssistantResponse(
        answer=state["answer"],
        sources=state.get("sources", []),
        confidence=state.get("confidence", 1.0),
    )
    return {**state, "response": validated.model_dump()}


def build_graph():
    graph = StateGraph(SupportState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)
    graph.add_node("finalize_response", finalize_response)
    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        lambda state: state["intent"],
        {"policy_question": "retrieve_and_answer", "general_question": "direct_answer"},
    )
    graph.add_edge("retrieve_and_answer", "finalize_response")
    graph.add_edge("direct_answer", "finalize_response")
    graph.add_edge("finalize_response", END)
    return graph.compile()


support_graph = build_graph()


def ask_support_assistant(query: str) -> AssistantResponse:
    state = support_graph.invoke({"query": query})
    return AssistantResponse.model_validate(state["response"])
