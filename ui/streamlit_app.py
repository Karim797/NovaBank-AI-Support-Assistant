"""Streamlit demo UI.

Architectural rule: the UI holds **no** business logic. It formats a question,
POSTs it to the API, and renders the response. Every decision - intent, routing,
retrieval, grounding, escalation - happens behind FastAPI. That is what makes
the UI disposable: replacing it with a web app, a WhatsApp bot or an IVR changes
nothing behind the API boundary.

Run locally:
    streamlit run ui/streamlit_app.py

Deployment:
    The production API defaults to the Railway service below. Override API_URL
    and API_KEY with environment variables / Streamlit Community Cloud secrets.
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

DEFAULT_API_URL = "https://novabank-ai-support-assistant-production.up.railway.app"
API_URL = os.getenv("API_URL", DEFAULT_API_URL).rstrip("/")
API_KEY = os.getenv("API_KEY", "")
HEADERS = {"x-api-key": API_KEY} if API_KEY else {}

ROUTE_HELP = {
    "deterministic_policy": "Fixed, compliance-approved answer. No LLM involved.",
    "rag_topic_filtered": "Confident intent - retrieval was restricted to that intent's topics.",
    "rag_unfiltered": "Low-confidence intent - searched the whole knowledge base.",
    "no_relevant_context": "Nothing in the knowledge base cleared the relevance floor.",
    "ungrounded_answer": "The model's answer cited nothing valid, so it was discarded.",
    "llm_unavailable": "The generation provider failed. Degraded response.",
}

st.set_page_config(page_title="NovaBank Support Assistant", page_icon="🏦", layout="centered")
st.title("NovaBank Support Assistant")
st.caption("NovaBank is a fictional bank. Policies are synthetic and exist to demonstrate RAG.")

with st.sidebar:
    st.subheader("Service")
    try:
        ready_response = httpx.get(f"{API_URL}/ready", timeout=8)
        ready_response.raise_for_status()
        ready = ready_response.json()
        st.success("ready" if ready.get("status") == "ready" else "not ready")
        st.write({k: v for k, v in ready.items() if k != "checks"})
    except Exception as exc:  # noqa: BLE001
        st.error(f"cannot reach API at {API_URL}: {exc}")

    st.divider()
    st.subheader("Try")
    for example in [
        "My card was charged twice",
        "Why is my cash withdrawal still pending?",
        "How do I replace a lost card?",
        "What are the international transfer fees?",
        "Why was my bank transfer rejected?",
        "What is the capital of France?",
    ]:
        if st.button(example, use_container_width=True):
            st.session_state["question"] = example

question = st.text_input("Your question", key="question", placeholder="Ask about your NovaBank account")

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Thinking..."):
        try:
            r = httpx.post(
                f"{API_URL}/chat",
                json={"question": question},
                headers=HEADERS,
                timeout=60,
            )
            if r.status_code == 401:
                st.error(
                    "The Railway API requires an API key. Add API_KEY to this app's "
                    "Streamlit Community Cloud secrets, using the same API_KEY configured on Railway."
                )
                st.stop()
            r.raise_for_status()
            data = r.json()
        except Exception as exc:  # noqa: BLE001
            st.error(f"request failed: {exc}")
            st.stop()

    st.session_state["last"] = data

if "last" in st.session_state:
    data = st.session_state["last"]
    st.markdown("### Answer")
    st.write(data["answer"])
    if data["needs_human_escalation"]:
        st.warning("This case is flagged for a human agent.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Intent", data["intent"])
    c2.metric("Confidence", f"{data['intent_confidence']:.2f}")
    c3.metric("Latency", f"{data['latency_ms']} ms")
    st.caption(f"Route: `{data['route']}` — {ROUTE_HELP.get(data['route'], '')}")

    if data["sources"]:
        with st.expander(f"Sources ({len(data['sources'])})", expanded=True):
            for s in data["sources"]:
                st.markdown(
                    f"**{s['title']} — {s['section']}**  \n"
                    f"`{s['chunk_id']}` (score {s['score']})"
                )
    else:
        st.info("No sources - the assistant declined to answer from the knowledge base.")

    st.caption(
        f"model={data['model_version']} · llm={data['llm_model']} · "
        f"prompt={data['prompt_version']} · request_id={data['request_id']}"
    )

    st.markdown("**Was this answer helpful?**")
    fb1, fb2 = st.columns(2)

    def _send(helpful: bool) -> None:
        try:
            feedback_response = httpx.post(
                f"{API_URL}/feedback",
                json={"request_id": data["request_id"], "helpful": helpful},
                headers=HEADERS,
                timeout=10,
            )
            feedback_response.raise_for_status()
            st.toast("Thanks - recorded.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"feedback failed: {exc}")

    if fb1.button("👍 Yes", use_container_width=True):
        _send(True)
    if fb2.button("👎 No", use_container_width=True):
        _send(False)
