import os
import requests
import streamlit as st
import pandas as pd
from core.schemas.utils import distance_to_score
import time
import json

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="Ask", layout="wide")
st.title("Ask (RAG)")

# Sidebar: API health
with st.sidebar:
    st.markdown("### API")
    try:
        hr = requests.get(f"{API_BASE}/health", timeout=5)
        st.success("API connected ✅" if hr.ok else "API error ❌")
        if hr.ok:
            st.caption(hr.json())
    except Exception:
        st.error("API not reachable")

#  Load documents for selection
docs = []
try:
    r = requests.get(f"{API_BASE}/documents", timeout=10)
    r.raise_for_status()
    docs = r.json().get("items", [])
except Exception as e:
    st.error(f"Could not load documents: {e}")

doc_options = [d["doc_id"] for d in docs] if docs else []

last_doc = st.session_state.get("last_ingested_doc_id")
if last_doc and last_doc in doc_options:
    if st.button("Use last uploaded doc"):
        st.session_state["prefill_doc_ids"] = [last_doc]


colA, colB = st.columns([2, 1])

with colA:
    # selected = st.multiselect("Choose guideline docs", options=doc_options)
    default_selected = st.session_state.pop("prefill_doc_ids", [])
    selected = st.multiselect(
        "Choose papers", options=doc_options, default=default_selected
    )
    question = st.text_area(
        "Question",
        height=90,
        placeholder="e.g., What are the key findings of the paper?",
    )

with colB:
    mode = st.radio("Mode", options=["rag", "no_rag"], horizontal=True)
    top_k = st.slider("top_k", 1, 10, 5)
    run = st.button(
        "Ask",
        type="primary",
        use_container_width=True,
        disabled=(len(question.strip()) < 3),
    )


if run:
    payload = {
        "question": question.strip(),
        "doc_ids": selected,
        "top_k": top_k,
        "mode": mode,
    }

    st.subheader("Answer")
    answer_placeholder = st.empty()
    full_answer = ""
    citations_raw = []

    start = time.perf_counter()
    with requests.post(
        f"{API_BASE}/ask/stream",
        json=payload,
        stream=True,
        timeout=90,
    ) as r:
        for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
            if "__CITATIONS__:" in chunk:
                parts = chunk.split("__CITATIONS__:")
                full_answer += parts[0]
                citations_raw = json.loads(parts[1])
            else:
                full_answer += chunk
            answer_placeholder.markdown(full_answer + "▌")

    answer_placeholder.markdown(full_answer)

    cits = [
        {
            "doc_id": c["meta"].get("doc_id", ""),
            "page": c["meta"].get("page", 0),
            "chunk_id": c["meta"].get("chunk_id", ""),
            "score": distance_to_score(c["distance"]),
            "snippet": c["text"][:350],
        }
        for c in citations_raw
    ]

    # Store for Evidence page
    st.session_state["last_ask_payload"] = payload
    st.session_state["last_ask"] = {
        "answer": full_answer,
        "citations": cits,
    }

    st.subheader("Citations")
    if not cits:
        st.info("No citations returned.")
    else:
        df = pd.DataFrame(
            [
                {
                    "doc_id": c["doc_id"],
                    "page": c["page"],
                    "chunk_id": c["chunk_id"],
                    "score": c["score"],
                }
                for c in cits
            ]
        )
        st.dataframe(df, use_container_width=True)

        for i, c in enumerate(cits, start=1):
            with st.expander(
                f"[{i}] {c['doc_id']} • page {c['page']} • score {c['score']:.3f}"
            ):
                st.write(c["snippet"])
    latency_ms = int((time.perf_counter() - start) * 1000)
    st.caption(f"latency: {latency_ms} ms | model: gpt-4o-mini")

    st.info(
        "Tip: Open the **Evidence** tab to view retrieved snippets as an audit trail."
    )
