import os
import requests
import streamlit as st
import pandas as pd

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="Summarize", layout="wide")
st.title("Summarize")

# Load documents (DocList: {"items": [...]})
docs = []
try:
    r = requests.get(f"{API_BASE}/documents", timeout=10)
    r.raise_for_status()
    docs = r.json().get("items", [])
except Exception as e:
    st.error(f"Could not load documents: {e}")

doc_options = [d["doc_id"] for d in docs] if docs else []
# selected = st.multiselect("Choose guideline docs", options=doc_options)
label_map = {
    d["doc_id"]: f"{d['doc_id']} — {d.get('title') or 'Untitled'}" for d in docs
}

selected = st.multiselect(
    "Choose papers",
    options=list(label_map.keys()),
    format_func=lambda k: label_map[k],
)


style = st.radio(
    "Style", ["tldr",  "methods", "key_findings", "materials_properties"], horizontal=True
)

if st.button("Summarize", type="primary"):
    payload = {"doc_ids": selected, "style": style}
    r = requests.post(f"{API_BASE}/summarize", json=payload, timeout=90)

    if not r.ok:
        st.error(f"API error {r.status_code}")
        st.code(r.text[:2000])
    else:
        out = r.json()
        st.session_state["last_summary"] = out
        st.session_state["last_summary_payload"] = payload

        with st.expander("Debug (request + raw response)"):
            st.json(payload)
            st.json(out)

        st.subheader("Summary")
        st.write(out["summary"])

        cits = out.get("citations", [])
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
                        "score": round(c["score"], 3),
                    }
                    for c in cits
                ]
            )
            # st.dataframe(df, use_container_width=True)
            st.dataframe(df, width="stretch")

            for i, c in enumerate(cits, start=1):
                with st.expander(
                    f"[{i}] {c['doc_id']} • page {c['page']} • score {c['score']:.3f}"
                ):
                    st.write(c["snippet"])

        meta = out.get("meta", {})
        st.caption(
            f"request_id: {meta.get('request_id')} | latency: {meta.get('latency_ms')} ms | model: {meta.get('model')}"
        )
