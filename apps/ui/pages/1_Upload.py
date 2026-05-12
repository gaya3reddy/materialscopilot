import os
import time
import requests
import streamlit as st
import pandas as pd

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="Upload", layout="wide")
st.title("Upload & Ingest")

st.warning("Materials science papers only. No proprietary or confidential data. Research use only.")

uploaded = st.file_uploader("Upload a materials science paper (PDF)", type=["pdf"])
col1, col2, col3 = st.columns(3)
title = col1.text_input("Title (optional)")
source = col2.text_input("Source (optional)")
category = col3.text_input("Category (optional)")
doc_id = st.text_input("doc_id (optional, leave empty to auto-generate)")

if st.button("Ingest", type="primary", disabled=(uploaded is None)):
    files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
    data = {"doc_id": doc_id, "title": title, "source": source, "category": category}

    # POST /ingest — returns immediately with job_id
    r = requests.post(f"{API_BASE}/ingest", files=files, data=data, timeout=60)

    if r.status_code >= 400:
        st.error(r.text)
    else:
        out = r.json()

        # Duplicate detected — no job needed
        if out.get("deduped"):
            st.success(f"Duplicate detected — reusing doc_id: {out['doc_id']}")
            st.session_state["last_ingested_doc_id"] = out["doc_id"]

        else:
            job_id = out["job_id"]
            doc_id_created = out["doc_id"]
            st.session_state["last_ingested_doc_id"] = doc_id_created

            st.info(f"Ingest started — doc_id: `{doc_id_created}`")

            # Progress bar + polling loop
            progress_bar = st.progress(0, text="Starting ingest...")
            status_placeholder = st.empty()

            while True:
                time.sleep(2)
                status_r = requests.get(
                    f"{API_BASE}/ingest/status/{job_id}", timeout=10
                )
                if not status_r.ok:
                    st.error("Could not fetch job status.")
                    break

                job = status_r.json()
                status = job["status"]
                progress = job["progress"]
                indexed = job["indexed_chunks"]
                total = job["total_chunks"]

                if total > 0:
                    progress_bar.progress(
                        progress, text=f"Embedding chunks... {indexed}/{total}"
                    )
                else:
                    progress_bar.progress(0, text="Processing PDF...")

                if status == "done":
                    progress_bar.progress(100, text="Done!")
                    status_placeholder.success(
                        f"Ingested ✅ (pages: {job['pages']}, "
                        f"chunks: {job['indexed_chunks']})"
                    )
                    break

                elif status == "error":
                    progress_bar.empty()
                    status_placeholder.error(f"Ingest failed: {job['error']}")
                    break

st.markdown("---")
st.subheader("Available docs")

try:
    r = requests.get(f"{API_BASE}/documents", timeout=10)
    if not r.ok:
        st.error(f"API error {r.status_code}")
        st.code(r.text[:1500])
    else:
        data = r.json()
        items = data.get("items", [])
        st.dataframe(pd.DataFrame(items))
        if not items:
            st.info("No documents ingested yet.")
        else:
            st.json(items)
except Exception as e:
    st.error(f"Could not load available docs: {e}")
