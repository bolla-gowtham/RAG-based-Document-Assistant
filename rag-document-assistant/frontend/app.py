"""Simple Streamlit chat UI that talks to the FastAPI backend."""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Document Assistant", page_icon="📄", layout="wide")
st.title("📄 RAG Document Assistant")
st.caption("Ask questions grounded in your own documents — every answer is cited.")

with st.sidebar:
    st.header("Ingest a document")
    uploaded = st.file_uploader("Upload a .txt, .md, or .pdf file", type=["txt", "md", "pdf"])
    if uploaded and st.button("Ingest"):
        with st.spinner("Chunking, embedding, and indexing..."):
            files = {"file": (uploaded.name, uploaded.getvalue())}
            try:
                resp = requests.post(f"{API_URL}/ingest", files=files, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                st.success(f"Added {data['chunks_added']} chunks from '{uploaded.name}'.")
            except requests.RequestException as e:
                st.error(f"Ingestion failed: {e}")

    st.divider()
    st.caption(f"Backend: {API_URL}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"**[{s['id']}] {s['source']}** (score: {s['score']:.3f})")
                    st.text(s["text"][:400] + ("..." if len(s["text"]) > 400 else ""))

if question := st.chat_input("Ask a question about your ingested documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            try:
                resp = requests.post(f"{API_URL}/query", json={"question": question}, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                st.markdown(data["answer"])
                if data.get("sources"):
                    with st.expander("Sources"):
                        for s in data["sources"]:
                            st.markdown(f"**[{s['id']}] {s['source']}** (score: {s['score']:.3f})")
                            st.text(s["text"][:400] + ("..." if len(s["text"]) > 400 else ""))
                st.session_state.messages.append(
                    {"role": "assistant", "content": data["answer"], "sources": data.get("sources")}
                )
            except requests.RequestException as e:
                error_msg = f"Request failed: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
