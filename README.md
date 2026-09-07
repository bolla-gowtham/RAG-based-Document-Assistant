# RAG Document Assistant

A production-style Retrieval-Augmented Generation (RAG) system that lets you ask natural-language questions over your own documents and get **grounded, cited answers** — built from first principles (no LangChain/LlamaIndex black boxes) so every stage of the pipeline is transparent & testable.

> Ingest → Chunk → Embed → Retrieve → Generate → Evaluate — each stage is a swappable, independently tested module.

## Why this project exists

Most "RAG projects" on GitHub are a 40-line notebook wrapping `langchain.load_qa_chain`. This one is built like a real service:

- **Modular pipeline** — ingestion, chunking, embedding, retrieval, generation, and evaluation are separate packages with clean interfaces, so any component (embedding model, vector store, LLM provider) can be swapped without touching the rest.
- **Provider-agnostic generation** — works with OpenAI or Anthropic models behind one interface.
- **Grounded answers with citations** — every answer references the source chunk(s) it was built from, and the system explicitly says "I don't know" when retrieval confidence is low, instead of hallucinating.
- **Built-in evaluation harness** — retrieval hit-rate/MRR and LLM-as-judge faithfulness/relevance scoring, run automatically against a labeled QA set so you can quantify pipeline quality, not just eyeball it.
- **Actually deployable** — FastAPI backend, Streamlit UI, Dockerized, CI pipeline that runs tests on every push.

## Architecture

```
e:\RAG-based-Document-Assistant\architecture.jpg
```

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Chunking | Custom recursive character splitter | No hidden behavior; tunable overlap/boundaries |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) | Free, local, no API key needed to index docs |
| Vector store | FAISS | Fast, disk-persistable, industry standard |
| Generation | OpenAI / Anthropic (pluggable) | Provider-agnostic interface |
| API | FastAPI | Async, typed, auto-docs |
| UI | Streamlit | Fast to ship, good enough for a demo |
| Evaluation | Custom retrieval metrics + LLM-as-judge | Quantifies quality instead of "looks good to me" |
| Packaging | Docker + docker-compose | One command to run the whole stack |
| CI | GitHub Actions | Tests run on every push/PR |


## Project structure

```
rag-document-assistant/
├── src/
│   ├── config.py              # centralized settings via env vars
│   ├── ingestion/              # load + chunk documents
│   ├── embeddings/             # embedding model wrapper
│   ├── vectorstore/            # FAISS wrapper (add/search/persist)
│   ├── retrieval/              # query embedding + top-k + MMR
│   ├── generation/             # LLM prompt construction + citation formatting
│   ├── evaluation/             # retrieval + generation quality metrics
│   ├── api/                    # FastAPI app
│   └── pipeline.py             # orchestrates the full RAG flow
├── frontend/app.py             # Streamlit chat UI
├── scripts/                    # ingest.py, evaluate.py CLIs
├── tests/                      # pytest unit + API tests
├── data/                       # sample docs + eval QA set
├── Dockerfile / docker-compose.yml
└── .github/workflows/ci.yml
```

## What this demonstrates (for recruiters/interviewers)

- Understanding of RAG internals beyond calling a library function (chunking strategy, embedding trade-offs, retrieval ranking, prompt grounding, hallucination mitigation).
- Software engineering fundamentals: clean module boundaries, dependency injection for swappable providers, typed interfaces, unit + integration tests.
- Production awareness: containerization, CI, persisted state, config via environment, structured logging, graceful failure ("I don't know" instead of hallucinating).
- Evaluation-driven development: the system reports its own quality metrics rather than relying on manual spot checks.

## Possible extensions

- Swap FAISS for a managed vector DB (Pinecone/Weaviate/Qdrant) for multi-tenant scale.
- Add hybrid search (BM25 + dense retrieval).
- Add re-ranking with a cross-encoder.
- Stream tokens from the LLM to the UI.
- Add auth + per-user document namespaces.

## License

MIT
