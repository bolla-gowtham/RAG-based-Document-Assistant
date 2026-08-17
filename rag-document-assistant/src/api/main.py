"""FastAPI service exposing the RAG pipeline: /ingest and /query."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import HealthResponse, IngestResponse, QueryRequest, QueryResponse
from src.pipeline import RAGPipeline

app = FastAPI(
    title="RAG Document Assistant",
    description="Ask questions over your own documents with grounded, cited answers.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = RAGPipeline()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile) -> IngestResponse:
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".txt", ".md", ".pdf"}:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        n_chunks = pipeline.ingest_file(tmp_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return IngestResponse(chunks_added=n_chunks, message=f"Ingested '{file.filename}' successfully.")


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    try:
        result = pipeline.query(request.question)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return QueryResponse(**result)
