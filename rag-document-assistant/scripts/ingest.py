#!/usr/bin/env python
"""CLI: ingest all documents in a directory into the vector store.

Usage:
    python scripts/ingest.py --source data/sample_docs
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import RAGPipeline  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG vector store.")
    parser.add_argument("--source", required=True, help="Directory of .txt/.md/.pdf files to ingest")
    args = parser.parse_args()

    print(f"Ingesting documents from: {args.source}")
    start = time.time()

    pipeline = RAGPipeline()
    n_chunks = pipeline.ingest_directory(args.source)

    elapsed = time.time() - start
    print(f"Done. Added {n_chunks} chunks in {elapsed:.1f}s.")
    print(f"Vector store saved to: {pipeline.store and 'data/vector_store'}")


if __name__ == "__main__":
    main()
