from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@patch("src.api.main.pipeline")
def test_query_endpoint_returns_answer_and_sources(mock_pipeline):
    mock_pipeline.query.return_value = {
        "answer": "The stipend is $500 [1].",
        "sources": [{"id": 1, "source": "company_handbook.txt", "score": 0.87, "text": "...$500 stipend..."}],
    }
    resp = client.post("/query", json={"question": "What is the home office stipend?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "$500" in data["answer"]
    assert data["sources"][0]["source"] == "company_handbook.txt"


@patch("src.api.main.pipeline")
def test_query_endpoint_handles_missing_vector_store(mock_pipeline):
    mock_pipeline.query.side_effect = RuntimeError("No vector store found.")
    resp = client.post("/query", json={"question": "anything"})
    assert resp.status_code == 400
    assert "No vector store found" in resp.json()["detail"]


def test_query_endpoint_rejects_empty_question():
    resp = client.post("/query", json={"question": ""})
    assert resp.status_code == 422


@patch("src.api.main.pipeline")
def test_ingest_endpoint_rejects_unsupported_file_type(mock_pipeline):
    resp = client.post(
        "/ingest", files={"file": ("data.csv", b"a,b,c\n1,2,3", "text/csv")}
    )
    assert resp.status_code == 400
    assert "Unsupported file type" in resp.json()["detail"]


@patch("src.api.main.pipeline")
def test_ingest_endpoint_success(mock_pipeline):
    mock_pipeline.ingest_file.return_value = 5
    resp = client.post(
        "/ingest", files={"file": ("notes.txt", b"some plain text content", "text/plain")}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["chunks_added"] == 5
