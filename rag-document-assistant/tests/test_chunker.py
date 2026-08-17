from src.ingestion.chunker import chunk_document, _split_text, _add_overlap
from src.ingestion.loader import RawDocument


def test_split_text_respects_chunk_size():
    text = "word " * 500  # long text
    pieces = _split_text(text, chunk_size=100, separators=[" ", ""])
    assert all(len(p) <= 100 for p in pieces)


def test_split_text_short_text_returns_single_chunk():
    text = "short text"
    pieces = _split_text(text, chunk_size=100, separators=["\n\n", "\n", " ", ""])
    assert pieces == ["short text"]


def test_add_overlap_prepends_previous_tail():
    chunks = ["aaaa", "bbbb", "cccc"]
    overlapped = _add_overlap(chunks, overlap=2)
    assert overlapped[0] == "aaaa"
    assert overlapped[1] == "aa" + "bbbb"
    assert overlapped[2] == "bb" + "cccc"


def test_add_overlap_noop_when_overlap_zero():
    chunks = ["aaaa", "bbbb"]
    assert _add_overlap(chunks, overlap=0) == chunks


def test_chunk_document_produces_chunks_with_metadata():
    doc = RawDocument(text="Paragraph one.\n\nParagraph two.\n\nParagraph three.", source="test.txt")
    chunks = chunk_document(doc, chunk_size=1000, chunk_overlap=0)
    assert len(chunks) >= 1
    for c in chunks:
        assert c.source == "test.txt"
        assert c.metadata["source"] == "test.txt"
        assert c.id  # non-empty uuid


def test_chunk_document_empty_text_returns_no_chunks():
    doc = RawDocument(text="   ", source="empty.txt")
    chunks = chunk_document(doc)
    assert chunks == []


def test_chunk_document_long_text_produces_multiple_chunks():
    doc = RawDocument(text="Sentence number %d. " % 0 * 0 + ". ".join(f"Sentence {i}" for i in range(200)), source="long.txt")
    chunks = chunk_document(doc, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    # chunk_index should be sequential starting at 0
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
