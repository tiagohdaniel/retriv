"""Tests for POST /analyze — analytics pipeline.

Strategy:
- PandasAnalyzer is tested directly (unit) for intent inference and computation
- The HTTP endpoint is tested with a mock LLM and a real PandasAnalyzer
- No Chroma, no embeddings needed — analytics is independent of the RAG pipeline
"""
import base64
import io
import csv
import json
import uuid

import pytest
import chromadb
from unittest.mock import AsyncMock
from fastapi.testclient import TestClient

from app.main import app
from app.dependencies import get_analyzer, get_llm_client
from app.core.backends.pandas_analyzer import PandasAnalyzer, _infer_op, _find_column, _find_group_column


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _csv_b64(rows: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return base64.b64encode(buf.getvalue().encode()).decode()


def _mock_llm():
    mock = AsyncMock()
    mock.generate = AsyncMock(return_value={
        "answer": "LLM formatted answer.",
        "tokens_used": 10,
        "model": "mock-model",
    })
    return mock


@pytest.fixture
def analyze_client():
    analyzer = PandasAnalyzer()
    llm = _mock_llm()

    app.dependency_overrides = {
        get_analyzer: lambda: analyzer,
        get_llm_client: lambda: llm,
    }
    with TestClient(app) as c:
        yield c
    app.dependency_overrides = {}


# ---------------------------------------------------------------------------
# Unit: intent inference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question, expected_op", [
    ("Qual o total de vendas?", "sum"),
    ("What is the sum of revenue?", "sum"),
    ("Qual a média de salários?", "mean"),
    ("Average salary per department", "mean"),
    ("Qual o maior valor?", "max"),
    ("Qual o menor preço?", "min"),
    ("Quantos registros existem?", "count"),
    ("How many entries are there?", "count"),
    ("Liste os primeiros 5", "list"),
    ("Show top 10 records", "list"),
    ("Descreva os dados", "describe"),
])
def test_infer_op(question, expected_op):
    assert _infer_op(question) == expected_op


def test_find_column():
    cols = ["vendas", "nome", "data", "valor"]
    assert _find_column("qual o total de vendas?", cols) == "vendas"
    assert _find_column("qual o total de VALOR?", cols) == "valor"
    assert _find_column("quantos registros?", cols) is None


def test_find_group_column():
    cols = ["vendas", "regiao", "produto"]
    assert _find_group_column("total de vendas por regiao", cols) == "regiao"
    assert _find_group_column("vendas agrupado por produto", cols) == "produto"
    assert _find_group_column("total de vendas", cols) is None


# ---------------------------------------------------------------------------
# Unit: PandasAnalyzer
# ---------------------------------------------------------------------------

SAMPLE_ROWS = [
    {"produto": "A", "valor": 100.0, "regiao": "Sul"},
    {"produto": "B", "valor": 200.0, "regiao": "Norte"},
    {"produto": "C", "valor": 150.0, "regiao": "Sul"},
    {"produto": "D", "valor": 50.0, "regiao": "Norte"},
]


def test_analyzer_sum():
    content = _csv_b64(SAMPLE_ROWS).encode()
    file_bytes = base64.b64decode(_csv_b64(SAMPLE_ROWS))
    analyzer = PandasAnalyzer()
    result = analyzer.analyze("Qual o total de valor?", file_bytes, "csv")
    assert result.result == pytest.approx(500.0)
    assert "sum" in result.summary.lower() or "total" in result.summary.lower()


def test_analyzer_mean():
    file_bytes = base64.b64decode(_csv_b64(SAMPLE_ROWS))
    analyzer = PandasAnalyzer()
    result = analyzer.analyze("Qual a média de valor?", file_bytes, "csv")
    assert result.result == pytest.approx(125.0)


def test_analyzer_max():
    file_bytes = base64.b64decode(_csv_b64(SAMPLE_ROWS))
    analyzer = PandasAnalyzer()
    result = analyzer.analyze("Qual o maior valor?", file_bytes, "csv")
    assert result.result == pytest.approx(200.0)


def test_analyzer_min():
    file_bytes = base64.b64decode(_csv_b64(SAMPLE_ROWS))
    analyzer = PandasAnalyzer()
    result = analyzer.analyze("Qual o menor valor?", file_bytes, "csv")
    assert result.result == pytest.approx(50.0)


def test_analyzer_count_total():
    file_bytes = base64.b64decode(_csv_b64(SAMPLE_ROWS))
    analyzer = PandasAnalyzer()
    result = analyzer.analyze("Quantos registros existem?", file_bytes, "csv")
    assert result.result == 4


def test_analyzer_count_by_group():
    file_bytes = base64.b64decode(_csv_b64(SAMPLE_ROWS))
    analyzer = PandasAnalyzer()
    result = analyzer.analyze("Quantos registros por regiao?", file_bytes, "csv")
    assert isinstance(result.result, dict)
    assert result.result.get("Sul") == 2
    assert result.result.get("Norte") == 2


def test_analyzer_sum_by_group():
    file_bytes = base64.b64decode(_csv_b64(SAMPLE_ROWS))
    analyzer = PandasAnalyzer()
    result = analyzer.analyze("Total de valor por regiao?", file_bytes, "csv")
    assert isinstance(result.result, dict)
    assert result.result.get("Sul") == pytest.approx(250.0)
    assert result.result.get("Norte") == pytest.approx(250.0)


def test_analyzer_list():
    file_bytes = base64.b64decode(_csv_b64(SAMPLE_ROWS))
    analyzer = PandasAnalyzer()
    result = analyzer.analyze("Liste os primeiros 3", file_bytes, "csv")
    assert isinstance(result.result, list)
    assert len(result.result) == 3


def test_analyzer_max_rows_exceeded():
    import pandas as pd
    rows = [{"valor": i} for i in range(10)]
    file_bytes = base64.b64decode(_csv_b64(rows))
    analyzer = PandasAnalyzer(max_rows=5)
    with pytest.raises(ValueError, match="limit is"):
        analyzer.analyze("total de valor", file_bytes, "csv")


def test_analyzer_unsupported_type():
    analyzer = PandasAnalyzer()
    with pytest.raises(ValueError, match="Unsupported file type"):
        analyzer.analyze("test", b"data", "pdf")


def test_analyzer_json():
    data = json.dumps(SAMPLE_ROWS).encode()
    file_bytes = data
    analyzer = PandasAnalyzer()
    result = analyzer.analyze("Qual o total de valor?", file_bytes, "json")
    assert result.result == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# Integration: HTTP endpoint
# ---------------------------------------------------------------------------

def test_analyze_endpoint_success(analyze_client):
    payload = {
        "question": "Qual o total de valor?",
        "source": {
            "type": "csv",
            "content": _csv_b64(SAMPLE_ROWS),
        },
    }
    resp = analyze_client.post("/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert data["answer"] == "LLM formatted answer."
    assert data["computation"]["result"] == pytest.approx(500.0)
    assert data["computation"]["row_count"] == 4


def test_analyze_endpoint_invalid_base64(analyze_client):
    payload = {
        "question": "Qual o total de valor?",
        "source": {
            "type": "csv",
            "content": "not-valid-base64!!!",
        },
    }
    resp = analyze_client.post("/analyze", json=payload)
    assert resp.status_code == 422
    assert "base64" in resp.json()["detail"].lower()


def test_analyze_endpoint_file_too_large(analyze_client):
    # Build a content string that exceeds the size check
    big = "A" * (14 * 1024 * 1024)  # ~14 MB encoded → ~10.5 MB decoded (exceeds 10 MB limit)
    payload = {
        "question": "quantos registros existem?",
        "source": {"type": "csv", "content": big},
    }
    resp = analyze_client.post("/analyze", json=payload)
    assert resp.status_code == 413


def test_analyze_endpoint_question_too_short(analyze_client):
    payload = {
        "question": "hi",
        "source": {"type": "csv", "content": _csv_b64(SAMPLE_ROWS)},
    }
    resp = analyze_client.post("/analyze", json=payload)
    assert resp.status_code == 422
