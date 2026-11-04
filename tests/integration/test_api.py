"""Integration tests for FastAPI endpoints and HTMX partials."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_upload_non_pdf_rejected():
    files = {"file": ("test.png", b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR", "image/png")}
    resp = client.post("/api/upload", files=files)
    assert resp.status_code == 422


def test_upload_valid_pdf_and_export_roundtrip():
    pdf_path = Path("samples/test_dataset/text_based/01_text_bordered_debit_credit_carry_forward.pdf")
    if not pdf_path.is_file():
        pytest.skip("Test dataset not available")

    files = {"file": (pdf_path.name, pdf_path.read_bytes(), "application/pdf")}
    resp = client.post("/api/upload", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "session_id" in data
    session_id = data["session_id"]
    assert len(data["transactions"]) == 103

    # Check result page HTML
    page_resp = client.get(f"/result/{session_id}")
    assert page_resp.status_code == 200
    assert "Bank Statement" in page_resp.text or "HDFC" in page_resp.text

    # Check transactions partial HTML
    tx_resp = client.get(f"/result/{session_id}/transactions")
    assert tx_resp.status_code == 200
    assert "table" in tx_resp.text

    # Check inline edit PATCH
    patch_resp = client.patch(
        f"/transaction/{session_id}/row_1",
        data={"new_category": "Rent"}
    )
    assert patch_resp.status_code == 200
    assert "Rent" in patch_resp.text
    assert "recon-badge" in patch_resp.text

    # Check CSV export
    csv_resp = client.get(f"/export/{session_id}/csv")
    assert csv_resp.status_code == 200
    assert "text/csv" in csv_resp.headers["content-type"]

    # Check Excel export
    xlsx_resp = client.get(f"/export/{session_id}/xlsx")
    assert xlsx_resp.status_code == 200
    assert "openxmlformats" in xlsx_resp.headers["content-type"]

