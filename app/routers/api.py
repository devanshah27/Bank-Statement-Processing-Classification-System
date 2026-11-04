"""JSON API router providing REST endpoints for bank statement processing and classification."""

from __future__ import annotations

import csv
import datetime
import logging
from pathlib import Path
from typing import Optional, Any
from pydantic import BaseModel, Field
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends

from core.pipeline import StatementPipeline
from core.storage.backend import get_storage_backend, StorageBackend
from core.validation.balance_reconciler import reconcile

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["api"])
FEEDBACK_FILE = Path("data/user_feedback.csv")
_shared_pipeline = StatementPipeline()


def get_backend() -> StorageBackend:
    return get_storage_backend()


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    version: str
    ocr_engine: str
    deploy_target: str


class PatchTransactionRequest(BaseModel):
    new_category: Optional[str] = None
    description: Optional[str] = None
    debit: Optional[float] = None
    credit: Optional[float] = None


class RetrainHintRequest(BaseModel):
    row_id: str
    old_category: str
    new_category: str
    description: str
    amount: Optional[float] = 0.0
    direction: Optional[str] = "DR"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint exposing runtime configuration."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        ocr_engine=_shared_pipeline.ocr_engine_name or "rapid",
        deploy_target="local",
    )


@router.post("/upload")
async def api_upload_statement(
    file: UploadFile = File(...),
    password: Optional[str] = Form(None),
    backend: StorageBackend = Depends(get_backend),
):
    """Upload and process a single bank statement PDF, returning structured JSON."""
    file_bytes = await file.read()
    result = _shared_pipeline.process(file_bytes, file.filename or "statement.pdf", password=password)

    if not result.success:
        if result.needs_password:
            raise HTTPException(status_code=401, detail={"error": "PASSWORD_REQUIRED", "message": result.error_message})
        raise HTTPException(status_code=422, detail={"error": result.error_code, "message": result.error_message})

    # Save to storage backend
    token = backend.save(result.session_id, result.to_dict())
    res_dict = result.to_dict()
    res_dict["session_id"] = token
    res_dict["session_token"] = token
    return res_dict


@router.get("/statement/{session_id}")
async def get_statement(session_id: str, backend: StorageBackend = Depends(get_backend)):
    """Retrieve full statement data by session ID."""
    data = backend.load(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="Statement not found or session expired")
    return data


@router.patch("/statement/{session_id}/transaction/{row_id}")
async def patch_transaction(
    session_id: str,
    row_id: str,
    payload: PatchTransactionRequest,
    backend: StorageBackend = Depends(get_backend),
):
    """Inline update a transaction row, re-running reconciliation automatically."""
    data = backend.load(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="Statement session expired")

    transactions = data.get("transactions", [])
    target_row = None
    for r in transactions:
        if r.get("row_id") == row_id:
            target_row = r
            break

    if not target_row:
        raise HTTPException(status_code=404, detail="Transaction row not found")

    # Apply updates
    if payload.new_category is not None:
        target_row["category"] = payload.new_category
        target_row["source"] = "manual_correction"
    if payload.description is not None:
        target_row["description"] = payload.description
    if payload.debit is not None:
        target_row["debit"] = payload.debit
    if payload.credit is not None:
        target_row["credit"] = payload.credit

    # Re-reconcile
    acc = data.get("account", {})
    recon = reconcile(
        transactions,
        opening_balance=acc.get("opening_balance"),
        closing_balance=acc.get("closing_balance"),
    )
    data["quality"]["reconciliation_rate"] = recon.reconciliation_rate
    data["quality"]["summary_balance_ok"] = recon.summary_balance_ok

    backend.save(session_id, data)
    return {"status": "updated", "row": target_row, "reconciliation_rate": recon.reconciliation_rate}


@router.post("/statement/{session_id}/retrain-hint")
async def save_retrain_hint(session_id: str, hint: RetrainHintRequest):
    """Save user manual categorization feedback for future model retraining."""
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    is_new = not FEEDBACK_FILE.exists()

    with open(FEEDBACK_FILE, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "row_id", "description", "direction", "amount", "old_category", "new_category"])
        writer.writerow([
            datetime.datetime.now().isoformat(),
            hint.row_id,
            hint.description,
            hint.direction,
            hint.amount,
            hint.old_category,
            hint.new_category,
        ])

    return {"status": "saved", "message": "Feedback captured for next training cycle"}
