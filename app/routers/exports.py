"""Exports router providing Excel (.xlsx) and CSV (.csv) streaming downloads."""

from __future__ import annotations

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import Response

from core.storage.backend import get_storage_backend, StorageBackend
from core.export.excel_exporter import export_to_excel
from core.export.csv_exporter import export_to_csv

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/export", tags=["exports"])


def get_backend() -> StorageBackend:
    return get_storage_backend()


@router.get("/{session_id}/xlsx")
async def download_excel(session_id: str, backend: StorageBackend = Depends(get_backend)):
    """Download statement dataset as a formatted multi-sheet Excel (.xlsx) file."""
    data = backend.load(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="Statement session expired or not found")

    transactions = data.get("transactions", [])
    account = data.get("account", {})
    quality = data.get("quality", {})
    recurring = data.get("recurring_groups", [])
    filename = data.get("filename", "statement.pdf")

    xlsx_bytes = export_to_excel(
        transactions=transactions,
        account_details=account,
        quality_report=quality,
        recurring_groups=recurring,
        filename=filename,
    )

    clean_name = filename.rsplit(".", 1)[0]
    return Response(
        content=xlsx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{clean_name}_processed.xlsx"'}
    )


@router.get("/{session_id}/csv")
async def download_csv(session_id: str, backend: StorageBackend = Depends(get_backend)):
    """Download transactions as a UTF-8-SIG encoded CSV."""
    data = backend.load(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="Statement session expired or not found")

    transactions = data.get("transactions", [])
    filename = data.get("filename", "statement.pdf")

    csv_bytes = export_to_csv(transactions)
    clean_name = filename.rsplit(".", 1)[0]
    return Response(
        content=csv_bytes,
        media_type="text/csv; charset=utf-8-sig",
        headers={"Content-Disposition": f'attachment; filename="{clean_name}_transactions.csv"'}
    )

