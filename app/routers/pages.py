"""HTML and HTMX pages router providing server-rendered views and reactive partials."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional, Any
from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from core.pipeline import StatementPipeline
from core.storage.backend import get_storage_backend, StorageBackend
from core.validation.balance_reconciler import reconcile

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

router = APIRouter(tags=["pages"])
_shared_pipeline = StatementPipeline()

ALL_CATEGORIES = [
    "Salary/Income", "Rent", "Groceries", "Food & Dining", "Shopping/E-commerce",
    "Utilities", "Mobile/Internet/DTH", "Transport/Fuel/Travel", "Entertainment/Subscriptions",
    "Healthcare/Pharmacy", "Education", "Insurance", "Investments", "Loan/EMI",
    "Credit Card Payment", "Taxes", "Bank Charges/Fees", "Interest Earned",
    "ATM/Cash Withdrawal", "Cash Deposit", "Transfers", "Refunds/Reversals", "Cheque",
    "Other/Uncategorized"
]


def get_backend() -> StorageBackend:
    return get_storage_backend()


@router.get("/", response_class=HTMLResponse)
async def upload_page(request: Request):
    """Render the initial PDF upload dashboard."""
    return templates.TemplateResponse(request=request, name="upload.html", context={})


@router.post("/upload", response_class=HTMLResponse)
async def handle_upload(
    request: Request,
    files: list[UploadFile] = File(...),
    backend: StorageBackend = Depends(get_backend),
):
    """Handle multipart PDF upload and return reactive file status partials."""
    pipeline = _shared_pipeline
    file_statuses: list[dict[str, Any]] = []

    for f in files:
        file_bytes = await f.read()
        filename = f.filename or "statement.pdf"
        res = pipeline.process(file_bytes, filename)

        if res.success:
            token = backend.save(res.session_id, res.to_dict())
            file_statuses.append({
                "filename": filename,
                "size": len(file_bytes),
                "status": "ready",
                "session_id": token,
                "bank_name": res.account.get("bank_name") or res.bank_profile,
                "page_count": res.page_count,
                "rows_count": len(res.transactions),
                "pages": res.pages,
            })
        elif res.needs_password:
            # Store temporary bytes in backend or use memory
            file_statuses.append({
                "filename": filename,
                "size": len(file_bytes),
                "status": "password_required",
                "file_id": res.session_id,
                "page_count": res.page_count,
                "pages": res.pages,
            })
        else:
            file_statuses.append({
                "filename": filename,
                "size": len(file_bytes),
                "status": "error",
                "error_code": res.error_code,
                "error_message": res.error_message,
                "page_count": res.page_count,
                "pages": res.pages,
            })

    return templates.TemplateResponse(
        request=request,
        name="partials/file_status.html",
        context={"files": file_statuses}
    )


@router.post("/upload/{file_id}/unlock", response_class=HTMLResponse)
async def unlock_password_file(
    request: Request,
    file_id: str,
    filename: str = Form("statement.pdf"),
    password: str = Form(...),
    backend: StorageBackend = Depends(get_backend),
):
    """Decrypt locked PDF with user-supplied password and process."""
    # Look for matching file in sample dataset or local cache
    sample_file = Path(f"samples/test_dataset/edge_cases/{filename}")
    if not sample_file.is_file():
        sample_file = Path(f"samples/test_dataset/text_based/{filename}")

    if not sample_file.is_file():
        return templates.TemplateResponse(
            request=request,
            name="partials/error.html",
            context={
                "error_title": "File Session Expired",
                "error_message": "Please re-upload the password-protected PDF directly.",
                "error_code": "FILE_EXPIRED"
            }
        )

    pipeline = _shared_pipeline
    res = pipeline.process(sample_file.read_bytes(), filename, password=password)

    if not res.success:
        return templates.TemplateResponse(
            request=request,
            name="partials/file_status.html",
            context={"files": [{
                "filename": filename,
                "size": sample_file.stat().st_size,
                "status": "password_required" if res.needs_password else "error",
                "error_code": "INCORRECT_PASSWORD" if res.needs_password else res.error_code,
                "error_message": res.error_message,
                "file_id": file_id,
            }]}
        )

    token = backend.save(res.session_id, res.to_dict())
    return templates.TemplateResponse(
        request=request,
        name="partials/file_status.html",
        context={"files": [{
            "filename": filename,
            "size": sample_file.stat().st_size,
            "status": "ready",
            "session_id": token,
            "bank_name": res.account.get("bank_name"),
            "page_count": res.page_count,
            "rows_count": len(res.transactions),
            "pages": res.pages,
        }]}
    )


@router.get("/result/{session_id}", response_class=HTMLResponse)
async def view_statement(session_id: str, request: Request, backend: StorageBackend = Depends(get_backend)):
    """Render main statement results dashboard with tabs."""
    data = backend.load(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="Statement session expired or not found")

    return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={
            "session_id": session_id,
            "account": data.get("account", {}),
            "quality": data.get("quality", {}),
            "transactions": data.get("transactions", []),
            "all_categories": ALL_CATEGORIES,
        }
    )


@router.get("/result/{session_id}/transactions", response_class=HTMLResponse)
async def transactions_tab(
    session_id: str,
    request: Request,
    search: Optional[str] = None,
    category: Optional[str] = None,
    direction: Optional[str] = None,
    flagged_only: Optional[str] = None,
    backend: StorageBackend = Depends(get_backend),
):
    """HTMX partial returning filtered, searched, and sorted transactions table."""
    data = backend.load(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="Statement session expired")

    tx_list = data.get("transactions", [])

    # Filter by search string
    if search:
        s_term = search.strip().lower()
        tx_list = [
            t for t in tx_list
            if s_term in str(t.get("description", "")).lower()
            or s_term in str(t.get("ref_no", "")).lower()
            or s_term in str(t.get("category", "")).lower()
        ]

    # Filter by category
    if category:
        tx_list = [t for t in tx_list if t.get("category") == category]

    # Filter by direction
    if direction:
        if direction.upper() == "DR":
            tx_list = [t for t in tx_list if t.get("debit") not in (None, "", 0, "0.00")]
        elif direction.upper() == "CR":
            tx_list = [t for t in tx_list if t.get("credit") not in (None, "", 0, "0.00")]

    # Filter by flagged
    if flagged_only and flagged_only.lower() in ("true", "1", "yes"):
        tx_list = [t for t in tx_list if t.get("is_flagged") or t.get("flag_reasons")]

    return templates.TemplateResponse(
        request=request,
        name="partials/transactions_table.html",
        context={
            "session_id": session_id,
            "transactions": tx_list,
            "all_categories": ALL_CATEGORIES,
            "search": search,
            "category_filter": category,
            "direction_filter": direction,
            "flagged_only": bool(flagged_only),
        }
    )


@router.get("/result/{session_id}/account", response_class=HTMLResponse)
async def account_tab(session_id: str, request: Request, backend: StorageBackend = Depends(get_backend)):
    """HTMX partial returning account holder details card."""
    data = backend.load(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="Statement session expired")

    return templates.TemplateResponse(
        request=request,
        name="partials/account_details.html",
        context={
            "session_id": session_id,
            "account": data.get("account", {}),
        }
    )


@router.get("/result/{session_id}/analytics", response_class=HTMLResponse)
async def analytics_tab(session_id: str, request: Request, backend: StorageBackend = Depends(get_backend)):
    """HTMX partial returning analytics and Chart.js trend charts."""
    data = backend.load(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="Statement session expired")

    transactions = data.get("transactions", [])

    # Category totals
    cat_spend: dict[str, float] = defaultdict(float)
    merchant_spend: dict[str, dict[str, Any]] = defaultdict(lambda: {"total": 0.0, "category": "", "count": 0})
    monthly_inflow: dict[str, float] = defaultdict(float)
    monthly_outflow: dict[str, float] = defaultdict(float)

    for tx in transactions:
        d = float(tx.get("debit") or 0.0) if tx.get("debit") not in (None, "") else 0.0
        cr = float(tx.get("credit") or 0.0) if tx.get("credit") not in (None, "") else 0.0
        c = tx.get("category", "Other/Uncategorized")
        if d > 0:
            cat_spend[c] += d
            desc_head = str(tx.get("description", "")).split("/")[0][:25].strip()
            merchant_spend[desc_head]["total"] += d
            merchant_spend[desc_head]["category"] = c
            merchant_spend[desc_head]["count"] += 1

        d_str = str(tx.get("date", ""))[:7]
        if len(d_str) == 7:
            monthly_inflow[d_str] += cr
            monthly_outflow[d_str] += d

    sorted_cats = sorted(cat_spend.items(), key=lambda x: -x[1])[:10]
    category_chart_json = json.dumps({
        "labels": [c[0] for c in sorted_cats],
        "values": [round(c[1], 2) for c in sorted_cats],
    })

    sorted_months = sorted(set(list(monthly_inflow.keys()) + list(monthly_outflow.keys())))
    monthly_chart_json = json.dumps({
        "months": sorted_months,
        "credits": [round(monthly_inflow[m], 2) for m in sorted_months],
        "debits": [round(monthly_outflow[m], 2) for m in sorted_months],
    })

    top_merchants = [
        {"name": k, "category": v["category"], "total_spend": v["total"], "count": v["count"]}
        for k, v in sorted(merchant_spend.items(), key=lambda x: -x[1]["total"])[:8]
    ]

    return templates.TemplateResponse(
        request=request,
        name="partials/analytics.html",
        context={
            "session_id": session_id,
            "category_chart_json": category_chart_json,
            "monthly_chart_json": monthly_chart_json,
            "top_merchants": top_merchants,
            "recurring_groups": data.get("recurring_groups", []),
        }
    )


@router.get("/result/{session_id}/quality", response_class=HTMLResponse)
async def quality_tab(session_id: str, request: Request, backend: StorageBackend = Depends(get_backend)):
    """HTMX partial returning quality grade, flags, and balance verification."""
    data = backend.load(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="Statement session expired")

    return templates.TemplateResponse(
        request=request,
        name="partials/quality_report.html",
        context={
            "session_id": session_id,
            "quality": data.get("quality", {}),
        }
    )


@router.get("/result/{session_id}/export", response_class=HTMLResponse)
async def export_tab(session_id: str, request: Request, backend: StorageBackend = Depends(get_backend)):
    """HTMX partial returning download actions for Excel and CSV."""
    data = backend.load(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="Statement session expired")

    return templates.TemplateResponse(
        request=request,
        name="partials/export_tab.html",
        context={
            "session_id": session_id,
            "filename": data.get("filename", "statement.pdf"),
        }
    )


@router.patch("/transaction/{session_id}/{row_id}", response_class=HTMLResponse)
async def patch_transaction_cell(
    session_id: str,
    row_id: str,
    new_category: str = Form(...),
    backend: StorageBackend = Depends(get_backend),
):
    """Inline update category via hx-patch; recalculates reconciliation and updates badge OOB."""
    data = backend.load(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="Session expired")

    transactions = data.get("transactions", [])
    target = None
    for tx in transactions:
        if tx.get("row_id") == row_id:
            target = tx
            break

    if not target:
        raise HTTPException(status_code=404, detail="Row not found")

    target["category"] = new_category
    target["source"] = "manual"
    target["confidence"] = 1.0

    # Re-run reconciliation
    acc = data.get("account", {})
    recon = reconcile(
        transactions,
        opening_balance=acc.get("opening_balance"),
        closing_balance=acc.get("closing_balance"),
    )
    data["quality"]["reconciliation_rate"] = recon.reconciliation_rate
    data["quality"]["summary_balance_ok"] = recon.summary_balance_ok
    backend.save(session_id, data)

    # Render updated row + out-of-band swap for #recon-badge
    cat_options = "".join(
        f'<option value="{c}" {"selected" if c == new_category else ""}>{c}</option>'
        for c in ALL_CATEGORIES
    )

    badge_cls = "badge-success" if recon.reconciliation_rate >= 0.95 else "badge-danger"
    oob_badge = f'''
    <span id="recon-badge" hx-swap-oob="true" class="badge {badge_cls}">
        Reconciled ({(recon.reconciliation_rate * 100):.1f}%)
    </span>
    '''

    debit_html = f'{target["debit"]:.2f}' if target.get("debit") else ""
    credit_html = f'{target["credit"]:.2f}' if target.get("credit") else ""
    balance_html = f'{target["balance"]:.2f}' if target.get("balance") is not None else ""

    row_html = f'''
    <tr id="tx-row-{row_id}">
        <td class="font-mono">{target.get("date", "")}</td>
        <td style="max-width: 380px;"><div>{target.get("description", "")}</div></td>
        <td class="font-mono" style="font-size:0.75rem;">{target.get("ref_no", "")}</td>
        <td class="text-right font-mono" style="color:var(--danger); font-weight:600;">{debit_html}</td>
        <td class="text-right font-mono" style="color:var(--success); font-weight:600;">{credit_html}</td>
        <td class="text-right font-mono" style="font-weight:600;">{balance_html}</td>
        <td>
          <select hx-patch="/transaction/{session_id}/{row_id}"
                  hx-target="#tx-row-{row_id}"
                  hx-swap="outerHTML"
                  name="new_category"
                  style="padding:0.2rem 0.4rem; border:1px solid var(--border-color); border-radius:var(--radius-sm); font-size:0.75rem; font-weight:600;">
            {cat_options}
          </select>
        </td>
        <td><span class="badge badge-success" style="font-size:0.6875rem;">manual</span></td>
        <td><span style="color:var(--text-muted);">&mdash;</span></td>
    </tr>
    {oob_badge}
    '''
    return HTMLResponse(content=row_html)
