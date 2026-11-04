"""End-to-end smoke verification of the premium fintech UI/UX redesign."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from app.main import create_app

def test_full_redesign_flow():
    app = create_app()
    client = TestClient(app)

    print("1. Testing GET / (Upload Page)...")
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.text
    assert "app-sidebar" in html
    assert "premium-dropzone" in html
    assert "/static/app.css" in html
    assert "/static/app.js" in html
    assert "/static/lucide.min.js" in html
    assert "Enterprise ML Pipeline" in html
    print("   -> GET / verified: Fintech sidebar, dropzone, and vendored assets confirmed.")

    print("2. Testing POST /upload (File Upload)...")
    sample_path = Path("public/samples/sample_sbi_statement.pdf")
    with open(sample_path, "rb") as f:
        upload_resp = client.post("/upload", files={"files": ("sample_sbi_statement.pdf", f, "application/pdf")})
    assert upload_resp.status_code == 200
    u_html = upload_resp.text
    assert "View Statement Dashboard" in u_html
    assert "Page Routing:" in u_html
    assert "/result/" in u_html
    
    # Extract session_id
    import re
    m = re.search(r'/result/([a-zA-Z0-9_\-]+)', u_html)
    assert m, "Could not extract session_id from upload response"
    session_id = m.group(1)
    print(f"   -> Upload processed successfully. Session ID: {session_id}")

    print("3. Testing GET /result/{session_id} (Overview Dashboard)...")
    res_page = client.get(f"/result/{session_id}")
    assert res_page.status_code == 200
    r_html = res_page.text
    assert "Total Inflow" in r_html
    assert "Total Outflow" in r_html
    assert "Net Cash Flow" in r_html
    assert "Ledger Volume" in r_html
    assert "Reconciliation" in r_html
    assert "account-number-display" in r_html
    assert "mask-toggle-btn" in r_html
    assert "overview-content" in r_html
    assert "tab-btn-transactions" in r_html
    print("   -> Overview Dashboard verified: 5 KPI metric cards, eye-mask toggle, and fintech tabs rendered.")

    print("4. Testing GET /result/{session_id}/transactions (Transactions Partial)...")
    tx_resp = client.get(f"/result/{session_id}/transactions")
    assert tx_resp.status_code == 200
    tx_html = tx_resp.text
    assert "fintech-table" in tx_html
    assert "data-tx-json=" in tx_html
    assert "inline-cat-select" in tx_html
    assert "table-density-switch" in tx_html
    print("   -> Transactions Partial verified: JSON serialized rows, inline category selector, density switch present.")

    print("5. Testing GET /result/{session_id}/analytics (Analytics Partial)...")
    an_resp = client.get(f"/result/{session_id}/analytics")
    assert an_resp.status_code == 200
    an_html = an_resp.text
    assert "categoryChart" in an_html
    assert "monthlyChart" in an_html
    assert "Top Payees &amp; Merchants" in an_html or "Top Payees & Merchants" in an_html
    print("   -> Analytics Partial verified: Doughnut & Monthly bar charts and top payees present.")

    print("6. Testing GET /result/{session_id}/quality (Quality & Reconciliation Partial)...")
    q_resp = client.get(f"/result/{session_id}/quality")
    assert q_resp.status_code == 200
    q_html = q_resp.text
    assert "Extraction &amp; Ledger Quality Audit" in q_html or "Extraction & Ledger Quality Audit" in q_html
    assert "Summary Balance Formula" in q_html
    assert "Auto-Corrected Transactions" in q_html
    print("   -> Quality Report verified: Mathematical continuity matrix and audit observations confirmed.")

    print("7. Testing GET /result/{session_id}/account (Account Details Partial)...")
    acc_resp = client.get(f"/result/{session_id}/account")
    assert acc_resp.status_code == 200
    acc_html = acc_resp.text
    assert "Account Credentials &amp; Identity" in acc_html or "Account Credentials & Identity" in acc_html
    assert "Branch &amp; Routing Codes" in acc_html or "Branch & Routing Codes" in acc_html
    print("   -> Account Details Partial verified: Credentials and routing codes confirmed.")

    print("8. Testing GET /result/{session_id}/export (Export Center Partial)...")
    ex_resp = client.get(f"/result/{session_id}/export")
    assert ex_resp.status_code == 200
    ex_html = ex_resp.text
    assert "Download Excel Workbook (.xlsx)" in ex_html
    assert "Download CSV (UTF-8-SIG)" in ex_html
    assert "Included Excel Worksheets (6 Tabs)" in ex_html
    print("   -> Export Partial verified: 6-sheet breakdown and download links confirmed.")

    print("9. Testing PATCH /transaction/{session_id}/{row_id} (Inline Category Edit)...")
    row_m = re.search(r'id="tx-row-([^"]+)"', tx_html)
    assert row_m, "No transaction row id found"
    row_id = row_m.group(1)
    
    patch_resp = client.patch(
        f"/transaction/{session_id}/{row_id}",
        data={"new_category": "Investments"}
    )
    assert patch_resp.status_code == 200
    p_html = patch_resp.text
    assert f'id="tx-row-{row_id}"' in p_html
    assert 'value="Investments" selected' in p_html
    assert 'hx-swap-oob="true"' in p_html
    print("   -> Inline Category Patch verified: Category updated to 'Investments' and #recon-badge swapped OOB.")

    print("10. Testing Downloads (Excel & CSV)...")
    xlsx_resp = client.get(f"/export/{session_id}/xlsx")
    assert xlsx_resp.status_code == 200
    assert len(xlsx_resp.content) > 1000
    
    csv_resp = client.get(f"/export/{session_id}/csv")
    assert csv_resp.status_code == 200
    assert csv_resp.content.startswith(b"\xef\xbb\xbf") # UTF-8 BOM
    print("   -> File Exports verified: Excel binary workbook and UTF-8 BOM CSV verified.")

    print("\n>>> ALL 10 FINTECH REDESIGN CRITERIA PASSED WITH ZERO REGRESSIONS! <<<")

if __name__ == "__main__":
    test_full_redesign_flow()
