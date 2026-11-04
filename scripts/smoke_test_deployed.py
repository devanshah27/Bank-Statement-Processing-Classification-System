#!/usr/bin/env python3
"""Smoke test script for validating a deployed bank statement system on Vercel or any remote host.

Usage:
    python scripts/smoke_test_deployed.py https://your-deployment-url.vercel.app
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
import httpx

DATASET_DIR = Path("samples/test_dataset")
if not DATASET_DIR.is_dir():
    DATASET_DIR = Path("bank_statement_test_dataset")


def run_smoke_test(base_url: str) -> bool:
    """Run smoke tests against deployed URL."""
    base_url = base_url.rstrip("/")
    client = httpx.Client(base_url=base_url, timeout=60.0, follow_redirects=True)
    all_passed = True

    print(f"\n=======================================================")
    print(f"SMOKE TEST TARGET: {base_url}")
    print(f"=======================================================\n")

    # 1. Health check
    print("[1/6] Testing /api/health...")
    try:
        r = client.get("/api/health")
        if r.status_code == 200:
            data = r.json()
            print(f"  ✓ Health OK: version={data.get('version')}, engine={data.get('ocr_engine')}")
        else:
            print(f"  ✗ Health failed: status={r.status_code}, response={r.text}")
            all_passed = False
    except Exception as e:
        print(f"  ✗ Health request exception: {e}")
        all_passed = False

    # 2. Text PDF upload
    print("\n[2/6] Testing native text PDF upload (01_text_bordered)...")
    pdf_text = DATASET_DIR / "text_based" / "01_text_bordered_debit_credit_carry_forward.pdf"
    if pdf_text.is_file():
        try:
            files = {"file": (pdf_text.name, pdf_text.read_bytes(), "application/pdf")}
            r = client.post("/api/upload", files=files)
            if r.status_code == 200:
                data = r.json()
                rows = len(data.get("transactions", []))
                print(f"  ✓ Upload text PDF OK: extracted {rows} rows (expected 103)")
            else:
                print(f"  ✗ Upload text PDF failed: status={r.status_code}, body={r.text}")
                all_passed = False
        except Exception as e:
            print(f"  ✗ Text upload exception: {e}")
            all_passed = False

    # 3. Scanned PDF upload (07_scanned_clean_200dpi)
    print("\n[3/6] Testing scanned OCR PDF upload (07_scanned_clean)...")
    pdf_scan = DATASET_DIR / "scanned_image_based" / "07_scanned_clean_200dpi.pdf"
    if pdf_scan.is_file():
        try:
            files = {"file": (pdf_scan.name, pdf_scan.read_bytes(), "application/pdf")}
            r = client.post("/api/upload", files=files)
            if r.status_code == 200:
                data = r.json()
                rows = len(data.get("transactions", []))
                print(f"  ✓ Upload scanned PDF OK: extracted {rows} rows via OCR")
            else:
                print(f"  ✗ Upload scanned PDF failed: status={r.status_code}, body={r.text}")
                all_passed = False
        except Exception as e:
            print(f"  ✗ Scanned upload exception: {e}")
            all_passed = False

    # 4. Password-protected PDF (16_password_protected)
    print("\n[4/6] Testing password-protected PDF handling...")
    pdf_pwd = DATASET_DIR / "edge_cases" / "16_password_protected_password_is_test1234.pdf"
    if pdf_pwd.is_file():
        try:
            # First attempt without password (must return 401 PASSWORD_REQUIRED)
            files = {"file": (pdf_pwd.name, pdf_pwd.read_bytes(), "application/pdf")}
            r = client.post("/api/upload", files=files)
            if r.status_code == 401 and "PASSWORD_REQUIRED" in r.text:
                print("  ✓ Correctly rejected unauthenticated password PDF (401)")
            else:
                print(f"  ✗ Expected 401 PASSWORD_REQUIRED, got: {r.status_code}")
                all_passed = False

            # Second attempt with password test1234
            files = {"file": (pdf_pwd.name, pdf_pwd.read_bytes(), "application/pdf")}
            r = client.post("/api/upload", files=files, data={"password": "test1234"})
            if r.status_code == 200:
                data = r.json()
                print(f"  ✓ Successfully decrypted with test1234: {len(data.get('transactions', []))} rows")
            else:
                print(f"  ✗ Password decryption failed: {r.status_code}")
                all_passed = False
        except Exception as e:
            print(f"  ✗ Password test exception: {e}")
            all_passed = False

    # 5. Non-statement invoice rejection (21_non_statement_invoice)
    print("\n[5/6] Testing non-statement rejection (21_invoice)...")
    pdf_inv = DATASET_DIR / "other_document_types" / "21_non_statement_invoice_should_be_rejected.pdf"
    if pdf_inv.is_file():
        try:
            files = {"file": (pdf_inv.name, pdf_inv.read_bytes(), "application/pdf")}
            r = client.post("/api/upload", files=files)
            if r.status_code == 422 and "NOT_A_BANK_STATEMENT" in r.text:
                print("  ✓ Correctly rejected invoice as non-statement document (422)")
            else:
                print(f"  ✗ Invoice rejection test failed: status={r.status_code}, response={r.text}")
                all_passed = False
        except Exception as e:
            print(f"  ✗ Invoice rejection exception: {e}")
            all_passed = False

    # 6. UI Landing page test
    print("\n[6/6] Testing UI landing page HTML...")
    try:
        r = client.get("/")
        if r.status_code == 200 and "Bank Statement" in r.text and "dropzone" in r.text:
            print("  ✓ UI Dashboard loaded successfully (HTML 200)")
        else:
            print(f"  ✗ UI check failed: status={r.status_code}")
            all_passed = False
    except Exception as e:
        print(f"  ✗ UI landing exception: {e}")
        all_passed = False

    print("\n-------------------------------------------------------")
    if all_passed:
        print("ALL SMOKE TESTS PASSED! Deployment is functional and healthy.")
    else:
        print("SOME SMOKE TESTS FAILED. Please review the logs above.")
    print("-------------------------------------------------------\n")

    return all_passed


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/smoke_test_deployed.py <DEPLOYED_URL>")
        sys.exit(1)

    url = sys.argv[1]
    success = run_smoke_test(url)
    sys.exit(0 if success else 1)

