import time
from pathlib import Path
from core.pipeline import StatementPipeline

folder = Path("bank_statement_test_dataset/scanned_image_based")
pipeline = StatementPipeline()

for pdf in sorted(folder.glob("*.pdf")):
    with open(pdf, "rb") as f:
        pdf_bytes = f.read()
    t0 = time.time()
    try:
        res = pipeline.process(pdf_bytes, filename=pdf.name)
        elapsed = time.time() - t0
        dates_valid = sum(1 for tx in res.transactions if tx.get("date"))
        print(f"[{pdf.name}] {elapsed:.1f}s | Success: {res.success} | Txs: {len(res.transactions)} (Valid dates: {dates_valid}) | Bank: {res.bank_profile}")
    except Exception as e:
        elapsed = time.time() - t0
        print(f"[{pdf.name}] FAILED after {elapsed:.1f}s: {e}")

