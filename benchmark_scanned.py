import time
from pathlib import Path
from core.pipeline import StatementPipeline

pdf_path = Path("bank_statement_test_dataset/scanned_image_based/07_scanned_clean_200dpi.pdf")
with open(pdf_path, "rb") as f:
    pdf_bytes = f.read()

t0 = time.time()
pipeline = StatementPipeline()
res = pipeline.process(pdf_bytes, filename=pdf_path.name)
t1 = time.time()

print(f"Total time: {t1 - t0:.2f}s")
print(f"Success: {res.success}")
print(f"Transactions extracted: {len(res.transactions)}")
print(f"Bank detected: {res.bank_profile}")
print(f"Account holder: {res.account.get('account_holder_name')}")
print(f"Account num: {res.account.get('account_number')}")
print(f"Reconciliation rate: {res.quality.get('reconciliation_rate')}")
print(f"Quality grade: {res.quality.get('quality_grade')}")
print("First 3 transactions:")
for tx in res.transactions[:3]:
    print(f"  {tx['date']} | {tx['description'][:30]} | Dr: {tx['debit']} | Cr: {tx['credit']} | Bal: {tx['balance']}")

