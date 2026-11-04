from core.pipeline import StatementPipeline
from pathlib import Path

pipeline = StatementPipeline()
pdf_path = Path('bank_statement_test_dataset/scanned_image_based/08_scanned_skewed_noisy_low_contrast.pdf')
res = pipeline.process(pdf_path.read_bytes(), '08.pdf')
print(f"Total transactions: {len(res.transactions)}")
for i, t in enumerate(res.transactions[:15]):
    print(f"Row {i:02d}: {t.get('date')} | DR={t.get('debit')} | CR={t.get('credit')} | BAL={t.get('balance')} | {t.get('description')[:40]}")

