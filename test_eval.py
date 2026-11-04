import sys
from pathlib import Path
import csv
from decimal import Decimal as D
from core.pipeline import StatementPipeline

def num(x):
    return D(x.replace(',', '')) if x not in ('', None) else D(0)

def key(r):
    return (r['date'], num(r.get('debit', '')), num(r.get('credit', '')), num(r.get('balance', '')))

def evaluate_pdf(pdf_rel_path, gt_rel_path):
    pdf_path = Path(pdf_rel_path)
    gt_path = Path(gt_rel_path)
    
    print(f"\n==================================================")
    print(f"Processing: {pdf_path.name}")
    print(f"==================================================")
    
    pipeline = StatementPipeline()
    result = pipeline.process(pdf_path.read_bytes(), pdf_path.name)
    
    print("Success:", result.success)
    print("Total extracted txns:", len(result.transactions))
    print("Quality Grade:", result.quality.get("grade") if result.quality else None)
    print("Reconciliation:", result.quality.get("reconciliation", {}).get("status") if result.quality else None)
    
    gt = [r for r in csv.DictReader(open(gt_path, encoding='utf-8'))]
    print("Ground truth rows:", len(gt))
    
    out_rows = []
    for t in result.transactions:
        out_rows.append({
            'date': str(t.get('date', '')) if t.get('date') is not None else '',
            'debit': str(t.get('debit', '')) if t.get('debit') is not None else '',
            'credit': str(t.get('credit', '')) if t.get('credit') is not None else '',
            'balance': str(t.get('balance', '')) if t.get('balance') is not None else '',
            'description': t.get('description', '') or ''
        })
        
    pool = [key(r) for r in out_rows]
    hit = 0
    for r in gt:
        k = key(r)
        if k in pool:
            pool.remove(k)
            hit += 1
            
    pct = 100 * hit / max(1, len(gt))
    print(f"Matches: {hit}/{len(gt)} ({pct:.1f}%) | Missing: {len(gt) - hit} | Extra: {len(pool)}")
    
    if hit < len(gt):
        print("\nFirst 3 ground-truth rows:")
        for r in gt[:3]:
            print("  GT :", r.get('date'), "DR:", r.get('debit'), "CR:", r.get('credit'), "BAL:", r.get('balance'), "DESC:", r.get('description')[:40])
        print("\nFirst 3 extracted rows:")
        for r in out_rows[:3]:
            print("  EXT:", r.get('date'), "DR:", r.get('debit'), "CR:", r.get('credit'), "BAL:", r.get('balance'), "DESC:", r.get('description')[:40])

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "07"
    if target == "07":
        evaluate_pdf(
            "bank_statement_test_dataset/scanned_image_based/07_scanned_clean_200dpi.pdf",
            "bank_statement_test_dataset/ground_truth/07_scanned_clean_200dpi.csv"
        )
