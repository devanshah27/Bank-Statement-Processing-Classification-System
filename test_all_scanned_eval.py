import sys
import glob
from pathlib import Path
import csv
from decimal import Decimal as D
from core.pipeline import StatementPipeline

def num(x):
    return D(x.replace(',', '')) if x not in ('', None) else D(0)

def key(r):
    return (r['date'], num(r.get('debit', '')), num(r.get('credit', '')), num(r.get('balance', '')))

def run_all_scanned():
    scanned_dir = Path("bank_statement_test_dataset/scanned_image_based")
    gt_dir = Path("bank_statement_test_dataset/ground_truth")
    
    files = sorted(list(scanned_dir.glob("*.pdf")))
    pipeline = StatementPipeline()
    
    print(f"{'PDF File':<42} | {'GT':<4} | {'Ext':<4} | {'Match':<6} | {'Accuracy':<8} | {'Grade':<5}")
    print("-" * 80)
    
    for f in files:
        gt_file = gt_dir / f"{f.stem}.csv"
        if not gt_file.exists():
            print(f"{f.name:<42} | No GT found")
            continue
            
        gt = [r for r in csv.DictReader(open(gt_file, encoding='utf-8'))]
        
        try:
            res = pipeline.process(f.read_bytes(), f.name)
            out_rows = []
            for t in res.transactions:
                out_rows.append({
                    'date': str(t.get('date', '')) if t.get('date') is not None else '',
                    'debit': str(t.get('debit', '')) if t.get('debit') is not None else '',
                    'credit': str(t.get('credit', '')) if t.get('credit') is not None else '',
                    'balance': str(t.get('balance', '')) if t.get('balance') is not None else '',
                })
            pool = [key(r) for r in out_rows]
            hit = 0
            for r in gt:
                k = key(r)
                if k in pool:
                    pool.remove(k)
                    hit += 1
            grade = res.quality.get("grade", "?") if res.quality else "?"
            pct = 100 * hit / max(1, len(gt))
            print(f"{f.name:<42} | {len(gt):<4} | {len(out_rows):<4} | {hit:<6} | {pct:>6.1f}%  | {grade:<5}")
        except Exception as e:
            print(f"{f.name:<42} | ERROR: {e}")

if __name__ == "__main__":
    run_all_scanned()

