#!/usr/bin/env python3
"""Score your extractor against the ground truth.
Usage: python evaluate_extraction.py ground_truth/01_xxx.csv your_output.csv
Your CSV needs columns: date (ISO), debit, credit, balance. Description is compared whitespace-insensitively."""
import csv, sys
from decimal import Decimal as D
def num(x): return D(x.replace(',', '')) if x not in ('', None) else D(0)
def key(r): return (r['date'], num(r.get('debit', '')), num(r.get('credit', '')), num(r.get('balance', '')))
gt = [r for r in csv.DictReader(open(sys.argv[1], encoding='utf-8'))]
out = [r for r in csv.DictReader(open(sys.argv[2], encoding='utf-8-sig'))]
pool = [key(r) for r in out]; hit = 0
for r in gt:
    k = key(r)
    if k in pool: pool.remove(k); hit += 1
print(f"ground-truth rows: {len(gt)} | extracted rows: {len(out)} | exact matches (date+debit+credit+balance): {hit} ({100*hit/max(1,len(gt)):.1f}%)")
print(f"missing: {len(gt)-hit} | extra/incorrect: {len(pool)}")
