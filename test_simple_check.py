import sys
from pathlib import Path
from core.pipeline import StatementPipeline

def test_simple():
    pdf_path = Path("TESTING DATA/simple_inputs/2_simple_image_statement.pdf")
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        return
        
    pipeline = StatementPipeline()
    res = pipeline.process(pdf_path.read_bytes(), pdf_path.name)
    print("Success:", res.success)
    print("Total txns:", len(res.transactions))
    print("Quality Grade:", res.quality.get("grade") if res.quality else None)
    
    print("\nTransactions:")
    for i, t in enumerate(res.transactions[:5]):
        print(f"Row {i:02d}: Date={t.get('date')} | DR={t.get('debit')} | CR={t.get('credit')} | BAL={t.get('balance')} | DESC={t.get('description')[:35]} | CAT={t.get('category')}")

if __name__ == "__main__":
    test_simple()

