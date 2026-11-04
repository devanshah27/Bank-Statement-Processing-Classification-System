import time, fitz, cv2, numpy as np
from core.ocr.preprocessing import preprocess_for_ocr
from core.ocr.rapidocr_engine import RapidOCREngine
from core.ocr.table_recovery import recover_table

doc = fitz.open("bank_statement_test_dataset/scanned_image_based/07_scanned_clean_200dpi.pdf")
engine = RapidOCREngine()

print(f"Profiling {len(doc)} pages of 07_scanned_clean_200dpi.pdf:")
total_t0 = time.perf_counter()

for p_idx in range(len(doc)):
    page_t0 = time.perf_counter()
    
    t_rend0 = time.perf_counter()
    pix = doc[p_idx].get_pixmap(dpi=200)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.h, pix.w, pix.n))
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    t_rend = (time.perf_counter() - t_rend0) * 1000

    t_prep0 = time.perf_counter()
    prep = preprocess_for_ocr(img, binarize=False)
    t_prep = (time.perf_counter() - t_prep0) * 1000

    t_ocr0 = time.perf_counter()
    ocr_res = engine.recognize(prep.image)
    t_ocr = (time.perf_counter() - t_ocr0) * 1000

    t_rec0 = time.perf_counter()
    cells = recover_table(ocr_res, prep.image.shape[1])
    t_rec = (time.perf_counter() - t_rec0) * 1000

    page_tot = (time.perf_counter() - page_t0) * 1000
    print(f"Page {p_idx+1}: Render={t_rend:.0f}ms, Prep={t_prep:.0f}ms, OCR={t_ocr:.0f}ms, Table={t_rec:.0f}ms -> Total={page_tot:.0f}ms (conf={ocr_res.page_confidence:.2f}, words={len(ocr_res.words)}, rows={len(cells)})")

print(f"Overall time: {(time.perf_counter() - total_t0):.2f}s")
