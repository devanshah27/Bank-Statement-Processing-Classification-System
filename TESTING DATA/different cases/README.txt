Simple test inputs (all data is fictional)

1_simple_text_statement.pdf   - normal PDF with a text layer (1 page, ~20 rows)
2_simple_image_statement.pdf  - the SAME statement as a scanned image (no text layer, needs OCR)

expected_output.csv           - the correct rows for both files (compare descriptions ignoring whitespace)
expected_account_details.json - the correct account holder details, period, opening/closing balance

Both PDFs contain identical data, so your text and OCR paths should produce the same result.
