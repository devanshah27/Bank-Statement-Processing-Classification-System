# Bank Statement Test Dataset (all data is FICTIONAL)

Synthetic statements for testing a bank-statement extraction + classification pipeline.
Banks, names, account numbers, IFSC codes and transactions are invented. No real bank, person or account is represented.
Every PDF has a matching file in `ground_truth/` (CSV of expected rows + JSON of account details).

Compare descriptions ignoring whitespace: long narrations are wrapped across lines in the PDF.
`evaluate_extraction.py` gives a quick match score: `python evaluate_extraction.py ground_truth/01_*.csv your_output.csv`

## text_based/ (native text layer, use pdfplumber/PyMuPDF)
| File | Layout / what it tests |
|---|---|
| 01_text_bordered_debit_credit_carry_forward | Bordered table, separate Debit/Credit, Value Date + Ref columns, BROUGHT/CARRIED FORWARD rows on every page, repeated headers, Indian number format (1,23,456.78), DD/MM/YYYY, summary box + disclaimers |
| 02_text_borderless_amount_with_Dr_Cr_suffix | No grid lines, zebra shading, single Amount column with "Dr"/"Cr" suffix, balance with "Cr" suffix, "01 Jan 2026" dates |
| 03_text_multiline_narration_withdrawal_deposit | Narrow narration column, long narrations wrapping to 2-3 lines (date only on first line), headers "Withdrawal Amt." / "Deposit Amt." / "Closing Balance", DD-MM-YY |
| 04_text_monospace_whitespace_aligned_signed_amounts | Courier, no lines at all (text-strategy extraction), ISO dates, +/- signed amounts |
| 05_text_landscape_passbook_style_zero_filled | Landscape page, "Particulars / Withdrawals / Deposits", empty columns show 0.00 instead of blank |
| 06_text_bracket_negatives_western_number_format | Debits in brackets "(1,250.00)", Western 1,234,567.89 grouping, "Apr 12, 2026" dates |

## scanned_image_based/ (NO text layer, OCR required). Difficulty rises 07 -> 10
| File | Degradation |
|---|---|
| 07_scanned_clean_200dpi | Clean grayscale scan (layout of 01) |
| 08_scanned_skewed_noisy_low_contrast | 1.7 degree skew, blur, noise, speckles, low contrast, heavy JPEG (layout of 02) |
| 09_scanned_phone_photo_shadow_perspective | Phone-photo style: paper tint, shadow gradient, perspective warp, dark background corners, reduced resolution (layout of 03) |
| 10_scanned_fax_black_white_low_res | 1-bit black/white, 150 dpi, fax streaks (layout of 04, hardest) |
| 11_scanned_rotated_90_degrees | Page rotated 90 degrees plus noise: needs orientation detection (layout of 06) |

## mixed_and_multi/
| File | What it tests |
|---|---|
| 12_mixed_text_pages_and_scanned_pages | 6 pages: pages 2 and 6 are noisy scans, the rest native text. Needs PER-PAGE text-vs-image detection; one transaction table spans both |
| 13_two_accounts_in_one_pdf | Two different statements (different banks/layouts/accounts) concatenated; balance chain restarts at the boundary |
| 14_large_statement_100plus_pages | 218 pages, 7,700 rows, small font. Performance / memory / streaming test |

## edge_cases/
| File | What it tests |
|---|---|
| 15_edge_cases_balance_breaks_duplicates_reversals | Injected defects (see `ground_truth/15_*.csv` column `edge_flag` and the JSON): one row missing from print (balance gap), one exact duplicate row, one date outside the period and out of order, one zero-amount row, a debit + reversal-credit pair, one row printed in the wrong column (debit shown under Credit; balance proves it is a debit). The PDF summary totals are the TRUE ones, so printed rows will not reconcile |
| 16_password_protected_password_is_test1234 | Encrypted copy of file 01. Password: `test1234` |
| 17_corrupted_truncated_file | Truncated PDF, should fail gracefully |
| 18_not_really_a_pdf_png_renamed | A PNG with a .pdf extension, so check magic bytes, not the extension |
| 19_blank_empty_page | Valid PDF, no content |

## other_document_types/
| File | What it tests |
|---|---|
| 20_credit_card_statement_no_balance_column | Credit card statement: no running balance, "Cr" suffix on payments/refunds, due-date block. Should be handled or flagged, not mis-parsed |
| 21_non_statement_invoice_should_be_rejected | An invoice. Should be rejected as "not a bank statement" |

## classification_data/ (for the non-LLM classifier)
- `classification_train_labeled.csv`: 3,500 narrations (description, direction DR/CR, amount, category) across 4 narration styles (slash-, hyphen-, word-style, compact)
- `classification_holdout_test_noisy.csv`: 700 rows from different random seeds with lowercasing, truncation and OCR-style character errors (O->0, I->1, S->5)
- Categories: Salary/Income, Rent, Groceries, Food & Dining, Shopping/E-commerce, Utilities, Mobile/Internet/DTH, Transport/Fuel/Travel, Entertainment/Subscriptions, Healthcare/Pharmacy, Education, Insurance, Investments, Loan/EMI, Credit Card Payment, Taxes, Bank Charges/Fees, Interest Earned, ATM/Cash Withdrawal, Cash Deposit, Transfers, Refunds/Reversals, Cheque
- Caveat: labels come from a generator, so accuracy on these will look better than on real statements. Always also test on a few real (redacted) statements.

## Regenerate
`python generate_dataset.py` (needs reportlab, pypdf, Pillow, opencv-python, numpy, poppler-utils). Seeds are fixed, so output is reproducible.
