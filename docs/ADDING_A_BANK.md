# Adding a New Bank Profile

This guide explains how to add support for a new bank statement layout to the Bank Statement Processing & Classification System without modifying any core code.

The system uses a **Plugin / Registry Pattern** (`core/banks/registry.py`). Adding a new bank requires adding a single YAML profile under `core/banks/profiles/` and optionally updating synonym dictionaries in `config/header_synonyms.yaml`.

---

## 1. Bank Profile Structure

Every bank profile is a YAML file placed in `core/banks/profiles/<bank_code>.yaml`.

### Complete Specification Schema

```yaml
# core/banks/profiles/sample_bank.yaml

name: "Sample Bank Ltd"
code: "sample_bank"

# Bank Auto-Detection Rules
bank_name_keywords:
  - "Sample Bank"
  - "Sample Bank of India"
  - "samplebank.com"

# 4-character RBI IFSC prefix
ifsc_prefix: "SMPL"

# Account Details Extraction Regexes
account_number_regex: "(?:Account|A/c|Account\\s+No)[.:\\s]*([0-9]{9,18})"
ifsc_regex: "(SMPL0[0-9A-Z]{6})"
micr_regex: "([0-9]{9})"
customer_id_regex: "(?:Cust(?:omer)?\\s*(?:ID|No))[.:\\s]*([0-9A-Z]{6,12})"

# Name Extraction Anchor Keywords
holder_name_anchors:
  - "Account Name"
  - "Customer Name"
  - "Name :"
  - "Messrs"

# Transaction Table Configuration
table_strategy: "lines"  # "lines" for bordered tables, "text" for borderless whitespace tables

# Column Configuration
debit_credit_style: "separate_columns"
# Supported values:
# - separate_columns: Distinct Debit / Withdrawal and Credit / Deposit columns
# - single_with_suffix: Single Amount column with "Dr" / "Cr" or "D" / "C"
# - signed: Single Amount column where debits are negative (-1250.00)
# - bracketed: Single Amount column with bracketed debits (1,250.00)

amount_format: "indian"
# Supported values:
# - indian: 1,23,456.78
# - western: 1,234,567.89

date_format_hints:
  - "DD/MM/YYYY"
  - "DD-MM-YYYY"
  - "DD MMM YYYY"

# Bank-Specific Column Synonyms (Merged with config/header_synonyms.yaml)
column_synonyms:
  date:
    - "txn dt"
    - "post date"
  description:
    - "particulars / narration"
    - "transaction description"
  ref_no:
    - "cheque / ref"
    - "utr / ref no"
  debit:
    - "withdrawal (dr)"
    - "dr amt"
  credit:
    - "deposit (cr)"
    - "cr amt"
  balance:
    - "balance amount (inr)"
    - "running bal"

# Page Header / Footer Discard Sentinels
discard_row_patterns:
  - "brought forward"
  - "carried forward"
  - "b/f"
  - "c/f"
  - "page total"
  - "opening balance"
  - "closing balance"
```

---

## 2. Step-by-Step Walkthrough: Adding "Federal Bank"

### Step 1: Create Profile YAML
Create `core/banks/profiles/federal.yaml`:

```yaml
name: "Federal Bank"
code: "federal"
bank_name_keywords:
  - "Federal Bank"
  - "The Federal Bank Limited"
  - "fednet"
ifsc_prefix: "FDRL"
account_number_regex: "(?:Account\\s+Number|A/c\\s+No)[.:\\s]*([0-9]{14})"
ifsc_regex: "(FDRL0[0-9]{6})"
micr_regex: "([0-9]{9})"
customer_id_regex: "(?:Fed\\s*ID|Customer\\s*ID)[.:\\s]*([0-9]{7,10})"
holder_name_anchors:
  - "Account Name"
  - "Name of Account Holder"
table_strategy: "lines"
debit_credit_style: "separate_columns"
amount_format: "indian"
date_format_hints:
  - "DD-MM-YYYY"
  - "DD/MM/YYYY"
discard_row_patterns:
  - "opening balance"
  - "closing balance"
  - "page summary"
```

### Step 2: Register Global Header Synonyms (Optional)
If Federal Bank uses a unique column header not already present in `config/header_synonyms.yaml`, add it under the corresponding canonical column name in `config/header_synonyms.yaml`:

```yaml
# config/header_synonyms.yaml
debit:
  - "fed debits"
  - "dr amount"
```

### Step 3: Verify Registration
The registry automatically scans `core/banks/profiles/*.yaml` upon startup. Test that your new profile loads properly:

```python
from core.banks.registry import BankProfileRegistry

registry = BankProfileRegistry()
profile = registry.get_profile("federal")
assert profile is not None
assert profile.ifsc_prefix == "FDRL"
print("Federal Bank profile registered successfully!")
```

### Step 4: Run Auto-Detection Test
Test that a sample document text containing Federal Bank keywords or IFSC routes to your new profile:

```python
sample_text = """
THE FEDERAL BANK LIMITED
Branch: Aluva Main Branch, IFSC: FDRL0001001
Account Name: RAJESH KUMAR
Account Number: 10010100123456
"""
detected = registry.detect_bank(sample_text)
assert detected.code == "federal"
print("Auto-detection verified!")
```

---

## 3. Fallback Mechanism

If no bank profile matches the document text (or IFSC code), the system automatically routes extraction to `core/banks/profiles/generic.yaml`. The generic profile utilizes fuzzy regex anchors, broad header synonym matching, and adaptive balance reconciliation to parse non-standard bank statements.

