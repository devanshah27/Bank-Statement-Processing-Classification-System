# Hybrid Transaction Classification Engine (Non-LLM)

The Bank Statement Processing & Classification System features a **two-stage hybrid classifier** strictly engineered without any LLM APIs or external AI services. The engine is entirely self-hosted, deterministic, audit-friendly, and operates in sub-millisecond latency per transaction.

---

## 1. Architectural Philosophy

1. **Strictly Non-LLM**: Zero dependence on OpenAI, Anthropic, Gemini, or any paid cloud AI.
2. **Deterministic & Auditable**: Every classified transaction carries a machine-readable `classification_source` (`rule`, `ml`, or `override`) and an explicit `reason` explaining the trigger.
3. **Ultra-Fast & Lightweight**: The deployed inference pipeline requires only `numpy`—no heavy runtimes such as PyTorch, TensorFlow, or Scikit-Learn at serving time.
4. **Resilient to OCR Errors & Typographical Drift**: Fuzzy token matching (RapidFuzz) and character n-grams absorb OCR noise and vendor naming variances.

---

## 2. Category Taxonomy (23 Standard Classes)

All categories are configured in `config/categories.yaml`:

| Category Name | Type | Typical Triggers |
|---|---|---|
| `Salary/Income` | Credit | `SALARY`, `PAYROLL`, `CREDIT INTEREST`, `DIVIDEND` |
| `Rent` | Debit | `RENT`, `LANDLORD`, `RENTAL`, `HOUSING` |
| `Groceries` | Debit | `BIGBASKET`, `BLINKIT`, `ZEPTO`, `DMART`, `NATURES BASKET` |
| `Food & Dining` | Debit | `SWIGGY`, `ZOMATO`, `MCDONALDS`, `STARBUCKS`, `RESTAURANT` |
| `Shopping/E-commerce` | Debit | `AMAZON`, `FLIPKART`, `MYNTRA`, `MEESHO`, `AJIO` |
| `Utilities` | Debit | `BESCOM`, `TATA POWER`, `IGL`, `MAHADISCOM`, `WATER BILL` |
| `Mobile/Internet/DTH` | Debit | `AIRTEL`, `JIO`, `VODAFONE`, `ACT FIBERNET`, `TATA PLAY` |
| `Transport/Fuel/Travel` | Debit | `UBER`, `OLA`, `IRCTC`, `MAKEMYTRIP`, `HPCL`, `BPCL`, `IOCL` |
| `Entertainment/Subscriptions` | Debit | `NETFLIX`, `SPOTIFY`, `BOOKMYSHOW`, `HOTSTAR`, `PRIME VIDEO` |
| `Healthcare/Pharmacy` | Debit | `APOLLO PHARMACY`, `PHARMEASY`, `1MG`, `HOSPITAL`, `CLINIC` |
| `Education` | Debit | `SCHOOL FEES`, `COLLEGE`, `COURSERA`, `UDEMY`, `UNIVERSITY` |
| `Insurance` | Debit | `LIC`, `HDFC LIFE`, `ICICI PRU`, `STAR HEALTH`, `MAX LIFE` |
| `Investments` | Debit/Credit | `ZERODHA`, `GROWW`, `MUTUAL FUND`, `SIP`, `KFINTECH`, `CAMS` |
| `Loan/EMI` | Debit | `LOAN EMI`, `HOME LOAN`, `BAJAJ FINANCE`, `AUTO LOAN` |
| `Credit Card Payment` | Debit | `CRED`, `CC PAYMENT`, `SBI CARD`, `HDFC CC`, `CREDIT CARD` |
| `Taxes` | Debit | `GST`, `INCOME TAX`, `CHALLAN`, `ADVANCE TAX`, `TDS` |
| `Bank Charges/Fees` | Debit | `ANNUAL FEE`, `SMS CHARGES`, `MIN BAL CHG`, `DEBIT CARD CHG` |
| `Interest Earned` | Credit | `SB INT`, `INTEREST CREDIT`, `TERM DEPOSIT INT` |
| `ATM/Cash Withdrawal` | Debit | `ATM WDL`, `CASH WDL`, `NFS CASH`, `ATM-` |
| `Cash Deposit` | Credit | `CASH DEP`, `CDM DEP`, `BY CASH` |
| `Transfers` | Both | `NEFT`, `IMPS`, `RTGS`, `UPI TRANSFER`, `INTERNAL TRANSFER` |
| `Refunds/Reversals` | Credit | `REFUND`, `REVERSAL`, `REV-`, `CHARGEBACK` |
| `Cheque` | Both | `CHQ DEPOSIT`, `CHQ RETURN`, `CLEARING CHEQUE` |
| `Other/Uncategorized` | Both | Unmatched fallback (< 0.15 confidence) |

---

## 3. Two-Stage Classification Pipeline

```mermaid
flowchart TD
    Input[Transaction Narration + Direction + Amount] --> FeatExt[Feature Extraction & Normalization]
    FeatExt --> StageA{Stage A: Rules Engine}
    StageA -- Rule Match (Confidence >= 0.70) --> Result[Final Category & Source: Rule]
    StageA -- No Rule Match or Low Score --> StageB[Stage B: ML Inference (Numpy)]
    StageB --> CalibCheck{Calibrated Prob >= 0.15?}
    CalibCheck -- Yes --> Result2[Final Category & Source: ML]
    CalibCheck -- No --> Fallback[Category: Other/Uncategorized & Source: ML Fallback]
```

### Stage A: Heuristic Rules Engine (`core/classification/rules_engine.py`)

1. **Pre-Processing & Sanitization**:
   - Strips transaction IDs, reference sequences (e.g., `UPI/1234567890/REF`), terminal codes, dates, and amounts.
   - Normalizes channel identifiers: `UPI`, `NEFT`, `IMPS`, `RTGS`, `POS`, `ATM`, `NACH`, `ACH`, `CHQ`.
2. **Channel & Direction Filters**:
   - Rules can constrain direction (`debit`, `credit`, or `any`).
   - For example, `Salary/Income` is strictly enforced for `credit` transactions only.
3. **Merchant Alias Matching**:
   - Searches `config/merchants.yaml` using regex word boundaries (`\b<name>\b`) to avoid false positives (e.g. `cred` vs `credit`).
   - RapidFuzz token matching for phonetic variations.
4. **Rule Score Evaluation**:
   - Matches are evaluated in priority order. If a rule evaluates to confidence $\ge 0.70$, Stage A immediately classifies the transaction.

### Stage B: Traditional ML Serving (`core/classification/ml_model.py`)

If Stage A yields no definitive rule match, execution passes to Stage B.

1. **Dual Feature Extraction**:
   - **Word TF-IDF**: captures semantic keywords (`ngram_range=(1, 2)`).
   - **Char N-gram TF-IDF**: captures morphological roots and OCR typos (`ngram_range=(3, 5)`).
   - **One-Hot Metadata**: Direction (`is_debit`, `is_credit`) and payment channel.
2. **Cross-Validation Training (`training/train.py`)**:
   - Evaluates LinearSVC, LogisticRegression, and ComplementNB via 5-Fold Stratified Cross-Validation.
   - Best model (LinearSVC) achieves **0.9989 Macro F1** on synthetic training benchmarks and **0.8886 Accuracy** on noisy real-world holdout datasets.
   - Calibrated via `CalibratedClassifierCV(method="sigmoid", cv=3)` to output true class probability distributions.

---

## 4. Ultra-Fast Numpy Serving Format

To support deployment on serverless environments such as **Vercel** without triggering bundle size limits, the trained scikit-learn model is compiled into pure NumPy arrays:

```
models/v1/
├── tfidf_vocab.json         # Word vocabulary mapping {word: index}
├── tfidf_idf.npy            # Precomputed word IDF vector (float32)
├── char_tfidf_vocab.json    # Char n-gram vocabulary mapping
├── char_tfidf_idf.npy       # Precomputed char IDF vector (float32)
├── classifier_weights.npz   # Calibrated linear weights W (k x d) & bias b (k)
└── metadata.json            # Model version, class labels, hyperparameters
```

### Serving Without Scikit-Learn

```python
# Pure NumPy Inference (~0.4 ms)
# 1. Sparse TF-IDF Vectorization
x_vec = vectorize_text(tokens, vocab, idf)

# 2. Linear Projection
logits = np.dot(weights, x_vec) + bias

# 3. Sigmoid Calibration & Softmax Normalization
probs = sigmoid_calibration(logits, calib_a, calib_b)
predicted_idx = int(np.argmax(probs))
predicted_class = classes[predicted_idx]
confidence = float(probs[predicted_idx])
```

**Benefits**:
- Zero dependencies on `scipy`, `scikit-learn`, or `pandas` at runtime.
- Reduces bundle size by **~150 MB**.
- Cold start in under 5 milliseconds.

---

## 5. Recurring Transaction Detector (`core/classification/recurring.py`)

The pipeline automatically identifies recurring expenses (such as SIPs, rent, EMIs, and utility subscriptions):
1. **Merchant / Narration Clustering**: Groups transactions by counterparty/normalized merchant.
2. **Interval Consistency**: Computes day deltas between consecutive transactions ($28 \le \Delta d \le 32$ for monthly cadence).
3. **Amount Variance**: Checks that $|A_i - A_{avg}| / A_{avg} \le 0.05$ (within 5% tolerance).
4. **Cadence Labeling**: Flags transactions as `monthly`, `quarterly`, `weekly`, or `annual` and predicts the next occurrence date.

---

## 6. Human-in-the-Loop & Feedback Retraining

When an end-user corrects a category in the web UI:
1. The correction is persisted to `data/feedback.jsonl`:
   ```json
   {"description": "UPI-SWIGGY-12948-BANGALORE", "corrected_category": "Food & Dining", "timestamp": "2026-11-04T15:30:00"}
   ```
2. Running `python training/feedback_retrain.py` merges feedback instances with the training set, automatically retrains the linear classifier, updates calibrated probabilities, and exports the new model artifacts to `models/v1/`.

