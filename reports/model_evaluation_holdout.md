# Model Evaluation on Noisy Holdout Dataset

- **Evaluated At**: 2026-11-04T16:35:22.550702
- **Holdout File**: `classification_holdout_test_noisy.csv`
- **Test Rows**: 700
- **Accuracy**: **98.71%**
- **Macro F1**: **0.9528**
- **Weighted F1**: **0.9928**

## Hybrid Engine Coverage Breakdown

- **Stage A (Rules / Heuristics)**: 579 rows (82.7%)
- **Stage B (ML Model)**: 113 rows (16.1%)
- **Uncategorized / Needs Review**: 8 rows (1.1%)

## Per-Class Classification Report

```
                             precision    recall  f1-score   support

        ATM/Cash Withdrawal     1.0000    1.0000    1.0000        25
          Bank Charges/Fees     1.0000    1.0000    1.0000        34
               Cash Deposit     1.0000    1.0000    1.0000         8
                     Cheque     1.0000    1.0000    1.0000         5
        Credit Card Payment     1.0000    1.0000    1.0000        22
                  Education     1.0000    1.0000    1.0000         1
Entertainment/Subscriptions     1.0000    0.9804    0.9901        51
              Food & Dining     1.0000    0.9894    0.9947        94
                  Groceries     1.0000    1.0000    1.0000        60
        Healthcare/Pharmacy     1.0000    0.9474    0.9730        19
                  Insurance     1.0000    1.0000    1.0000        29
            Interest Earned     1.0000    1.0000    1.0000        13
                Investments     1.0000    1.0000    1.0000        34
                   Loan/EMI     1.0000    1.0000    1.0000        29
        Mobile/Internet/DTH     1.0000    0.9643    0.9818        28
        Other/Uncategorized     0.0000    0.0000    0.0000         0
          Refunds/Reversals     1.0000    1.0000    1.0000         5
                       Rent     1.0000    0.9677    0.9836        31
              Salary/Income     1.0000    1.0000    1.0000        27
        Shopping/E-commerce     1.0000    0.9756    0.9877        41
                      Taxes     1.0000    1.0000    1.0000         1
                  Transfers     0.9804    0.9804    0.9804        51
      Transport/Fuel/Travel     1.0000    0.9821    0.9910        56
                  Utilities     1.0000    0.9722    0.9859        36

                   accuracy                         0.9871       700
                  macro avg     0.9575    0.9483    0.9528       700
               weighted avg     0.9986    0.9871    0.9928       700

```
