"""Synthetic transaction data generator for augmenting training sets."""

from __future__ import annotations

import csv
import logging
import random
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SEED = 42
random.seed(SEED)

CATEGORIES = [
    "Salary/Income", "Rent", "Groceries", "Food & Dining", "Shopping/E-commerce",
    "Utilities", "Mobile/Internet/DTH", "Transport/Fuel/Travel", "Entertainment/Subscriptions",
    "Healthcare/Pharmacy", "Education", "Insurance", "Investments", "Loan/EMI",
    "Credit Card Payment", "Taxes", "Bank Charges/Fees", "Interest Earned",
    "ATM/Cash Withdrawal", "Cash Deposit", "Transfers", "Refunds/Reversals", "Cheque"
]

TEMPLATES = {
    "Salary/Income": [
        ("NEFT*{bank}*{ref}*{company}*SALARY", "CR", (25000, 250000)),
        ("SALARY CREDIT FOR {month} {company}", "CR", (30000, 200000)),
        ("PAYROLL DIRECT DEP {company}", "CR", (40000, 300000)),
        ("ACH CR- {company} SALARY-{ref}", "CR", (35000, 180000)),
    ],
    "Rent": [
        ("UPI/DR/{ref}/{landlord}/RENT FOR {month}", "DR", (10000, 60000)),
        ("IMPS/{ref}/{landlord}/HOUSE RENT", "DR", (8000, 50000)),
        ("NEFT*{bank}*{ref}*{landlord}*RENT", "DR", (12000, 75000)),
    ],
    "Groceries": [
        ("UPI/DR/{ref}/DMART/{bank}/dmart@{bank}", "DR", (500, 8000)),
        ("POS BIGBASKET ORDER {ref}", "DR", (400, 4500)),
        ("UPI/DR/{ref}/RELIANCE FRESH/{bank}/relfresh@{bank}", "DR", (300, 3500)),
        ("BLINKIT GROCERIES {ref}", "DR", (200, 2500)),
    ],
    "Food & Dining": [
        ("UPI/{ref}/Payment to Swiggy/swiggy@{bank}", "DR", (150, 2500)),
        ("UPI/DR/{ref}/ZOMATO/{bank}/zomato@{bank}", "DR", (200, 3000)),
        ("POS STARBUCKS COFFEE {city}", "DR", (250, 1500)),
        ("POS MCDONALDS RESTAURANT {ref}", "DR", (300, 1800)),
    ],
    "Shopping/E-commerce": [
        ("UPI/DR/{ref}/AMAZON INDIA/{bank}/amazon@{bank}", "DR", (250, 25000)),
        ("FLIPKART INTERNET ORDER {ref}", "DR", (350, 35000)),
        ("MYNTRA DESIGNS PVT LTD {ref}", "DR", (500, 8000)),
    ],
    "Utilities": [
        ("BBPS ELECTRICITY BILL {ref}", "DR", (500, 7000)),
        ("UPI/DR/{ref}/BESCOM POWER/{bank}/bescom@{bank}", "DR", (400, 5000)),
        ("IGL GAS BILL PAYMENT {ref}", "DR", (300, 2500)),
    ],
    "Mobile/Internet/DTH": [
        ("UPI/DR/{ref}/AIRTEL PREPAID/{bank}/airtel@{bank}", "DR", (199, 1499)),
        ("JIO RECHARGE MOBILITY {ref}", "DR", (149, 1199)),
        ("TATA PLAY DTH RECHARGE {ref}", "DR", (250, 999)),
    ],
    "Transport/Fuel/Travel": [
        ("UPI/DR/{ref}/UBER INDIA/{bank}/uber@{bank}", "DR", (100, 1500)),
        ("OLA CABS TRIP {ref}", "DR", (80, 1200)),
        ("IRCTC TRAIN TICKETING {ref}", "DR", (300, 4500)),
        ("HPCL PETROL PUMP {city}", "DR", (500, 4000)),
    ],
    "Entertainment/Subscriptions": [
        ("ACH D- NETFLIX SUBSCRIPTION", "DR", (199, 649)),
        ("SPOTIFY INDIA RECURRING", "DR", (119, 299)),
        ("HOTSTAR SUBSCRIPTION {ref}", "DR", (299, 1499)),
    ],
    "Healthcare/Pharmacy": [
        ("UPI/DR/{ref}/APOLLO PHARMACY/{bank}/apollo@{bank}", "DR", (150, 4500)),
        ("NETMEDS ORDER {ref}", "DR", (300, 3500)),
        ("1MG HEALTHCARE TECHNOLOGIES {ref}", "DR", (250, 5000)),
    ],
    "Education": [
        ("COLLEGE TUITION FEE {ref}", "DR", (5000, 120000)),
        ("COURSERA SUBSCRIPTION {ref}", "DR", (1500, 4500)),
        ("SCHOOL FEES QUARTER 1", "DR", (8000, 60000)),
    ],
    "Insurance": [
        ("ACH D- LIC OF INDIA PREMIUM", "DR", (2500, 25000)),
        ("HDFC LIFE INSURANCE PREMIUM", "DR", (3000, 40000)),
        ("STAR HEALTH INSURANCE {ref}", "DR", (5000, 35000)),
    ],
    "Investments": [
        ("ACH D- ZERODHA MF SIP-{ref}", "DR", (1000, 25000)),
        ("UPI/DR/{ref}/GROWW INVEST/{bank}/groww@{bank}", "DR", (500, 50000)),
        ("NIPPON INDIA MUTUAL FUND SIP", "DR", (1000, 20000)),
    ],
    "Loan/EMI": [
        ("ACH D- HDFC HOME LOAN EMI", "DR", (15000, 75000)),
        ("BAJAJ FINANCE AUTO EMI {ref}", "DR", (4000, 18000)),
        ("ICICI PERSONAL LOAN EMI", "DR", (5000, 35000)),
    ],
    "Credit Card Payment": [
        ("CRED PAYMENT {ref}", "DR", (2000, 85000)),
        ("AUTOPAY HDFC CREDIT CARD", "DR", (3500, 95000)),
        ("SBICARD PAYMENT VIA NEFT", "DR", (1500, 60000)),
    ],
    "Taxes": [
        ("CHALLAN 280 INCOME TAX {ref}", "DR", (5000, 150000)),
        ("GST PAYMENT PMT-06 {ref}", "DR", (10000, 250000)),
    ],
    "Bank Charges/Fees": [
        ("CONSOLIDATED CHARGES FOR {month}", "DR", (11.80, 250.0)),
        ("ANNUAL SMS CHARGES + GST", "DR", (17.70, 59.0)),
        ("DEBIT CARD AMC CHARGES", "DR", (150.0, 590.0)),
    ],
    "Interest Earned": [
        ("INT.PD: {bank} SB INTEREST", "CR", (25.0, 4500.0)),
        ("INTEREST CREDIT FOR QUARTER", "CR", (50.0, 12000.0)),
        ("FD INTEREST PAYOUT {ref}", "CR", (500.0, 25000.0)),
    ],
    "ATM/Cash Withdrawal": [
        ("ATM WDL/{ref}/{city}", "DR", (500, 10000)),
        ("ATM CASH WITHDRAWAL {ref}", "DR", (1000, 20000)),
    ],
    "Cash Deposit": [
        ("CASH DEPOSIT/{city} BR", "CR", (1000, 50000)),
        ("CDM CASH DEPOSIT {ref}", "CR", (2000, 45000)),
    ],
    "Transfers": [
        ("UPI/DR/{ref}/ARJUN IYER/{bank}/arjun@{bank}", "DR", (200, 15000)),
        ("IMPS/{ref}/PRIYA SHAH/TRANSFER", "DR", (500, 25000)),
        ("UPI/CR/{ref}/ROHAN MEHTA/{bank}/rohan@{bank}", "CR", (300, 12000)),
    ],
    "Refunds/Reversals": [
        ("UPI/CR/{ref}/AMAZON REFUND", "CR", (250, 4500)),
        ("REVERSAL OF CHG {ref}", "CR", (11.80, 590.0)),
        ("SWIGGY REFUND FOR ORDER {ref}", "CR", (150, 850)),
    ],
    "Cheque": [
        ("CLG/{ref}/MEERA PATEL", "DR", (5000, 65000)),
        ("CHEQUE DEPOSIT CLG {ref}", "CR", (10000, 85000)),
    ],
}

BANKS = ["sbi", "hdfc", "icici", "axis", "kotak", "pnb"]
CITIES = ["MUMBAI", "DELHI", "BANGALORE", "HYDERABAD", "CHENNAI", "PUNE", "AHMEDABAD"]
COMPANIES = ["INFOSYS LTD", "TCS LIMITED", "WIPRO TECH", "ACCENTURE", "BLUEWAVE SOFTWARE", "TECH MAHINDRA"]
PEOPLE = ["ARJUN IYER", "PRIYA SHAH", "ROHAN MEHTA", "KABIR SHARMA", "ANJALI DESAI", "VIKRAM NAIR"]
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def generate_synthetic_data(samples_per_category: int = 150, output_path: Optional[Path] = None) -> list[dict]:
    """Generate balanced synthetic dataset for training/augmentation."""
    records = []

    for cat in CATEGORIES:
        templates = TEMPLATES.get(cat, [("TXN {ref}", "DR", (100, 1000))])
        for _ in range(samples_per_category):
            tmpl, direction, (min_amt, max_amt) = random.choice(templates)
            ref_num = f"{random.randint(100000000000, 999999999999)}"
            bank = random.choice(BANKS)
            city = random.choice(CITIES)
            company = random.choice(COMPANIES)
            person = random.choice(PEOPLE)
            month = random.choice(MONTHS)

            desc = tmpl.format(
                ref=ref_num,
                bank=bank,
                city=city,
                company=company,
                landlord=person,
                month=month
            )
            amount = round(random.uniform(min_amt, max_amt), 2)

            records.append({
                "description": desc,
                "direction": direction,
                "amount": f"{amount:.2f}",
                "category": cat
            })

    random.shuffle(records)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["description", "direction", "amount", "category"])
            writer.writeheader()
            writer.writerows(records)
        logger.info("Saved %d synthetic samples to %s", len(records), output_path)

    return records

