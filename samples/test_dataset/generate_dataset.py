#!/usr/bin/env python3
"""Synthetic bank-statement test dataset generator. ALL DATA IS FICTIONAL."""
import os, io, csv, json, random, subprocess, shutil, glob, tempfile, datetime as dt
from decimal import Decimal, ROUND_HALF_UP
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfbase.pdfmetrics import stringWidth
from pypdf import PdfReader, PdfWriter
from PIL import Image
import numpy as np, cv2

ROOT = '/home/claude/bank_statement_test_dataset'
GT = f'{ROOT}/ground_truth'
for sub in ['text_based', 'scanned_image_based', 'mixed_and_multi', 'edge_cases',
            'other_document_types', 'ground_truth', 'classification_data']:
    os.makedirs(f'{ROOT}/{sub}', exist_ok=True)

D = Decimal
def q(x): return D(str(x)).quantize(D('0.01'), ROUND_HALF_UP)

def inr(x):
    s = f"{abs(q(x)):.2f}"; ip, fp = s.split('.')
    if len(ip) > 3:
        head, tail = ip[:-3], ip[-3:]; parts = []
        while len(head) > 2: parts.insert(0, head[-2:]); head = head[:-2]
        if head: parts.insert(0, head)
        ip = ','.join(parts) + ',' + tail
    return ip + '.' + fp
def west(x): return f"{abs(q(x)):,.2f}"

# --------------------------------------------------------------------------- data model
FIRST = ['Aarav','Vihaan','Ishita','Priya','Rohan','Neha','Karan','Anjali','Meera','Kabir','Sneha','Arjun','Divya','Rahul','Pooja']
LAST = ['Mehta','Shah','Patel','Sharma','Iyer','Nair','Gupta','Joshi','Desai','Reddy','Verma','Kulkarni']
CITIES = ['AHMEDABAD','MUMBAI','PUNE','BENGALURU','SURAT','DELHI','VADODARA']
EMPLOYERS = ['ACME TECHNOLOGIES PVT LTD','BLUEWAVE SOFTWARE LTD','NORTHSTAR SYSTEMS INDIA']
def person(r): return f"{r.choice(FIRST)} {r.choice(LAST)}".upper()

HB = {'icici':'ICIC','hdfcbank':'HDFC','ybl':'YESB','axisbank':'UTIB','sbi':'SBIN','paytm':'PYTM','okaxis':'UTIB','yesbank':'YESB','apl':'AMZN','oksbi':'SBIN'}
def hbank(h): return HB.get(h.split('@')[-1], 'ICIC')

MERCH = {
 'Food & Dining': [('SWIGGY','swiggy@icici'),('ZOMATO','zomato@hdfcbank'),('DOMINOS PIZZA','dominos@ybl'),('STARBUCKS','starbucks@axisbank'),('MCDONALDS','mcd@paytm'),('CAFE COFFEE DAY','ccd@okaxis')],
 'Groceries': [('BIGBASKET','bigbasket@hdfcbank'),('DMART','dmart@icici'),('RELIANCE FRESH','relfresh@sbi'),('BLINKIT','blinkit@ybl'),('ZEPTO','zepto@axisbank')],
 'Transport/Fuel/Travel': [('UBER INDIA','uber@icici'),('OLA CABS','olacabs@hdfcbank'),('INDIAN OIL PETROL','ioc@sbi'),('IRCTC','irctc@sbi'),('REDBUS','redbus@hdfcbank'),('HP PETROL PUMP','hppump@paytm')],
 'Shopping/E-commerce': [('AMAZON PAY','amazon@apl'),('FLIPKART','flipkart@axisbank'),('MYNTRA','myntra@axisbank'),('AJIO','ajio@icici')],
 'Healthcare/Pharmacy': [('APOLLO PHARMACY','apollo@icici'),('MEDPLUS','medplus@ybl'),('PRACTO','practo@hdfcbank'),('TATA 1MG','1mg@sbi')],
 'Entertainment/Subscriptions': [('NETFLIX','netflix@hdfcbank'),('SPOTIFY','spotify@icici'),('BOOKMYSHOW','bms@yesbank'),('HOTSTAR','hotstar@sbi')],
 'Mobile/Internet/DTH': [('JIO PREPAID','jio@icici'),('AIRTEL','airtel@axisbank'),('TATA PLAY','tataplay@hdfcbank')],
 'Education': [('UDEMY','udemy@icici'),('COURSERA','coursera@hdfcbank'),('BYJUS','byjus@ybl')],
}

def ref12(r): return str(r.randint(10**11, 10**12 - 1))

def narr(style, kind, name, handle, debit, r, extra='', verbose=False):
    """Return (narration, ref). Styles A..D mimic different bank narration conventions."""
    ref = ref12(r); d = 'DR' if debit else 'CR'
    ifsc = f"{r.choice(['HDFC','ICIC','SBIN','UTIB'])}0{r.randint(0,999999):06d}"
    card = f"{r.choice(['4012','5234','4147'])}XXXXXX{r.randint(1000,9999)}"
    city = r.choice(CITIES); atm = f"{r.randint(100000,999999)}"; chq = f"{r.randint(100000,999999)}"
    ex = f" {extra}" if extra else ''
    if kind in ('upi_m', 'upi_p'):
        hb = hbank(handle)
        s = {'A': f"UPI/{d}/{ref}/{name}/{hb}/{handle}",
             'B': f"UPI-{name}-{handle}-{hb}0000001-{ref}-{'PAYMENT' if kind=='upi_m' else 'UPI'}",
             'C': f"UPI/{ref}/{'Payment to' if debit else 'Payment from'} {name.title()}/{handle}/{hb} Bank",
             'D': f"UPI-{name}-{ref}"}[style]; rf = ref
    elif kind == 'neft':
        s = {'A': f"NEFT*{ifsc}*N{ref[:9]}*{name}*{extra}", 'B': f"NEFT {d}-{ifsc}-{name}-{extra}-N{ref}",
             'C': f"NEFT-{ifsc}-{name}-{ref[:10]}{ex}", 'D': f"NEFT {name}{ex} {ref[:8]}"}[style]; rf = ref[:9]
    elif kind == 'imps':
        s = {'A': f"IMPS/{ref}/{name}/{extra}", 'B': f"IMPS-{ref}-{name}-{extra}",
             'C': f"MMT/IMPS/{ref}/{extra or 'TRANSFER'}/{name}", 'D': f"IMPS {name}{ex}"}[style]; rf = ref
    elif kind == 'ach':
        s = {'A': f"ACH D- {name}-{ref[:8]}", 'B': f"ACH/{name}/{ref[:8]}",
             'C': f"NACH-DR-{name}-{ref[:8]}", 'D': f"ECS {name}"}[style]; rf = ref[:8]
    elif kind == 'atm':
        s = {'A': f"ATM WDL/{atm}/{city}", 'B': f"ATW-{card}-{atm}-{city}",
             'C': f"NWD-{card}-{atm}", 'D': f"ATM CASH {atm}"}[style]; rf = atm
    elif kind == 'pos':
        s = {'A': f"POS {card} {name} {city}", 'B': f"POS {card} {name} {city} IN",
             'C': f"PUR/{name}/{city}/{card[-4:]}", 'D': f"POS {name}"}[style]; rf = ''
    elif kind == 'chq':
        s = {'A': f"CHEQUE PAID {chq} {name}", 'B': f"CHQ PAID-{chq}-{name}",
             'C': f"CLG/{chq}/{name}", 'D': f"CHQ {chq} {name}"}[style]; rf = chq
    elif kind == 'cash':
        s = {'A': f"CASH DEPOSIT/{city} BR", 'B': f"CASH DEP SELF-{city}",
             'C': f"BY CASH-{city} BRANCH", 'D': "CDM CASH DEPOSIT"}[style]; rf = ''
    elif kind == 'bill':
        s = {'A': f"BIL/BPAY/{ref[:9]}/{name}", 'B': f"BILLPAY-{name}-{ref[:8]}",
             'C': f"BIL/INFT/{ref[:9]}/{name}", 'D': f"BILLPAY {name}"}[style]; rf = ref[:9]
    else:  # int / chg / free text
        s = name; rf = ''
    if verbose and kind not in ('int', 'chg'):
        s += f" TXN ID {ref[:10]} REMARKS {extra or 'NA'} VIA MOBILE BANKING"
    return s, rf

CAT_RANDOM = [  # weight, category, kind, lo, hi
 (24,'Food & Dining','m',120,950),(14,'Groceries','m',250,3500),(12,'Transport/Fuel/Travel','m',60,2800),
 (10,'Shopping/E-commerce','m',400,6500),(5,'Healthcare/Pharmacy','m',90,1800),(4,'Entertainment/Subscriptions','m',149,1200),
 (6,'ATM/Cash Withdrawal','atm',1000,10000),(8,'Transfers','p_out',200,9000),(5,'Transfers','p_in',300,7500),
 (2,'Refunds/Reversals','refund',100,2500),(1.5,'Cash Deposit','cash',5000,40000),(1,'Education','m',500,8000),
 (0.5,'Taxes','tax',1500,25000),(1,'Cheque','chq',2000,30000)]

def build_events(r, start, end, style, rate=22, verbose=False, spend_scale=1.0, salary=True):
    ev = []; landlord = person(r)
    def mk(day, cat, kind, name='', handle='', debit=True, amt=0, extra='', flag=''):
        s, rf = narr(style, kind, name, handle, debit, r, extra, verbose)
        ev.append(dict(date=day, cat=cat, narr=s, ref=rf, debit=debit, amt=q(amt), flag=flag, name=name))
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        Dd = lambda n: dt.date(y, m, min(n, 28))
        if salary: mk(Dd(1), 'Salary/Income', 'neft', r.choice(EMPLOYERS), '', False, r.randint(62, 95) * 1000 + r.choice([0, 0, 500, 250]), 'SALARY')
        mk(Dd(3), 'Rent', 'imps', landlord, '', True, 18000, 'RENT')
        mk(Dd(5), 'Investments', 'ach', 'ZERODHA MF SIP', '', True, 5000)
        mk(Dd(7), 'Utilities', 'upi_m', 'TORRENT POWER', 'torrentpower@icici', True, r.randint(900, 3200))
        n, h = r.choice(MERCH['Mobile/Internet/DTH']); mk(Dd(8), 'Mobile/Internet/DTH', 'upi_m', n, h, True, r.choice([299, 399, 719, 859]))
        mk(Dd(12), 'Entertainment/Subscriptions', 'upi_m', 'NETFLIX', 'netflix@hdfcbank', True, 649)
        mk(Dd(15), 'Loan/EMI', 'ach', 'HOME LOAN EMI HDFC LTD', '', True, 14250)
        mk(Dd(18), 'Credit Card Payment', 'bill', 'BILLDESK CREDIT CARD', '', True, r.randint(4000, 22000))
        mk(Dd(20), 'Insurance', 'ach', 'LIC OF INDIA PREMIUM', '', True, 2100)
        mk(Dd(28), 'Bank Charges/Fees', 'chg', r.choice(['SMS ALERT CHARGES', 'DEBIT CARD ANNUAL FEE GST', 'ATM TXN CHARGES']), '', True, r.choice([17.70, 29.50, 236.00]))
        if m in (3, 6, 9, 12): mk(Dd(28), 'Interest Earned', 'int', 'INT.PD SAVINGS ACCOUNT INTEREST CREDIT', '', False, r.randint(300, 900) + r.random())
        weights = [c[0] for c in CAT_RANDOM]
        for _ in range(rate):
            w, cat, kind, lo, hi = r.choices(CAT_RANDOM, weights)[0]
            day = Dd(r.randint(1, 28)); amt = q(r.randint(int(lo * 100), int(hi * 100)) / 100 * (spend_scale if kind in ('m','p_out','chq') else 1))
            if amt < 20: amt = q(20 + r.random() * 30)
            if kind == 'm':
                nm, hd = r.choice(MERCH[cat])
                if cat in ('Groceries', 'Transport/Fuel/Travel', 'Shopping/E-commerce') and r.random() < 0.25: mk(day, cat, 'pos', nm, hd, True, amt)
                else: mk(day, cat, 'upi_m', nm, hd, True, amt)
            elif kind == 'atm': mk(day, cat, 'atm', '', '', True, (int(amt) // 500 + 1) * 500)
            elif kind in ('p_out', 'p_in'):
                p = person(r); hd = p.lower().replace(' ', '')[:10] + '@okaxis'
                debit = kind == 'p_out'
                if r.random() < 0.3: mk(day, cat, 'imps', p, '', debit, amt, r.choice(['FAMILY', 'SPLIT', 'GIFT', 'LOAN RETURN']))
                else: mk(day, cat, 'upi_p', p, hd, debit, amt)
            elif kind == 'refund':
                nm, hd = r.choice(MERCH['Shopping/E-commerce']); mk(day, cat, 'upi_m', nm + ' REFUND', hd, False, amt)
            elif kind == 'cash': mk(day, cat, 'cash', '', '', False, (int(amt) // 500) * 500)
            elif kind == 'tax': mk(day, cat, 'neft', r.choice(['CBDT INCOME TAX', 'GST PAYMENT CBIC']), '', True, amt, 'TAX')
            elif kind == 'chq': mk(day, cat, 'chq', person(r), '', True, (int(amt) // 100) * 100)
        m += 1
        if m == 13: m = 1; y += 1
    ev = [e for e in ev if start <= e['date'] <= end]
    ev.sort(key=lambda e: e['date'])
    return ev

def apply_balances(ev, opening, skip_overdraft=True):
    bal = q(opening); out = []
    for e in ev:
        if e['debit'] and skip_overdraft and e['amt'] > bal - 100: continue
        bal = bal - e['amt'] if e['debit'] else bal + e['amt']
        out.append(dict(date=e['date'], narr=e['narr'], ref=e['ref'], debit=e['amt'] if e['debit'] else None,
                        credit=None if e['debit'] else e['amt'], balance=bal, cat=e['cat'], flag=e.get('flag', '')))
    return out

def make_account(r, bank_key):
    b = {'SNB': ('Sample National Bank', 'SNBX', 'Ahmedabad Navrangpura'), 'DCB': ('Demo Co-operative Bank', 'DMOB', 'Pune Camp'),
         'TUB': ('Test Union Bank', 'TUBK', 'Mumbai Andheri East'), 'MFB': ('Mock Finance Bank Ltd', 'MFBL', 'Bengaluru Indiranagar')}[bank_key]
    return dict(bank_name=b[0], holder=person(r), acc_no=str(r.randint(10**11, 10**12 - 1)), ifsc=f"{b[1]}0{r.randint(0,999999):06d}",
                micr=str(r.randint(10**8, 10**9 - 1)), branch=b[2], cust_id=str(r.randint(10**8, 10**9 - 1)), acc_type='Savings Account',
                address=f"{r.randint(1,99)}, Sample Residency, {b[2].split()[0]} - {r.randint(380001, 411099)}")

# --------------------------------------------------------------------------- rendering
def wrap(text, font, fs, width):
    lines, cur = [], ''
    def hard(tok):
        out, s = [], ''
        for ch in tok:
            if stringWidth(s + ch, font, fs) > width: out.append(s); s = ch
            else: s += ch
        out.append(s); return out
    for w in text.split(' '):
        t = (cur + ' ' + w) if cur else w
        if stringWidth(t, font, fs) <= width: cur = t
        else:
            if cur: lines.append(cur); cur = ''
            if stringWidth(w, font, fs) <= width: cur = w
            else: p = hard(w); lines += p[:-1]; cur = p[-1]
    if cur: lines.append(cur)
    return lines or ['']

def cell(t, key, cfg):
    nf = inr if cfg['num'] == 'inr' else west; kind = cfg['kind']
    if key == 'date': return t['date'].strftime(cfg['date_fmt'])
    if key == 'vdate': return t.get('vdate', t['date']).strftime(cfg['date_fmt'])
    if key == 'ref': return t.get('ref', '')
    deb, cre = t['debit'], t['credit']
    if t.get('print_swap'): deb, cre = cre, deb
    if key == 'debit': return nf(deb) if deb is not None else ('0.00' if cfg.get('zero_fill') else '')
    if key == 'credit': return nf(cre) if cre is not None else ('0.00' if cfg.get('zero_fill') else '')
    if key == 'amount':
        isd = deb is not None; v = deb if isd else cre
        if kind == 'amount_drcr': return f"{nf(v)} {'Dr' if isd else 'Cr'}"
        if kind == 'signed': return ('-' if isd else '+') + nf(v)
        if kind == 'bracket': return f"({nf(v)})" if isd else nf(v)
    if key == 'balance':
        if t['balance'] is None: return ''
        return f"{nf(t['balance'])} Cr" if kind == 'amount_drcr' else nf(t['balance'])
    return ''

def render_statement(c, cfg, acct, txns, meta):
    W, H = cfg['page']; cols = cfg['cols']; tw = sum(k[2] for k in cols); x0 = (W - tw) / 2
    font, fb, fs, lh, pad = cfg['font'], cfg['fbold'], cfg['fs'], cfg['lh'], cfg['pad']
    nw = [k[2] for k in cols if k[0] == 'narr'][0] - 6
    layouts = [wrap(t['narr'], font, fs, nw) for t in txns]
    th = 20 if cfg.get('dashes') else 16
    carry_h = (lh + 2 * pad + 2) if cfg.get('carry') else 0
    top1 = H - 215; topn = H - 62; bottom = 48 + carry_h
    pages, cur, y = [], [], top1 - th
    for i, lines in enumerate(layouts):
        rh = len(lines) * lh + 2 * pad
        if y - rh < bottom and cur:
            pages.append(cur); cur = []; y = topn - th - carry_h
        cur.append(i); y -= rh
    pages.append(cur)
    summary_own = (y - 125) < 48
    N = len(pages) + (1 if summary_own else 0)
    c.setLineWidth(0.4); c.setStrokeColorRGB(0.35, 0.35, 0.35)

    def footer(pn):
        c.setFillColorRGB(0.3, 0.3, 0.3); c.setFont(cfg['font'], 6.5)
        c.drawString(x0, 28, 'Computer generated statement. SAMPLE / FICTIONAL DATA - FOR SOFTWARE TESTING ONLY.')
        c.drawRightString(W - x0, 28, f'Page {pn} of {N}'); c.setFillColorRGB(0, 0, 0)

    def compact_header():
        c.setFont(fb, 9); c.drawString(x0, H - 38, acct['bank_name'])
        c.setFont(font, 7.5); c.drawRightString(W - x0, H - 38, f"{cfg['title']} (contd.)   A/c: {acct['acc_no']}")
        c.line(x0, H - 44, W - x0, H - 44)

    def first_header():
        L = cfg['labels']
        OFF = max(stringWidth(x + ':', fb, 8) for x in list(L.values()) + ['Statement Period', 'Opening Balance', 'Total Amount Due']) + 6
        if cfg['header'] == 'band':
            c.setFillColorRGB(*cfg['color']); c.rect(0, H - 60, W, 60, fill=1, stroke=0); c.setFillColorRGB(1, 1, 1)
            c.setFont(fb, 16); c.drawString(x0, H - 30, acct['bank_name']); c.setFont(font, 9); c.drawString(x0, H - 46, cfg['title'])
            c.setFillColorRGB(0, 0, 0); y = H - 82
        else:
            c.setFont(fb, 12); c.drawString(x0, H - 38, acct['bank_name'].upper()); c.setFont(font, 9); c.drawString(x0, H - 52, cfg['title'])
            c.line(x0, H - 58, W - x0, H - 58); y = H - 78
        keys = [k for k in cfg['items'] if k != 'address']; c.setFont(font, 8)
        for i, k in enumerate(keys):
            cx = x0 if i % 2 == 0 else x0 + tw / 2; cy = y - (i // 2) * 13
            c.setFont(fb, 8); c.drawString(cx, cy, f"{L[k]}:"); c.setFont(font, 8); c.drawString(cx + OFF, cy, str(acct[k]))
        y -= ((len(keys) + 1) // 2) * 13
        if 'address' in cfg['items']:
            c.setFont(fb, 8); c.drawString(x0, y, f"{L['address']}:"); c.setFont(font, 8); c.drawString(x0 + OFF, y, acct['address']); y -= 13
        c.setFont(fb, 8); c.drawString(x0, y, 'Statement Period:'); c.setFont(font, 8)
        c.drawString(x0 + OFF, y, f"{meta['from'].strftime(cfg['date_fmt'])} to {meta['to'].strftime(cfg['date_fmt'])}"); y -= 13
        if cfg.get('cc'):
            c.setFont(fb, 8); c.drawString(x0, y, 'Total Amount Due:'); c.setFont(font, 8)
            c.drawString(x0 + OFF, y, f"Rs. {inr(meta['total_due'])}    Minimum Due: Rs. {inr(meta['min_due'])}    Payment Due Date: {meta['due_date'].strftime(cfg['date_fmt'])}")
        else:
            c.setFont(fb, 8); c.drawString(x0, y, 'Opening Balance:'); c.setFont(font, 8); c.drawString(x0 + OFF, y, f"Rs. {nfm(meta['opening'])}")

    nfm = inr if cfg['num'] == 'inr' else west

    def table_header(y):
        c.setFont(fb, fs)
        if cfg['grid']: c.setFillColorRGB(0.86, 0.91, 0.97); c.rect(x0, y - th, tw, th, fill=1, stroke=0); c.setFillColorRGB(0, 0, 0)
        x = x0
        for key, title, w, al in cols:
            if cfg['grid']: c.rect(x, y - th, w, th, stroke=1, fill=0)
            ty = y - 11 if not cfg.get('dashes') else y - 8
            (c.drawRightString(x + w - 3, ty, title) if al == 'r' else c.drawString(x + 3, ty, title)); x += w
        if cfg.get('dashes'): c.setFont(font, fs); c.drawString(x0, y - 17, '-' * int(tw / stringWidth('-', font, fs)))
        c.setFont(font, fs)

    def draw_row(y_top, lines, t, idx):
        rh = len(lines) * lh + 2 * pad
        if cfg.get('zebra') and idx % 2 == 0: c.setFillColorRGB(0.94, 0.94, 0.94); c.rect(x0, y_top - rh, tw, rh, fill=1, stroke=0); c.setFillColorRGB(0, 0, 0)
        x = x0; c.setFont(font, fs)
        for key, title, w, al in cols:
            if cfg['grid']: c.rect(x, y_top - rh, w, rh, stroke=1, fill=0)
            if key == 'narr':
                for i, ln in enumerate(lines): c.drawString(x + 3, y_top - pad - fs * 0.85 - i * lh, ln)
            else:
                txt = cell(t, key, cfg)
                if txt: (c.drawRightString(x + w - 3, y_top - pad - fs * 0.85, txt) if al == 'r' else c.drawString(x + 3, y_top - pad - fs * 0.85, txt))
            x += w
        return rh

    def carry_row(y_top, label, bal):
        h = carry_h
        x = x0; c.setFont(fb, fs)
        for key, title, w, al in cols:
            if cfg['grid']: c.rect(x, y_top - h, w, h, stroke=1, fill=0)
            if key == 'narr': c.drawString(x + 3, y_top - pad - fs * 0.85, label)
            if key == 'balance': c.drawRightString(x + w - 3, y_top - pad - fs * 0.85, nfm(bal) + (' Cr' if cfg['kind'] == 'amount_drcr' else ''))
            x += w
        c.setFont(font, fs)

    def summary(y):
        y -= 18; c.setFont(fb, 9); c.drawString(x0, y, 'Statement Summary'); c.setFont(font, 8); y -= 14
        deb = sum((t['debit'] or 0) for t in txns if not t.get('dup')); cre = sum((t['credit'] or 0) for t in txns if not t.get('dup'))
        td, tc = meta.get('total_debits', deb), meta.get('total_credits', cre)
        rows = ([('Previous Balance', '-'), ('Total Purchases / Debits', f"Rs. {inr(td)}"), ('Total Payments / Credits', f"Rs. {inr(tc)}"), ('Total Amount Due', f"Rs. {inr(meta['total_due'])}")]
                if cfg.get('cc') else
                [('Opening Balance', f"Rs. {nfm(meta['opening'])}"), ('Total Debits', f"Rs. {nfm(td)}"), ('Total Credits', f"Rs. {nfm(tc)}"), ('Closing Balance', f"Rs. {nfm(meta['closing'])}")])
        for lab, v in rows: c.drawString(x0, y, lab); c.drawString(x0 + 150, y, v); y -= 12
        y -= 8; c.setFont(font, 6.5)
        for ln in ['Please examine this statement at once. If no error is reported within 15 days it will be considered correct.',
                   'Closing balance includes funds under hold, if any. This statement does not require a signature.',
                   'For queries contact your home branch. (Fictional text - this bank does not exist.)']:
            c.drawString(x0, y, ln); y -= 9

    for pi, idxs in enumerate(pages):
        if pi == 0: first_header(); y = top1
        else: compact_header(); y = topn
        table_header(y); y -= th
        if pi > 0 and cfg.get('carry'):
            carry_row(y, 'BROUGHT FORWARD', txns[pages[pi - 1][-1]]['balance']); y -= carry_h
        for i in idxs: y -= draw_row(y, layouts[i], txns[i], i)
        if pi < len(pages) - 1 and cfg.get('carry'): carry_row(y, 'CARRIED FORWARD', txns[idxs[-1]]['balance'])
        if pi == len(pages) - 1 and not summary_own: summary(y)
        footer(pi + 1); c.showPage()
    if summary_own:
        compact_header(); summary(H - 70); footer(N); c.showPage()
    return N

# --------------------------------------------------------------------------- layouts
LABELS_A = dict(holder='Account Name', acc_no='Account Number', ifsc='IFS Code', micr='MICR Code', branch='Branch', cust_id='CIF No.', acc_type='Account Type', address='Address')
LABELS_B = dict(holder='Customer Name', acc_no='A/C No.', ifsc='IFSC', micr='MICR', branch='Branch Name', cust_id='Customer ID', acc_type='Product', address='Mailing Address')
LABELS_C = dict(holder='Name', acc_no='Acct No', ifsc='IFSC Code', micr='MICR', branch='Branch', cust_id='Cust Id', acc_type='Type', address='Address')
def W_(*a): return list(a)
CFG = {
 'L1': dict(bank='SNB', page=A4, font='Helvetica', fbold='Helvetica-Bold', fs=8, lh=10, pad=3, grid=True, header='band', color=(0.05, 0.25, 0.5),
            kind='dr_cr', date_fmt='%d/%m/%Y', num='inr', carry=True, title='Statement of Account', labels=LABELS_A,
            items=['holder', 'acc_no', 'ifsc', 'micr', 'branch', 'cust_id', 'acc_type', 'address'],
            cols=[('date', 'Txn Date', 52, 'l'), ('vdate', 'Value Date', 52, 'l'), ('narr', 'Description', 178, 'l'), ('ref', 'Ref No./Chq No.', 68, 'l'), ('debit', 'Debit', 58, 'r'), ('credit', 'Credit', 58, 'r'), ('balance', 'Balance', 62, 'r')]),
 'L2': dict(bank='DCB', page=A4, font='Helvetica', fbold='Helvetica-Bold', fs=8, lh=10, pad=3, grid=False, zebra=True, header='band', color=(0.5, 0.1, 0.15),
            kind='amount_drcr', date_fmt='%d %b %Y', num='inr', title='Account Statement', labels=LABELS_B,
            items=['holder', 'acc_no', 'ifsc', 'branch', 'cust_id', 'acc_type', 'address'],
            cols=[('date', 'Transaction Date', 72, 'l'), ('narr', 'Transaction Remarks', 268, 'l'), ('amount', 'Amount (INR)', 90, 'r'), ('balance', 'Balance (INR)', 95, 'r')]),
 'L3': dict(bank='TUB', page=A4, font='Helvetica', fbold='Helvetica-Bold', fs=7.5, lh=9.5, pad=3, grid=True, header='plain', color=(0, 0, 0),
            kind='dr_cr', date_fmt='%d-%m-%y', num='inr', title='Detailed Statement', labels=LABELS_C,
            items=['holder', 'acc_no', 'ifsc', 'branch', 'cust_id', 'address'],
            cols=[('date', 'Date', 46, 'l'), ('narr', 'Narration', 168, 'l'), ('ref', 'Chq./Ref.No.', 62, 'l'), ('debit', 'Withdrawal Amt.', 62, 'r'), ('credit', 'Deposit Amt.', 62, 'r'), ('balance', 'Closing Balance', 70, 'r')]),
 'L4': dict(bank='MFB', page=A4, font='Courier', fbold='Courier-Bold', fs=7.5, lh=9.5, pad=2, grid=False, header='plain', color=(0, 0, 0), dashes=True,
            kind='signed', date_fmt='%Y-%m-%d', num='inr', title='STATEMENT OF ACCOUNT', labels=LABELS_C,
            items=['holder', 'acc_no', 'ifsc', 'branch', 'acc_type'],
            cols=[('date', 'DATE', 62, 'l'), ('narr', 'PARTICULARS', 270, 'l'), ('amount', 'AMOUNT', 85, 'r'), ('balance', 'BALANCE', 95, 'r')]),
 'L5': dict(bank='SNB', page=landscape(A4), font='Helvetica', fbold='Helvetica-Bold', fs=8, lh=10, pad=3, grid=True, header='plain', color=(0, 0, 0),
            kind='dr_cr', date_fmt='%d/%m/%Y', num='inr', zero_fill=True, title='Passbook Entries (Printed Statement)', labels=LABELS_A,
            items=['holder', 'acc_no', 'ifsc', 'branch'],
            cols=[('date', 'Date', 62, 'l'), ('narr', 'Particulars', 300, 'l'), ('ref', 'Chq.No', 70, 'l'), ('debit', 'Withdrawals', 80, 'r'), ('credit', 'Deposits', 80, 'r'), ('balance', 'Balance', 90, 'r')]),
 'L6': dict(bank='MFB', page=A4, font='Helvetica', fbold='Helvetica-Bold', fs=8, lh=10, pad=3.5, grid=False, header='band', color=(0.1, 0.45, 0.3),
            kind='bracket', date_fmt='%b %d, %Y', num='west', title='Digital Account Statement', labels=LABELS_B,
            items=['holder', 'acc_no', 'ifsc', 'acc_type'],
            cols=[('date', 'Date', 75, 'l'), ('narr', 'Details', 270, 'l'), ('amount', 'Amount', 90, 'r'), ('balance', 'Balance', 90, 'r')]),
 'CC': dict(bank='TUB', page=A4, font='Helvetica', fbold='Helvetica-Bold', fs=8, lh=10, pad=3, grid=True, header='band', color=(0.3, 0.2, 0.5), cc=True,
            kind='amount_drcr', date_fmt='%d/%m/%Y', num='inr', title='Credit Card Statement', labels=dict(LABELS_A, acc_no='Card Number'),
            items=['holder', 'acc_no', 'address'],
            cols=[('date', 'Date', 70, 'l'), ('narr', 'Transaction Details', 335, 'l'), ('amount', 'Amount (Rs.)', 120, 'r')]),
}
CFG['L7_verbose'] = dict(CFG['L3'], fs=7.5)

# --------------------------------------------------------------------------- ground truth / helpers
def write_gt(name, txns, acct, meta, extra=None):
    with open(f'{GT}/{name}.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['date', 'description', 'ref', 'debit', 'credit', 'balance', 'category', 'edge_flag'])
        for t in txns:
            w.writerow([t['date'].isoformat(), t['narr'], t.get('ref', ''), '' if t['debit'] is None else t['debit'], '' if t['credit'] is None else t['credit'],
                        '' if t['balance'] is None else t['balance'], t['cat'], t.get('flag', '')])
    info = dict(account={k: v for k, v in acct.items()}, period_from=str(meta['from']), period_to=str(meta['to']),
                opening_balance=str(meta.get('opening', '')), closing_balance=str(meta.get('closing', '')), row_count=len(txns))
    if extra: info.update(extra)
    json.dump(info, open(f'{GT}/{name}.json', 'w'), indent=2)

def make_meta(txns, start, end, opening):
    return dict(**{'from': start, 'to': end}, opening=q(opening), closing=txns[-1]['balance'])

def build_pdf(path, parts):
    """parts: list of (cfg, acct, txns, meta)."""
    c = canvas.Canvas(path, pagesize=parts[0][0]['page']); c.setTitle('Sample statement'); c.setAuthor('Synthetic test data')
    tot = 0
    for cfg, acct, txns, meta in parts:
        c.setPageSize(cfg['page']); tot += render_statement(c, cfg, acct, txns, meta)
    c.save(); return tot

def raster(pdf, dpi, first=None, last=None):
    tmp = tempfile.mkdtemp(); cmd = ['pdftoppm', '-r', str(dpi), '-png']
    if first: cmd += ['-f', str(first), '-l', str(last or first)]
    subprocess.run(cmd + [pdf, f'{tmp}/p'], check=True)
    ims = [Image.open(p).copy() for p in sorted(glob.glob(f'{tmp}/p*.png'))]; shutil.rmtree(tmp); return ims

def jpg(im, qual):
    b = io.BytesIO(); im.save(b, 'JPEG', quality=qual); b.seek(0); return Image.open(b).convert(im.mode)

def rot(a, ang, border):
    h, w = a.shape[:2]; M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    return cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border)

def degrade(im, mode, seed=0):
    r = np.random.RandomState(seed)
    if mode == 'clean': return im.convert('L'), 1.0
    if mode in ('skew_noise', 'rot90'):
        a = np.array(im.convert('L')).astype(np.float32); a = rot(a, 1.7, 255); a = cv2.GaussianBlur(a, (3, 3), 0.9)
        a = a * 0.78 + 38 + r.normal(0, 9, a.shape); m = r.rand(*a.shape); a[m < 0.0015] = 0; a[m > 0.9985] = 255
        out = jpg(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)), 55)
        if mode == 'rot90': out = out.rotate(-90, expand=True, fillcolor=255)
        return out, 1.0
    if mode == 'photo':
        a = np.array(im.convert('RGB')).astype(np.float32); h, w = a.shape[:2]
        a *= np.array([1.0, 0.965, 0.88], dtype=np.float32)
        shade = (np.linspace(0.62, 1.0, w)[None, :] * np.linspace(0.9, 1.0, h)[:, None])[..., None]
        yy, xx = np.mgrid[0:h, 0:w]; d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
        a = np.clip(a * shade * (1 - 0.18 * np.clip(d, 0, 1.4) ** 2)[..., None], 0, 255).astype(np.uint8)
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]]); dst = np.float32([[w * .03, h * .015], [w * .985, 0], [w * .965, h * .985], [w * .01, h * .99]])
        a = cv2.warpPerspective(a, cv2.getPerspectiveTransform(src, dst), (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(48, 44, 40))
        a = rot(a, -2.3, (48, 44, 40)); a = cv2.GaussianBlur(a, (0, 0), 1.1)
        a = np.clip(a.astype(np.float32) + r.normal(0, 5, a.shape), 0, 255).astype(np.uint8)
        return jpg(Image.fromarray(a).resize((int(w * .75), int(h * .75)), Image.LANCZOS), 45), 0.75
    if mode == 'fax':
        g = cv2.GaussianBlur(np.array(im.convert('L')), (3, 3), 0); _, b = cv2.threshold(g, 170, 255, cv2.THRESH_BINARY); b = rot(b, -0.8, 255)
        for _ in range(8):
            y = r.randint(0, b.shape[0] - 2); x1 = r.randint(0, b.shape[1] // 2); b[y:y + 1, x1:x1 + r.randint(200, 900)] = 0
        return Image.fromarray(b).convert('1'), 1.0

def make_scanned(src_pdf, out_pdf, mode, dpi=200, seed=0):
    ims = raster(src_pdf, dpi); outs = []; eff = dpi
    for i, im in enumerate(ims):
        o, s = degrade(im, mode, seed + i); outs.append(o); eff = dpi * s
    outs[0].save(out_pdf, 'PDF', save_all=True, append_images=outs[1:], resolution=eff)

# --------------------------------------------------------------------------- build the dataset
def scenario(seed, cfgkey, start, end, opening, rate=22, verbose=False, bank=None, **kw):
    r = random.Random(seed); acct = make_account(r, bank or CFG[cfgkey]['bank'])
    ev = build_events(r, start, end, kw.pop('style', 'A'), rate=rate, verbose=verbose, **kw)
    return r, acct, ev

def std(name, seed, cfgkey, style, start, end, opening, rate=22, verbose=False, folder='text_based', **kw):
    r, acct, ev = scenario(seed, cfgkey, start, end, opening, rate, verbose, style=style, **kw)
    txns = apply_balances(ev, opening); meta = make_meta(txns, start, end, opening)
    path = f'{ROOT}/{folder}/{name}.pdf'; n = build_pdf(path, [(CFG[cfgkey], acct, txns, meta)])
    write_gt(name, txns, acct, meta); print(f'{name}: {len(txns)} rows, {n} pages'); return path, txns, acct, meta

d = dt.date
# ---- text based
p1, t1, a1, m1 = std('01_text_bordered_debit_credit_carry_forward', 101, 'L1', 'A', d(2026, 4, 1), d(2026, 6, 30), 38450, rate=24)
p2, t2, a2, m2 = std('02_text_borderless_amount_with_Dr_Cr_suffix', 102, 'L2', 'B', d(2026, 1, 1), d(2026, 3, 31), 52210.55, rate=22)
p3, t3, a3, m3 = std('03_text_multiline_narration_withdrawal_deposit', 103, 'L3', 'C', d(2026, 5, 1), d(2026, 7, 31), 21900, rate=20, verbose=True)
p4, t4, a4, m4 = std('04_text_monospace_whitespace_aligned_signed_amounts', 104, 'L4', 'D', d(2026, 7, 1), d(2026, 9, 30), 64300, rate=22)
p5, t5, a5, m5 = std('05_text_landscape_passbook_style_zero_filled', 105, 'L5', 'A', d(2026, 2, 1), d(2026, 4, 30), 15800, rate=20)
p6, t6, a6, m6 = std('06_text_bracket_negatives_western_number_format', 106, 'L6', 'C', d(2026, 6, 1), d(2026, 8, 31), 33000, rate=22)

# ---- scanned (image only) versions - each a different layout + a different kind of degradation
def scanned(name, seed, cfgkey, style, start, end, opening, mode, dpi, rate=22, verbose=False):
    path, txns, acct, meta = std(name + '__src', seed, cfgkey, style, start, end, opening, rate, verbose, folder='mixed_and_multi')
    out = f'{ROOT}/scanned_image_based/{name}.pdf'; make_scanned(path, out, mode, dpi, seed)
    os.rename(f'{GT}/{name}__src.csv', f'{GT}/{name}.csv'); os.rename(f'{GT}/{name}__src.json', f'{GT}/{name}.json'); os.remove(path)
scanned('07_scanned_clean_200dpi', 201, 'L1', 'B', d(2026, 4, 1), d(2026, 6, 30), 41200, 'clean', 200, rate=18)
scanned('08_scanned_skewed_noisy_low_contrast', 202, 'L2', 'A', d(2026, 1, 1), d(2026, 3, 31), 28750, 'skew_noise', 200, rate=18)
scanned('09_scanned_phone_photo_shadow_perspective', 203, 'L3', 'D', d(2026, 5, 1), d(2026, 6, 30), 19000, 'photo', 200, rate=16, verbose=False)
scanned('10_scanned_fax_black_white_low_res', 204, 'L4', 'C', d(2026, 7, 1), d(2026, 8, 31), 47000, 'fax', 150, rate=14)
scanned('11_scanned_rotated_90_degrees', 205, 'L6', 'B', d(2026, 6, 1), d(2026, 7, 31), 36500, 'rot90', 200, rate=14)

# ---- mixed pages: statement where some pages are text and some are scanned images
r = random.Random(301); acct = make_account(r, 'SNB'); ev = build_events(r, d(2026, 4, 1), d(2026, 6, 30), 'A', rate=40)
txns = apply_balances(ev, 44000); meta = make_meta(txns, d(2026, 4, 1), d(2026, 6, 30), 44000)
tmp = f'{ROOT}/mixed_and_multi/_tmp.pdf'; n = build_pdf(tmp, [(CFG['L1'], acct, txns, meta)])
scan_pages = {2, n}  # 1-indexed pages replaced by noisy scans
wr = PdfWriter(); rd = PdfReader(tmp)
for pg in range(1, n + 1):
    if pg in scan_pages:
        im, s = degrade(raster(tmp, 200, pg)[0], 'skew_noise', pg); tp = f'{ROOT}/mixed_and_multi/_img{pg}.pdf'; im.save(tp, 'PDF', resolution=200)
        wr.add_page(PdfReader(tp).pages[0]); os.remove(tp)
    else: wr.add_page(rd.pages[pg - 1])
wr.write(open(f'{ROOT}/mixed_and_multi/12_mixed_text_pages_and_scanned_pages.pdf', 'wb')); os.remove(tmp)
write_gt('12_mixed_text_pages_and_scanned_pages', txns, acct, meta, dict(scanned_pages_1_indexed=sorted(scan_pages), total_pages=n)); print('12 mixed', len(txns), 'rows', n, 'pages')

# ---- two statements / two accounts in a single PDF
r = random.Random(302); accA = make_account(r, 'SNB'); accB = make_account(r, 'TUB'); accB['holder'] = accA['holder']; accB['acc_type'] = 'Current Account'
evA = build_events(r, d(2026, 4, 1), d(2026, 6, 30), 'A', rate=18); tA = apply_balances(evA, 30000); mA = make_meta(tA, d(2026, 4, 1), d(2026, 6, 30), 30000)
evB = build_events(r, d(2026, 4, 1), d(2026, 6, 30), 'B', rate=14, salary=False, spend_scale=0.4); tB = apply_balances(evB, 120000); mB = make_meta(tB, d(2026, 4, 1), d(2026, 6, 30), 120000)
build_pdf(f'{ROOT}/mixed_and_multi/13_two_accounts_in_one_pdf.pdf', [(CFG['L1'], accA, tA, mA), (CFG['L3'], accB, tB, mB)])
for t in tA: t['flag'] = 'account_1'
for t in tB: t['flag'] = 'account_2'
write_gt('13_two_accounts_in_one_pdf', tA + tB, accA, mA, dict(second_account=accB, second_account_opening=str(mB['opening']), second_account_closing=str(mB['closing']), rows_account_1=len(tA), rows_account_2=len(tB))); print('13 multi', len(tA), len(tB))

# ---- large file (100+ pages)
big_cfg = dict(CFG['L1'], fs=6.5, lh=8, pad=1.5); CFG['BIG'] = big_cfg
r = random.Random(303); acct = make_account(r, 'SNB'); ev = build_events(r, d(2021, 7, 1), d(2026, 6, 30), 'B', rate=118, spend_scale=0.15)
txns = apply_balances(ev, 25000); meta = make_meta(txns, d(2021, 7, 1), d(2026, 6, 30), 25000)
n = build_pdf(f'{ROOT}/mixed_and_multi/14_large_statement_100plus_pages.pdf', [(big_cfg, acct, txns, meta)]); write_gt('14_large_statement_100plus_pages', txns, acct, meta, dict(total_pages=n)); print('14 large', len(txns), 'rows', n, 'pages')

# ---- edge cases: balance breaks, duplicates, reversal, zero amount, out-of-period, swapped columns
r = random.Random(401); acct = make_account(r, 'SNB'); start, end = d(2026, 4, 1), d(2026, 6, 30)
ev = build_events(r, start, end, 'A', rate=20)
ev.append(dict(date=d(2026, 5, 9), cat='Bank Charges/Fees', narr='CHARGES WAIVED - ZERO AMOUNT ENTRY', ref='', debit=True, amt=q(0), flag='zero_amount_row'))
s, rf = narr('A', 'upi_m', 'FLIPKART', 'flipkart@axisbank', True, r)
ev.append(dict(date=d(2026, 5, 20), cat='Shopping/E-commerce', narr=s + ' FAILED', ref=rf, debit=True, amt=q(2499), flag='reversal_debit'))
ev.append(dict(date=d(2026, 5, 22), cat='Refunds/Reversals', narr=f'REV-UPI-{rf}-FLIPKART-FAILED TXN', ref=rf, debit=False, amt=q(2499), flag='reversal_credit'))
ev.sort(key=lambda e: e['date']); txns = apply_balances(ev, 40000)
true_closing = txns[-1]['balance']; true_deb = sum(t['debit'] or 0 for t in txns); true_cre = sum(t['credit'] or 0 for t in txns)
def pick(i):
    while txns[i]['flag'] or txns[i].get('print_swap') or txns[i]['debit'] is None and txns[i]['credit'] is None: i += 1
    return i
n0 = len(txns); issues = {}
i_sw = pick(int(n0 * .75)); txns[i_sw]['print_swap'] = True; txns[i_sw]['flag'] = 'debit_credit_columns_swapped_in_print'
i_oop = pick(int(n0 * .60)); txns[i_oop]['date'] = start - dt.timedelta(days=45); txns[i_oop]['flag'] = 'date_out_of_period_and_out_of_order'
i_miss = pick(int(n0 * .45)); miss = txns.pop(i_miss); txns[i_miss]['flag'] = 'balance_gap_row_before_this_is_missing'
issues['missing_row_not_printed'] = dict(date=str(miss['date']), description=miss['narr'], debit=str(miss['debit'] or ''), credit=str(miss['credit'] or ''), balance_after=str(miss['balance']))
i_dup = pick(int(n0 * .20)); dup = dict(txns[i_dup]); dup['flag'] = 'exact_duplicate_printed_twice_same_balance'; dup['dup'] = True; txns.insert(i_dup + 1, dup)
meta = make_meta(txns, start, end, 40000); meta.update(closing=true_closing, total_debits=true_deb, total_credits=true_cre)
n = build_pdf(f'{ROOT}/edge_cases/15_edge_cases_balance_breaks_duplicates_reversals.pdf', [(CFG['L1'], acct, txns, meta)])
issues['note'] = 'Summary totals/closing balance on the PDF are the TRUE values; printed rows do not reconcile because of the injected defects.'
write_gt('15_edge_cases_balance_breaks_duplicates_reversals', txns, acct, meta, issues); print('15 edge', len(txns), 'rows', n, 'pages')

# ---- password protected (user password: test1234)
rd = PdfReader(p1); wr = PdfWriter()
for pg in rd.pages: wr.add_page(pg)
wr.encrypt('test1234'); wr.write(open(f'{ROOT}/edge_cases/16_password_protected_password_is_test1234.pdf', 'wb'))
shutil.copy(f'{GT}/01_text_bordered_debit_credit_carry_forward.csv', f'{GT}/16_password_protected_password_is_test1234.csv')
shutil.copy(f'{GT}/01_text_bordered_debit_credit_carry_forward.json', f'{GT}/16_password_protected_password_is_test1234.json')

# ---- corrupted (truncated) pdf, a PNG renamed to .pdf, and a blank page
b = open(p2, 'rb').read(); open(f'{ROOT}/edge_cases/17_corrupted_truncated_file.pdf', 'wb').write(b[:int(len(b) * 0.55)])
Image.new('RGB', (600, 400), (230, 240, 250)).save(f'{ROOT}/edge_cases/_t.png'); os.rename(f'{ROOT}/edge_cases/_t.png', f'{ROOT}/edge_cases/18_not_really_a_pdf_png_renamed.pdf')
c = canvas.Canvas(f'{ROOT}/edge_cases/19_blank_empty_page.pdf', pagesize=A4); c.showPage(); c.save()

# ---- other document types
r = random.Random(501); acct = make_account(r, 'TUB'); acct['acc_no'] = '4012 XXXX XXXX 9087'
ev = build_events(r, d(2026, 7, 1), d(2026, 9, 30), 'B', rate=16, salary=False)
cc = []
for e in ev:
    if e['cat'] in ('Food & Dining', 'Groceries', 'Transport/Fuel/Travel', 'Shopping/E-commerce', 'Healthcare/Pharmacy', 'Entertainment/Subscriptions', 'Education', 'Mobile/Internet/DTH') and e['debit']:
        e['narr'] = f"{e['name']} {r.choice(CITIES)}"; cc.append(e)
    elif e['cat'] == 'Refunds/Reversals': e['narr'] = e['name']; cc.append(e)
for mth in (7, 8, 9):
    cc.append(dict(date=d(2026, mth, 10), cat='Credit Card Payment', narr='PAYMENT RECEIVED - THANK YOU', ref='', debit=False, amt=q(r.randint(8000, 15000)), flag=''))
    cc.append(dict(date=d(2026, mth, 28), cat='Bank Charges/Fees', narr='FINANCE CHARGES + GST', ref='', debit=True, amt=q(r.randint(150, 600)), flag=''))
cc.sort(key=lambda e: e['date'])
ccx = [dict(date=e['date'], narr=e['narr'], ref='', debit=e['amt'] if e['debit'] else None, credit=None if e['debit'] else e['amt'], balance=None, cat=e['cat'], flag='') for e in cc]
due = sum(t['debit'] or 0 for t in ccx) - sum(t['credit'] or 0 for t in ccx)
meta = {'from': d(2026, 7, 1), 'to': d(2026, 9, 30), 'opening': 0, 'closing': due, 'total_due': due, 'min_due': q(due * D('0.05')), 'due_date': d(2026, 10, 18)}
n = build_pdf(f'{ROOT}/other_document_types/20_credit_card_statement_no_balance_column.pdf', [(CFG['CC'], acct, ccx, meta)])
write_gt('20_credit_card_statement_no_balance_column', ccx, acct, meta, dict(note='Not a bank account statement: no running balance column. Tests doc-type handling / graceful degradation.')); print('20 cc', len(ccx), n)

c = canvas.Canvas(f'{ROOT}/other_document_types/21_non_statement_invoice_should_be_rejected.pdf', pagesize=A4); W, H = A4
c.setFont('Helvetica-Bold', 18); c.drawString(50, H - 60, 'TAX INVOICE'); c.setFont('Helvetica', 9)
c.drawString(50, H - 80, 'Invoice No: INV-2026-00418    Date: 12/08/2026    (Fictional)'); c.drawString(50, H - 94, 'Bill To: Sample Customer Pvt Ltd, Ahmedabad')
y = H - 130
for hdr, x in [('Item', 50), ('Qty', 300), ('Rate', 360), ('Amount', 450)]: c.setFont('Helvetica-Bold', 9); c.drawString(x, y, hdr)
for it in [('Cloud hosting - monthly', 1, 4500), ('Support hours', 6, 1200), ('Domain renewal', 2, 899)]:
    y -= 18; c.setFont('Helvetica', 9); c.drawString(50, y, it[0]); c.drawString(300, y, str(it[1])); c.drawString(360, y, inr(it[2])); c.drawString(450, y, inr(it[1] * it[2]))
c.drawString(360, y - 30, 'Total: ' + inr(4500 + 7200 + 1798)); c.showPage(); c.save()

# --------------------------------------------------------------------------- classification data
def cls_rows(seeds, noisy):
    rows = []
    for sd in seeds:
        r = random.Random(sd)
        for st in 'ABCD':
            ev = build_events(r, d(2025, 1, 1), d(2026, 6, 30), st, rate=14)
            for e in ev:
                s = e['narr']
                if noisy:
                    if r.random() < 0.35: s = s.lower()
                    if r.random() < 0.25: s = s[:r.randint(28, 45)]
                    if r.random() < 0.12: s = s.replace('O', '0').replace('I', '1').replace('S', '5')
                rows.append((s, 'DR' if e['debit'] else 'CR', str(e['amt']), e['cat']))
    r.shuffle(rows); return rows
for nm, seeds, noisy, k in [('classification_train_labeled.csv', [1, 2], False, 3500), ('classification_holdout_test_noisy.csv', [90], True, 700)]:
    rows = cls_rows(seeds, noisy)[:k]
    with open(f'{ROOT}/classification_data/{nm}', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['description', 'direction', 'amount', 'category']); w.writerows(rows)
    print(nm, len(rows))
shutil.rmtree('/tmp/x', ignore_errors=True)
print('DONE')
