"""
📂 АУДИТЫН ӨГӨГДӨЛ БЭЛТГЭХ & ШИНЖИЛГЭЭ v1.0
Ямар ч файл оруулаад → автомат таних → хөрвүүлэх → шинжилгээ хийх
pip install streamlit pandas numpy scikit-learn plotly openpyxl
streamlit run data_prep.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import warnings, io, re, gzip
from datetime import datetime
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Аудитын ХОУ", page_icon="🔍", layout="wide")

# ═══════════════════════════════════════════════════════════
# ТОГТМОЛ & ТУСЛАХ ФУНКЦҮҮД
# ═══════════════════════════════════════════════════════════
EDT_COLUMNS = ['report_year','account_code','account_name','transaction_no','transaction_date',
               'journal_no','document_no','counterparty_name','counterparty_id',
               'transaction_description','debit_mnt','credit_mnt','balance_mnt','month']
ACCT_RE_B = re.compile(r'Данс:\s*\[([^\]]+)\]\s*(.*)')
ACCT_RE_P = re.compile(r'Данс:\s*(\d{3}-\d{2}-\d{2}-\d{3})\s+(.*)')

def safe_float(v):
    if v is None or v == '': return 0.0
    try: return float(v)
    except: return 0.0

def parse_account(text):
    m = ACCT_RE_B.match(text)
    if m: return m.group(1).strip(), m.group(2).strip()
    m = ACCT_RE_P.match(text)
    if m: return m.group(1).strip(), m.group(2).strip()
    return None, None

def get_year(name):
    for y in range(2020, 2030):
        if str(y) in name: return y
    return 2025

# ═══════════════════════════════════════════════════════════
# ФАЙЛ ТАНИХ (3 давхар: нэр → sheet → агуулга)
# ═══════════════════════════════════════════════════════════
def detect_file_type(f):
    name = f.name.lower()
    year = get_year(f.name)
    if name.endswith('.csv') or name.endswith('.gz'): return 'ledger', year
    if not name.endswith('.xlsx'): return 'unknown', year

    nc = f.name.lower().replace('_',' ').replace('-',' ')
    for kw in ['ерөнхий журнал','ерөнхий дэвтэр','едт','edt','general ledger','general journal','еренхий журнал']:
        if kw in nc: return 'edt', year
    for kw in ['гүйлгээ баланс','гүйлгээ_баланс','гуйлгээ баланс','trial balance']:
        if kw in nc: return 'raw_tb', year
    if 'tb_standardized' in nc or 'tb standardized' in nc: return 'tb_std', year
    if 'part1' in nc or 'part 1' in nc: return 'part1', year
    if 'ledger' in nc: return 'ledger', year

    import openpyxl
    try:
        raw = f.read(); f.seek(0)
        wb = openpyxl.load_workbook(io.BytesIO(raw), read_only=True)
        sheets = wb.sheetnames
        if '02_ACCOUNT_SUMMARY' in sheets:
            has_rm = '04_RISK_MATRIX' in sheets
            wb.close()
            return ('part1' if has_rm else 'tb_std'), year
        if '04_RISK_MATRIX' in sheets: wb.close(); return 'part1', year
        ws = wb[sheets[0]]
        sample = []
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            sample.append(row)
            if i >= 300: break
        wb.close()
        for row in sample:
            for cell in row[:6]:
                if cell and str(cell).strip().startswith(('Данс:','Компани:','ЕРӨНХИЙ','Журнал:')):
                    return 'edt', year
        for row in sample:
            if len(row)>=2 and row[1] and re.match(r'\d{3}-\d{2}-\d{2}-\d{3}', str(row[1]).strip()):
                return 'raw_tb', year
        for row in sample:
            if row[0]:
                try:
                    int(float(row[0]))
                    if len(row)>=8 and row[1] and re.match(r'\d{3}-', str(row[1])): return 'raw_tb', year
                except: pass
        return 'unknown', year
    except: f.seek(0); return 'unknown', year

FILE_LABELS = {
    'raw_tb':  ('📗 ГҮЙЛГЭЭ_БАЛАНС', '→ TB_standardized руу хөрвүүлнэ'),
    'edt':     ('📘 ЕДТ / Ерөнхий журнал', '→ Ledger + Part1 руу хөрвүүлнэ'),
    'tb_std':  ('✅ TB_standardized', 'Шинжилгээнд бэлэн'),
    'ledger':  ('✅ Ledger', 'Шинжилгээнд бэлэн'),
    'part1':   ('✅ Part1', 'Шинжилгээнд бэлэн'),
    'unknown': ('❓ Тодорхойгүй', 'Гараар сонгоно уу'),
}

# ═══════════════════════════════════════════════════════════
# ХӨРВҮҮЛЭХ ФУНКЦҮҮД
# ═══════════════════════════════════════════════════════════
def process_raw_tb(file_obj):
    import openpyxl
    wb = openpyxl.load_workbook(file_obj, read_only=True)
    ws = wb[wb.sheetnames[0]]
    rows = []
    for row in ws.iter_rows(values_only=True):
        if row[0] is None: continue
        try: int(float(row[0]))
        except: continue
        code = str(row[1]).strip() if row[1] else ''
        if not code or not re.match(r'\d{3}-', code): continue
        rows.append({'account_code': code, 'account_name': str(row[2]).strip() if row[2] else '',
            'opening_debit': safe_float(row[3]), 'opening_credit': safe_float(row[4]),
            'turnover_debit': safe_float(row[5]), 'turnover_credit': safe_float(row[6]),
            'closing_debit': safe_float(row[7]), 'closing_credit': safe_float(row[8])})
    wb.close()
    if not rows: return None, pd.DataFrame()
    df = pd.DataFrame(rows)
    df['opening_balance_signed'] = df['opening_debit'] - df['opening_credit']
    df['turnover_net_signed'] = df['turnover_debit'] - df['turnover_credit']
    df['closing_balance_signed'] = df['closing_debit'] - df['closing_credit']
    df['net_change_signed'] = df['closing_balance_signed'] - df['opening_balance_signed']
    tb_sum = df[['account_code','account_name','opening_debit','opening_credit','opening_balance_signed',
                  'turnover_debit','turnover_credit','turnover_net_signed',
                  'closing_debit','closing_credit','closing_balance_signed','net_change_signed']].copy()
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as w:
        df[['account_code','account_name','opening_debit','opening_credit','turnover_debit','turnover_credit','closing_debit','closing_credit']].to_excel(w, sheet_name='01_TB_CLEAN', index=False)
        tb_sum.to_excel(w, sheet_name='02_ACCOUNT_SUMMARY', index=False)
    buf.seek(0)
    return buf, tb_sum

def process_edt(file_obj, report_year):
    import openpyxl
    wb = openpyxl.load_workbook(file_obj, read_only=True)
    ws = wb[wb.sheetnames[0]]
    rows_out, cur_code, cur_name = [], None, None
    for row in ws.iter_rows(values_only=True):
        c0 = row[0]
        if c0 is None: continue
        s = str(c0).strip()
        if s.startswith('Данс:'):
            code, name = parse_account(s)
            if code: cur_code, cur_name = code, name
            continue
        if any(s.startswith(x) for x in ['Компани:','ЕРӨНХИЙ','Тайлант','Үүсгэсэн','Журнал:','№','Эцсийн','Дт -','Нийт','Эхний','Нээгээд']) or s in ('Валютаар','Төгрөгөөр',''): continue
        try: tx_no = int(float(c0))
        except: continue
        if cur_code is None: continue
        td = row[1] if len(row)>1 else ''
        tx_date = td.strftime('%Y-%m-%d') if isinstance(td, datetime) else (str(td).strip() if td else '')
        rows_out.append({'report_year':str(report_year),'account_code':cur_code,'account_name':cur_name,
            'transaction_no':str(tx_no),'transaction_date':tx_date,
            'journal_no':str(row[5]).strip() if len(row)>5 and row[5] else '',
            'document_no':str(row[6]).strip() if len(row)>6 and row[6] else '',
            'counterparty_name':str(row[3]).strip() if len(row)>3 and row[3] else '',
            'counterparty_id':str(row[4]).strip() if len(row)>4 and row[4] else '',
            'transaction_description':str(row[7]).strip() if len(row)>7 and row[7] else '',
            'debit_mnt':safe_float(row[9]) if len(row)>9 else 0.0,
            'credit_mnt':safe_float(row[11]) if len(row)>11 else 0.0,
            'balance_mnt':safe_float(row[13]) if len(row)>13 else 0.0,
            'month':tx_date[:7] if len(tx_date)>=7 else ''})
    wb.close()
    if not rows_out: return pd.DataFrame(columns=EDT_COLUMNS), 0
    return pd.DataFrame(rows_out), len(rows_out)

def generate_part1(df_led, year):
    df = df_led.copy(); yr = str(year)
    for c in ['debit_mnt','credit_mnt','balance_mnt']:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    monthly = df.groupby(['month','account_code']).agg(total_debit_mnt=('debit_mnt','sum'),total_credit_mnt=('credit_mnt','sum'),ending_balance_mnt=('balance_mnt','last'),transaction_count=('debit_mnt','count')).reset_index()
    monthly.insert(0,'report_year',yr)
    anames = df.groupby('account_code')['account_name'].first()
    acct = df.groupby('account_code').agg(total_debit_mnt=('debit_mnt','sum'),total_credit_mnt=('credit_mnt','sum'),closing_balance_mnt=('balance_mnt','last')).reset_index()
    acct['account_name'] = acct['account_code'].map(anames); acct.insert(0,'report_year',yr)
    rm = df.groupby(['month','account_code','counterparty_name']).agg(transaction_count=('debit_mnt','count'),total_debit=('debit_mnt','sum'),total_credit=('credit_mnt','sum')).reset_index()
    rm['total_amount_mnt'] = rm['total_debit'].abs()+rm['total_credit'].abs(); rm.insert(0,'report_year',yr)
    if len(rm)>0:
        rm['risk_score'] = ((rm['total_amount_mnt']>rm['total_amount_mnt'].quantile(0.75)).astype(int) + (rm['transaction_count']>rm['transaction_count'].quantile(0.75)).astype(int))
    else: rm['risk_score'] = 0
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as w:
        monthly.to_excel(w, sheet_name='02_MONTHLY_SUMMARY', index=False)
        acct.to_excel(w, sheet_name='03_ACCOUNT_SUMMARY', index=False)
        rm.to_excel(w, sheet_name='04_RISK_MATRIX', index=False)
    buf.seek(0)
    return buf, monthly, rm

def read_ledger(f):
    raw = f.read(); f.seek(0)
    if raw[:2]==b'\x1f\x8b': return pd.read_csv(io.StringIO(gzip.decompress(raw).decode('utf-8')), dtype={'account_code':str})
    return pd.read_csv(io.BytesIO(raw), dtype={'account_code':str})

def load_tb(files):
    frames, stats = [], {}
    for f in files:
        year = get_year(f.name)
        f.seek(0); df = pd.read_excel(f, sheet_name='02_ACCOUNT_SUMMARY'); df['year'] = year
        for c in ['turnover_debit','turnover_credit','closing_debit','closing_credit','opening_debit','opening_credit','net_change_signed']:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        stats[year] = {'accounts':len(df),'turnover_d':df['turnover_debit'].sum(),'turnover_c':df['turnover_credit'].sum()}
        frames.append(df)
    return pd.concat(frames, ignore_index=True), stats

def load_ledger_stats(files):
    stats = {}
    for f in files:
        year = get_year(f.name); f.seek(0); df = read_ledger(f)
        for c in ['debit_mnt','credit_mnt']: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        stats[year] = {'rows':len(df),'accounts':df['account_code'].nunique()}; del df
    return stats

def load_part1(files):
    all_rm, all_mo = [], []
    for f in files:
        year = get_year(f.name); f.seek(0)
        try: rm = pd.read_excel(f, sheet_name='04_RISK_MATRIX'); rm['year']=year; all_rm.append(rm)
        except: pass
        f.seek(0)
        try: mo = pd.read_excel(f, sheet_name='02_MONTHLY_SUMMARY'); mo['year']=year; all_mo.append(mo)
        except: pass
    return (pd.concat(all_rm,ignore_index=True) if all_rm else pd.DataFrame()), (pd.concat(all_mo,ignore_index=True) if all_mo else pd.DataFrame())

def run_ml(tb_all, cont, n_est):
    df = tb_all.copy()
    le = LabelEncoder(); df['cat_num'] = le.fit_transform(df['account_code'].astype(str).str[:3])
    df['log_turn_d'] = np.log1p(df['turnover_debit'].abs()); df['log_turn_c'] = np.log1p(df['turnover_credit'].abs())
    df['log_close_d'] = np.log1p(df['closing_debit'].abs()); df['log_close_c'] = np.log1p(df['closing_credit'].abs())
    df['turn_ratio'] = (df['turnover_debit']/df['turnover_credit'].replace(0,np.nan)).fillna(0).replace([np.inf,-np.inf],0)
    df['log_abs_change'] = np.log1p(df['net_change_signed'].abs()) if 'net_change_signed' in df.columns else np.log1p((df['closing_debit']-df['opening_debit']).abs())
    feats = ['cat_num','log_turn_d','log_turn_c','log_close_d','log_close_c','turn_ratio','log_abs_change','year']
    X = df[feats].fillna(0).replace([np.inf,-np.inf],0)
    df['iso_anomaly'] = (IsolationForest(contamination=cont,random_state=42,n_estimators=200).fit_predict(X)==-1).astype(int)
    df['zscore_anomaly'] = (np.abs(StandardScaler().fit_transform(X)).max(axis=1)>2.0).astype(int)
    p95 = df['turn_ratio'].quantile(0.95)
    df['turn_anomaly'] = ((df['turn_ratio']>p95)|(df['turn_ratio']<-p95)).astype(int)
    df['ensemble_anomaly'] = ((df['iso_anomaly']==1)|((df['zscore_anomaly']==1)&(df['turn_anomaly']==1))).astype(int)
    y = df['ensemble_anomaly'].values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = {'Random Forest': RandomForestClassifier(n_estimators=n_est,max_depth=10,random_state=42,class_weight='balanced'),
              'Gradient Boosting': GradientBoostingClassifier(n_estimators=150,max_depth=5,learning_rate=0.1,random_state=42),
              'Logistic Regression': LogisticRegression(max_iter=1000,random_state=42,class_weight='balanced')}
    res = {}
    for nm, mdl in models.items():
        yp = cross_val_predict(mdl,X,y,cv=cv,method='predict')
        ypr = cross_val_predict(mdl,X,y,cv=cv,method='predict_proba')[:,1]
        res[nm] = {'pred':yp,'prob':ypr,'precision':precision_score(y,yp),'recall':recall_score(y,yp),'f1':f1_score(y,yp),'auc':roc_auc_score(y,ypr)}
    best = max(res, key=lambda k: res[k]['f1'])
    rf = RandomForestClassifier(n_estimators=n_est,max_depth=10,random_state=42,class_weight='balanced'); rf.fit(X,y)
    fi = pd.DataFrame({'feature':feats,'importance':rf.feature_importances_}).sort_values('importance',ascending=False)
    nt=len(df); ns=int(nt*0.20); at=df['turnover_debit'].abs()+df['turnover_credit'].abs()
    wt=(at/at.sum()).fillna(1/nt); np.random.seed(42); ms=np.zeros(nt,dtype=int)
    ms[np.random.choice(nt,size=ns,replace=False,p=wt.values)]=1; ym=(ms&y).astype(int)
    return df, X, y, feats, res, best, fi, ym

# ═══════════════════════════════════════════════════════════
# ИНТЕРФЕЙС
# ═══════════════════════════════════════════════════════════
st.markdown('<h1 style="text-align:center;color:#1565c0">🔍 Аудитын ХОУ — Нэгдсэн хэрэгсэл</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="background:#E3F2FD; padding:15px; border-radius:10px; border-left:5px solid #1565C0; margin-bottom:20px;">
    <b>📂 Ямар ч файлаа оруулаарай!</b> Систем автоматаар таниж, хөрвүүлж, шинжилгээг ажиллуулна.<br>
    <span style="color:#555; font-size:13px;">
    ГҮЙЛГЭЭ_БАЛАНС, Ерөнхий журнал, ЕДТ, TB_standardized, Ledger CSV, Part1 — бүгдийг нэг дор.
    </span>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("📎 Бүх файлуудаа энд оруулна уу", type=['xlsx','csv','gz'], accept_multiple_files=True, key='all_files')

tb_files, led_files, p1_files = [], [], []
convert_log = []

if uploaded:
    detected = []
    for f in uploaded:
        ftype, year = detect_file_type(f); f.seek(0)
        detected.append({'file':f, 'type':ftype, 'year':year, 'name':f.name})

    # ── Таних хүснэгт ──
    st.markdown("### 🔍 Файл таних үр дүн")
    rows_d = []
    for d in detected:
        lb, desc = FILE_LABELS.get(d['type'], FILE_LABELS['unknown'])
        rows_d.append({'Файл':d['name'], 'Төрөл':lb, 'Он':d['year'], 'Үйлдэл':desc})
    st.dataframe(pd.DataFrame(rows_d), use_container_width=True, hide_index=True)

    # ── Гараар засах боломж ──
    unknowns = [d for d in detected if d['type']=='unknown']
    if unknowns:
        st.markdown("#### ❓ Тодорхойгүй файлуудын төрлийг сонгоно уу")
        for d in unknowns:
            choice = st.selectbox(f"**{d['name']}** — Төрөл сонгох:", ['unknown','raw_tb','edt'], format_func=lambda x: {'unknown':'❓ Алгасах','raw_tb':'📗 ГҮЙЛГЭЭ_БАЛАНС','edt':'📘 ЕДТ/Ерөнхий журнал'}[x], key=f"manual_{d['name']}")
            d['type'] = choice

    # ── Автомат хөрвүүлэлт + чиглүүлэлт ──
    for d in detected:
        d['file'].seek(0)
        if d['type'] == 'tb_std':
            tb_files.append(d['file'])
        elif d['type'] == 'ledger':
            led_files.append(d['file'])
        elif d['type'] == 'part1':
            p1_files.append(d['file'])
        elif d['type'] == 'raw_tb':
            with st.spinner(f"📗 {d['name']} → TB хөрвүүлж байна..."):
                buf, tb_s = process_raw_tb(d['file'])
            if buf is None:
                st.warning(f"⚠️ {d['name']} — дансны мэдээлэл олдсонгүй")
            else:
                buf.seek(0); tw = io.BytesIO(buf.getvalue()); tw.name = f"TB_standardized_{d['year']}.xlsx"
                tb_files.append(tw)
                convert_log.append(f"✅ {d['name']} → TB ({len(tb_s):,} данс)")
                st.download_button(f"📥 TB_{d['year']}.xlsx", buf.getvalue(), f"TB_standardized_{d['year']}1231.xlsx", key=f"dl_tb_{d['year']}")
        elif d['type'] == 'edt':
            with st.spinner(f"📘 {d['name']} → Ledger + Part1 хөрвүүлж байна..."):
                df_e, cnt = process_edt(d['file'], d['year'])
            if cnt == 0:
                st.warning(f"⚠️ {d['name']} — гүйлгээ уншигдсангүй. Файлын формат тохирохгүй байж магадгүй.")
            else:
                df_e['debit_mnt'] = pd.to_numeric(df_e['debit_mnt'],errors='coerce').fillna(0)
                df_e['credit_mnt'] = pd.to_numeric(df_e['credit_mnt'],errors='coerce').fillna(0)
                csv_b = df_e[EDT_COLUMNS].to_csv(index=False).encode('utf-8')
                lw = io.BytesIO(csv_b); lw.name = f"ledger_{d['year']}.csv"; led_files.append(lw)
                p1_buf, p1_mo, p1_rm = generate_part1(df_e, d['year'])
                p1_buf.seek(0); pw = io.BytesIO(p1_buf.getvalue()); pw.name = f"part1_{d['year']}.xlsx"; p1_files.append(pw)
                convert_log.append(f"✅ {d['name']} → Ledger ({cnt:,} гүйлгээ) + Part1 ({len(p1_rm):,} хос)")
                c1,c2 = st.columns(2)
                c1.download_button(f"📥 Ledger_{d['year']}.csv.gz", gzip.compress(csv_b), f"prototype_ledger_{d['year']}.csv.gz", key=f"dl_led_{d['year']}")
                c2.download_button(f"📥 Part1_{d['year']}.xlsx", p1_buf.getvalue(), f"prototype_part1_{d['year']}.xlsx", key=f"dl_p1_{d['year']}")

    if convert_log:
        for msg in convert_log: st.success(msg)

    # ── Бэлэн байдлын тойм ──
    st.markdown("---")
    c1,c2,c3 = st.columns(3)
    c1.metric("TB файл", f"{len(tb_files)}", delta="Бэлэн" if tb_files else "Дутуу", delta_color="normal" if tb_files else "inverse")
    c2.metric("Ledger файл", f"{len(led_files)}", delta="Бэлэн" if led_files else "Дутуу", delta_color="normal" if led_files else "inverse")
    c3.metric("Part1 файл", f"{len(p1_files)}", delta="Нэмэлт" if p1_files else "Алгасах боломжтой")

# ═══════════════════════════════════════════════════════════
# ШИНЖИЛГЭЭ
# ═══════════════════════════════════════════════════════════
if tb_files and led_files:
    st.markdown("---")
    st.markdown("### ⚙️ Шинжилгээний тохиргоо")
    c1s, c2s = st.columns(2)
    with c1s: cont = st.slider("Тусгаарлалтын ой — хэвийн бус хувь", 0.05, 0.20, 0.10, 0.01)
    with c2s: nest = st.slider("Санамсаргүй ой — модны тоо", 50, 500, 200, 50)

    if st.button("🚀 Шинжилгээ ажиллуулах", type="primary", use_container_width=True):
        with st.spinner("TB уншиж байна..."): tb_all, tb_st = load_tb(tb_files)
        with st.spinner("Ledger уншиж байна..."): led_st = load_ledger_stats(led_files)
        rm_all, mo_all = pd.DataFrame(), pd.DataFrame()
        if p1_files:
            with st.spinner("Part1 уншиж байна..."): rm_all, mo_all = load_part1(p1_files)
        with st.spinner("🤖 ХОУ шинжилгээ..."): df, X, y, feats, res, best, fi, ym = run_ml(tb_all, cont, nest)
        st.session_state.update({'done':True,'df':df,'X':X,'y':y,'res':res,'best':best,'fi':fi,'ym':ym,'tb_st':tb_st,'led_st':led_st,'rm_all':rm_all,'mo_all':mo_all})

if st.session_state.get('done'):
    df=st.session_state['df']; y=st.session_state['y']; res=st.session_state['res']
    best=st.session_state['best']; fi=st.session_state['fi']; ym=st.session_state['ym']
    tb_st=st.session_state['tb_st']; led_st=st.session_state['led_st']
    rm_all=st.session_state['rm_all']; mo_all=st.session_state['mo_all']
    bp=res[best]['pred']; yrs=sorted(tb_st.keys()); has_rm=len(rm_all)>0; has_mo=len(mo_all)>0

    st.success(f"✅ {len(df):,} данс, {sum(d['rows'] for d in led_st.values()):,} гүйлгээ шинжлэгдсэн")
    tabs = ["📊 Нэгтгэл","🔍 Хэвийн бус данс","⚖️ ХОУ ↔ Уламжлалт","🧠 Тайлбарлагдах ХОУ","📋 Жагсаалт"]
    if has_rm: tabs.append("🎯 Эрсдэлийн матриц")
    if has_mo: tabs.append("📈 Сарын хандлага")
    all_tabs = st.tabs(tabs)

    with all_tabs[0]:
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Данс",f"{len(df):,}"); m2.metric("Гүйлгээ",f"{sum(d['rows'] for d in led_st.values()):,}")
        m3.metric("Хэвийн бус",f"{df['ensemble_anomaly'].sum():,} ({df['ensemble_anomaly'].mean()*100:.1f}%)")
        m4.metric("Шилдэг загвар",f"{best} F1={res[best]['f1']:.4f}")
        if has_rm:
            mr1,mr2=st.columns(2); mr1.metric("Эрсдэлийн хос",f"{len(rm_all):,}"); mr2.metric("Эрсдэлтэй",f"{len(rm_all[rm_all['risk_score']>0]):,}")
        fg = make_subplots(rows=1,cols=3,subplot_titles=("Данс","Эргэлт (T₮)","ЕДТ мөр"))
        cl3=['#2196F3','#4CAF50','#FF9800']
        for i,yv in enumerate(yrs):
            fg.add_trace(go.Bar(x=[str(yv)],y=[tb_st[yv]['accounts']],marker_color=cl3[i%3],showlegend=False),row=1,col=1)
            fg.add_trace(go.Bar(x=[str(yv)],y=[tb_st[yv]['turnover_d']/1e9],marker_color=cl3[i%3],showlegend=False),row=1,col=2)
            if yv in led_st: fg.add_trace(go.Bar(x=[str(yv)],y=[led_st[yv]['rows']],marker_color=cl3[i%3],showlegend=False),row=1,col=3)
        fg.update_layout(height=350); st.plotly_chart(fg, use_container_width=True)

    with all_tabs[1]:
        mt={'Тусгаарлалтын ой':'iso_anomaly','Стандарт хазайлт':'zscore_anomaly','Эргэлтийн харьцаа':'turn_anomaly','Нэгдсэн дүн':'ensemble_anomaly'}
        ad=[]
        for m,c in mt.items():
            row_d={'Арга':m,'Нийт':int(df[c].sum())}
            for yv in yrs: mask=df['year']==yv; cnt=df.loc[mask,c].sum(); row_d[str(yv)]=f"{int(cnt)} ({cnt/mask.sum()*100:.1f}%)"
            ad.append(row_d)
        st.dataframe(pd.DataFrame(ad),use_container_width=True,hide_index=True)
        st.plotly_chart(px.scatter(df,x='log_turn_d',y='log_abs_change',color=df['ensemble_anomaly'].map({0:'Хэвийн',1:'Хэвийн бус'}),facet_col='year',opacity=0.5,color_discrete_map={'Хэвийн':'#90caf9','Хэвийн бус':'#c62828'},height=400),use_container_width=True)

    with all_tabs[2]:
        st.dataframe(pd.DataFrame([{'Загвар':n,'Нарийвчлал':f"{r['precision']:.4f}",'Бүрэн илрүүлэлт':f"{r['recall']:.4f}",'F1':f"{r['f1']:.4f}",'AUC':f"{r['auc']:.4f}"} for n,r in res.items()]),use_container_width=True,hide_index=True)
        fg2=go.Figure()
        for n,r in res.items():
            fpr,tpr,_=roc_curve(y,r['prob']); fg2.add_trace(go.Scatter(x=fpr,y=tpr,name=f"{n} (AUC={r['auc']:.4f})"))
        fg2.add_trace(go.Scatter(x=[0,1],y=[0,1],name='Санамсаргүй',line=dict(dash='dash',color='gray')))
        fg2.update_layout(title='ROC муруй',height=400); st.plotly_chart(fg2,use_container_width=True)
        st.subheader("Илрүүлэлтийн эрсдэл")
        dr=[]
        for yv in yrs:
            mk=(df['year']==yv).values; yt=y[mk]; nt2=yt.sum()
            a2=(1-(bp[mk]&yt).sum()/nt2) if nt2>0 else 0; m2x=(1-(ym[mk]&yt).sum()/nt2) if nt2>0 else 0
            dr.append({'Жил':yv,'ХОУ':f"{a2:.4f}",'MUS 20%':f"{m2x:.4f}",'Сайжрал':f"{m2x-a2:.4f}"})
        st.dataframe(pd.DataFrame(dr),use_container_width=True,hide_index=True)

    with all_tabs[3]:
        st.plotly_chart(px.bar(fi,x='importance',y='feature',orientation='h',color='importance',color_continuous_scale='Blues',title='Шинж чанарын ач холбогдол').update_layout(height=400,yaxis={'categoryorder':'total ascending'}),use_container_width=True)
        fd={'log_abs_change':'📈 Он дамнасан цэвэр өөрчлөлт — ISA 520','turn_ratio':'⚖️ Дебит/кредит эргэлтийн харьцаа — ISA 240','log_turn_d':'📊 Баримт дебит гүйлгээний хэмжээ — ISA 320','log_turn_c':'📊 Баримт кредит гүйлгээний хэмжээ — ISA 320','log_close_d':'📋 Жилийн эцсийн дебит үлдэгдэл — ISA 505','log_close_c':'📋 Жилийн эцсийн кредит үлдэгдэл — ISA 505','cat_num':'🏷️ Дансны ангиллын код — ISA 315','year':'📅 Тайлант жил'}
        for _,r in fi.iterrows(): st.markdown(f"**{r['feature']}** ({r['importance']:.4f}): {fd.get(r['feature'],'')}")

    with all_tabs[4]:
        adf=df[df['ensemble_anomaly']==1][['year','account_code','account_name','turnover_debit','turnover_credit','turn_ratio','log_abs_change']].copy()
        yf=st.selectbox("Жил",['Бүгд']+[str(y2) for y2 in yrs])
        if yf!='Бүгд': adf=adf[adf['year']==int(yf)]
        st.write(f"Нийт: {len(adf)}"); st.dataframe(adf,use_container_width=True,hide_index=True,height=500)
        st.download_button("📥 CSV татах",adf.to_csv(index=False).encode('utf-8-sig'),"anomaly.csv","text/csv")

    if has_rm:
        with all_tabs[5]:
            rm_all['risk_score']=pd.to_numeric(rm_all['risk_score'],errors='coerce').fillna(0)
            rs=[]
            for yv in sorted(rm_all['year'].unique()):
                rmy=rm_all[rm_all['year']==yv]; rs.append({'Жил':yv,'Нийт':f"{len(rmy):,}",'Эрсдэлтэй':f"{len(rmy[rmy['risk_score']>0]):,}"})
            st.dataframe(pd.DataFrame(rs),use_container_width=True,hide_index=True)
            st.subheader("Топ 20 харилцагч")
            top_cp=rm_all.groupby('counterparty_name').agg(txn=('transaction_count','sum'),accounts=('account_code','nunique')).sort_values('txn',ascending=False).head(20).reset_index()
            top_cp.columns=['Харилцагч','Гүйлгээний тоо','Дансны тоо']; st.dataframe(top_cp,use_container_width=True,hide_index=True)
    if has_mo:
        tidx=6 if has_rm else 5
        with all_tabs[tidx]:
            mo_all['total_debit_mnt']=pd.to_numeric(mo_all['total_debit_mnt'],errors='coerce').fillna(0)
            mo_all['transaction_count']=pd.to_numeric(mo_all['transaction_count'],errors='coerce').fillna(0)
            mo_agg=mo_all.groupby('month').agg(debit=('total_debit_mnt','sum'),txn=('transaction_count','sum')).reset_index()
            fig_mo=make_subplots(rows=2,cols=1,subplot_titles=("Эргэлт (тэрбум₮)","Гүйлгээний тоо"))
            fig_mo.add_trace(go.Scatter(x=mo_agg['month'],y=mo_agg['debit']/1e9,name='Дебит'),row=1,col=1)
            fig_mo.add_trace(go.Bar(x=mo_agg['month'],y=mo_agg['txn'],name='Гүйлгээ'),row=2,col=1)
            fig_mo.update_layout(height=500); st.plotly_chart(fig_mo,use_container_width=True)

elif uploaded and not tb_files and not led_files:
    st.info("👆 ГҮЙЛГЭЭ_БАЛАНС / TB + ЕДТ / Ledger файлуудаа оруулна уу")
