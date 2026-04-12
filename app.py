import os, time, warnings, math, uuid
from itertools import combinations
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import streamlit as st

st.set_page_config(
    page_title="Market Basket Analysis",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-base:    #070b14;
    --bg-card:    #0d1525;
    --bg-raised:  #111c30;
    --border:     #1e2d47;
    --gold:       #f5c518;
    --cyan:       #38bdf8;
    --red:        #f87171;
    --green:      #34d399;
    --purple:     #a78bfa;
    --text-main:  #e2e8f0;
    --text-muted: #64748b;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-base) !important;
    color: var(--text-main) !important;
}

[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * { color: var(--text-main) !important; }

.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: border-color 0.2s;
}
.kpi-card:hover { border-color: var(--gold); }
.kpi-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.kpi-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    line-height: 1;
}
.kpi-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-muted);
    margin-top: 4px;
}

.page-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, #f5c518, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
    line-height: 1.1;
}
.page-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-bottom: 32px;
    letter-spacing: 0.05em;
}

.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--gold);
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin: 28px 0 18px 0;
    letter-spacing: 0.02em;
}

.algo-badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 6px;
    letter-spacing: 0.08em;
}
.badge-apriori { background: rgba(56,189,248,0.15); color: var(--cyan); border: 1px solid rgba(56,189,248,0.3); }
.badge-pcy     { background: rgba(167,139,250,0.15); color: var(--purple); border: 1px solid rgba(167,139,250,0.3); }

.stat-row {
    display: flex;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    padding: 5px 0;
    border-bottom: 1px solid var(--border);
}
.stat-key  { color: var(--text-muted); }
.stat-val  { color: var(--gold); font-weight: 600; }

.insight-box {
    background: rgba(245,197,24,0.07);
    border: 1px solid rgba(245,197,24,0.25);
    border-left: 3px solid var(--gold);
    border-radius: 8px;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 0.88rem;
    color: var(--text-main);
    font-family: 'JetBrains Mono', monospace;
}

[data-testid="stTab"] {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

.stButton > button {
    background: linear-gradient(135deg, #f5c518, #f59e0b) !important;
    color: #070b14 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 28px !important;
    letter-spacing: 0.04em !important;
    font-size: 0.95rem !important;
}

.stSlider label, .stNumberInput label, .stSelectbox label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    color: var(--text-muted) !important;
    letter-spacing: 0.08em !important;
}

hr { border-color: var(--border) !important; }
.stSpinner { color: var(--gold) !important; }
.stAlert { border-radius: 8px !important; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


PLOTLY_LAYOUT = dict(
    paper_bgcolor="#070b14",
    plot_bgcolor="#0d1525",
    font=dict(family="JetBrains Mono, monospace", color="#e2e8f0", size=11),
    margin=dict(l=20, r=20, t=40, b=20),
    colorway=["#f5c518","#38bdf8","#f87171","#34d399","#a78bfa",
               "#fb923c","#22d3ee","#f472b6","#86efac","#fbbf24"],
    xaxis=dict(gridcolor="#1e2d47", linecolor="#1e2d47", zeroline=False),
    yaxis=dict(gridcolor="#1e2d47", linecolor="#1e2d47", zeroline=False),
    legend=dict(bgcolor="#111c30", bordercolor="#1e2d47", borderwidth=1),
    hoverlabel=dict(bgcolor="#111c30", bordercolor="#1e2d47",
                    font=dict(family="JetBrains Mono", color="#e2e8f0")),
)

def apply_theme(fig, title="", height=420):
    fig.update_layout(**PLOTLY_LAYOUT, title=dict(text=title, font=dict(
        family="Syne, sans-serif", size=14, color="#f5c518")), height=height)
    return fig


@st.cache_data(show_spinner=False)
def generate_synthetic(n_transactions=6000, seed=42):
    rng = np.random.default_rng(seed)
    categories = {
        "dairy"    : ["whole milk","butter","cream cheese","yogurt","whipped cream",
                       "sour cream","condensed milk","curd"],
        "produce"  : ["other vegetables","root vegetables","tropical fruit","citrus fruit",
                       "pip fruit","onions","berries","herbs"],
        "bakery"   : ["rolls/buns","brown bread","pastry","waffles","cake bar","cookies"],
        "beverages": ["soda","bottled water","fruit juice","bottled beer",
                       "canned beer","coffee","tea","misc beverages"],
        "meat"     : ["sausage","pork","beef","chicken","frankfurter","ham"],
        "snacks"   : ["chocolate","nuts","candy bars","chips","popcorn","specialty bar"],
        "household": ["dish cleaner","detergent","fabric softener","toilet cleaner",
                       "kitchen towels","napkins"],
        "pantry"   : ["pasta","rice","salt","sugar","oil","margarine","jam","ketchup"],
    }
    all_items = [i for cat in categories.values() for i in cat]
    item_to_cat = {i: c for c, items in categories.items() for i in items}
    import datetime
    base = datetime.date(2023, 1, 1)
    rows = []
    for tid in range(n_transactions):
        sz = int(rng.choice([2,3,4,5,6,7,8], p=[0.05,0.15,0.25,0.25,0.15,0.10,0.05]))
        base_cat = rng.choice(list(categories.keys()))
        base_items = categories[base_cat]
        n_base = min(sz, rng.integers(1, len(base_items)+1))
        basket = set(rng.choice(base_items, size=n_base, replace=False).tolist())
        rest = [i for i in all_items if i not in basket]
        extra = max(0, sz - len(basket))
        if extra and rest:
            basket.update(rng.choice(rest, size=min(extra, len(rest)), replace=False))
        day = int(rng.integers(0, 365))
        date = (base + datetime.timedelta(days=day)).isoformat()
        for item in basket:
            rows.append({"transaction_id": str(tid), "item": item,
                         "date": date, "category": item_to_cat[item]})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_data(path_groceries, path_retail, n_synth):
    import datetime
    if path_groceries and os.path.exists(path_groceries):
        df = pd.read_csv(path_groceries)
        df.columns = ["transaction_id","date","item"]
        df["transaction_id"] = df["transaction_id"].astype(str)
        df["item"] = df["item"].str.strip().str.lower()
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)
        df["category"] = "unknown"
        return df, "Groceries (Kaggle)"
    if path_retail and os.path.exists(path_retail):
        df = pd.read_excel(path_retail, engine="openpyxl")
        df = df[["InvoiceNo","InvoiceDate","Description"]].dropna()
        df.columns = ["transaction_id","date","item"]
        df = df[~df["transaction_id"].astype(str).str.startswith("C")]
        df["item"] = df["item"].str.strip().str.lower()
        df["transaction_id"] = df["transaction_id"].astype(str)
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)
        df["category"] = "unknown"
        return df, "Online Retail (UCI)"
    df = generate_synthetic(n_transactions=n_synth)
    return df, "Synthetic Grocery"


def build_baskets(df):
    return (df.groupby("transaction_id")["item"]
              .apply(frozenset).reset_index()
              .rename(columns={"item":"basket"}))


def _apriori_core(basket_data, min_support, max_k):
    baskets = [frozenset(b) for b in basket_data]
    n = len(baskets)
    fi, stats, timing = {}, [], {}

    t0 = time.time()
    counts = {}
    for b in baskets:
        for item in b:
            counts[item] = counts.get(item, 0) + 1
    L1 = {frozenset([i]): c for i, c in counts.items() if c >= min_support}
    timing[1] = time.time() - t0
    fi[1] = L1
    n_single = len(counts)
    stats.append({"k":1,"candidates":n_single,"frequent":len(L1),
                  "pruned":n_single-len(L1),"time_s":timing[1]})

    for k in range(2, max_k+1):
        all_items = frozenset().union(*fi[k-1].keys())
        cands = set()
        for iset in fi[k-1]:
            for item in all_items:
                if item not in iset:
                    c = iset | frozenset([item])
                    if len(c) == k:
                        cands.add(c)
        valid_cands = [c for c in cands
                       if all(frozenset(s) in fi[k-1] for s in combinations(c, k-1))]
        if not valid_cands:
            break
        t0 = time.time()
        Lk = {}
        for b in baskets:
            for c in valid_cands:
                if c.issubset(b):
                    Lk[c] = Lk.get(c, 0) + 1
        Lk = {c: cnt for c, cnt in Lk.items() if cnt >= min_support}
        timing[k] = time.time() - t0
        stats.append({"k":k,"candidates":len(valid_cands),"frequent":len(Lk),
                      "pruned":len(valid_cands)-len(Lk),"time_s":timing[k]})
        if not Lk:
            break
        fi[k] = Lk

    return fi, pd.DataFrame(stats), timing


@st.cache_data(show_spinner=False)
def run_apriori(basket_data, min_support, max_k):
    return _apriori_core(basket_data, min_support, max_k)


def _pcy_core(basket_data, min_support, n_buckets, max_k):
    return _run_pcy_impl(basket_data, min_support, n_buckets, max_k)


@st.cache_data(show_spinner=False)
def run_pcy(basket_data, min_support, n_buckets, max_k):
    return _run_pcy_impl(basket_data, min_support, n_buckets, max_k)


def _run_pcy_impl(basket_data, min_support, n_buckets, max_k):
    baskets = [frozenset(b) for b in basket_data]
    n = len(baskets)
    fi, stats, timing = {}, [], {}

    t0 = time.time()
    item_counts = {}
    bucket_arr = np.zeros(n_buckets, dtype=np.int32)
    for b in baskets:
        for item in b:
            item_counts[item] = item_counts.get(item, 0) + 1
        items_list = sorted(b)
        for a, bv in combinations(items_list, 2):
            h = (hash(a) * 2654435761 + hash(bv) * 40503) % n_buckets
            bucket_arr[h] += 1
    L1 = {frozenset([i]): c for i, c in item_counts.items() if c >= min_support}
    bitmap = bucket_arr >= min_support
    freq_buckets = int(bitmap.sum())
    timing[1] = time.time() - t0
    fi[1] = L1
    stats.append({"k":1,"candidates":len(item_counts),"frequent":len(L1),
                  "pruned":len(item_counts)-len(L1),"time_s":timing[1]})

    freq_items = sorted(frozenset().union(*L1.keys()))
    all_pairs = list(combinations(freq_items, 2))
    pair_cands = [frozenset([a, b]) for a, b in all_pairs
                  if bitmap[(hash(a)*2654435761 + hash(b)*40503) % n_buckets]]
    pair_savings = 100 * (1 - len(pair_cands) / max(len(all_pairs), 1))

    t0 = time.time()
    L2 = {}
    for b in baskets:
        for c in pair_cands:
            if c.issubset(b):
                L2[c] = L2.get(c, 0) + 1
    L2 = {c: cnt for c, cnt in L2.items() if cnt >= min_support}
    timing[2] = time.time() - t0
    fi[2] = L2
    stats.append({"k":2,"candidates":len(pair_cands),"frequent":len(L2),
                  "pruned":len(pair_cands)-len(L2),"time_s":timing[2],
                  "pcy_filtered":len(all_pairs)-len(pair_cands)})

    for k in range(3, max_k+1):
        all_items = frozenset().union(*fi[k-1].keys())
        cands = set()
        for iset in fi[k-1]:
            for item in all_items:
                if item not in iset:
                    c = iset | frozenset([item])
                    if len(c) == k:
                        cands.add(c)
        valid_cands = [c for c in cands
                       if all(frozenset(s) in fi[k-1] for s in combinations(c, k-1))]
        if not valid_cands:
            break
        t0 = time.time()
        Lk = {}
        for b in baskets:
            for c in valid_cands:
                if c.issubset(b):
                    Lk[c] = Lk.get(c, 0) + 1
        Lk = {c: cnt for c, cnt in Lk.items() if cnt >= min_support}
        timing[k] = time.time() - t0
        stats.append({"k":k,"candidates":len(valid_cands),"frequent":len(Lk),
                      "pruned":len(valid_cands)-len(Lk),"time_s":timing[k]})
        if not Lk:
            break
        fi[k] = Lk

    return fi, pd.DataFrame(stats), timing, freq_buckets, pair_savings


def mine_rules_nocache(fi_data, n_tx, min_conf, min_lift):
    return _mine_rules_impl(fi_data, n_tx, min_conf, min_lift)


@st.cache_data(show_spinner=False)
def mine_rules(fi_data, n_tx, min_conf, min_lift):
    return _mine_rules_impl(fi_data, n_tx, min_conf, min_lift)


def _mine_rules_impl(fi_data, n_tx, min_conf, min_lift):
    fi = {k: {frozenset(iset): cnt for iset, cnt in d.items()}
          for k, d in fi_data.items()}
    sup_lookup = {}
    for d in fi.values():
        for iset, cnt in d.items():
            sup_lookup[iset] = cnt / n_tx

    rows = []
    for k, d in fi.items():
        if k < 2:
            continue
        for iset, cnt in d.items():
            iset_sup = cnt / n_tx
            for ant_size in range(1, k):
                for ant_tuple in combinations(iset, ant_size):
                    ant = frozenset(ant_tuple)
                    cons = iset - ant
                    ant_sup = sup_lookup.get(ant, 0)
                    cons_sup = sup_lookup.get(cons, 0)
                    if ant_sup == 0:
                        continue
                    conf = iset_sup / ant_sup
                    if conf < min_conf:
                        continue
                    lift = conf / cons_sup if cons_sup > 0 else 0
                    if lift < min_lift:
                        continue
                    conviction = (1 - cons_sup) / (1 - conf) if conf < 1 else 999.0
                    leverage = iset_sup - ant_sup * cons_sup
                    conf_rev = iset_sup / cons_sup if cons_sup > 0 else 0
                    kulc_val = 0.5 * (conf + conf_rev)
                    rows.append({
                        "antecedent" : ", ".join(sorted(ant)),
                        "consequent" : ", ".join(sorted(cons)),
                        "ant_size"   : ant_size,
                        "cons_size"  : len(cons),
                        "support"    : round(iset_sup, 5),
                        "confidence" : round(conf, 5),
                        "lift"       : round(lift, 4),
                        "conviction" : round(min(conviction, 99.0), 4),
                        "leverage"   : round(leverage, 6),
                        "kulczynski" : round(kulc_val, 4),
                    })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("lift", ascending=False).reset_index(drop=True)


def serialize_fi(fi):
    return {k: {tuple(sorted(iset)): cnt for iset, cnt in d.items()}
            for k, d in fi.items()}


def fi_to_df(fi, n_tx):
    rows = []
    for k, d in fi.items():
        for iset, cnt in d.items():
            rows.append({"size":k, "itemset":frozenset(iset),
                         "support_count":cnt, "support":cnt/n_tx})
    return pd.DataFrame(rows).sort_values("support_count", ascending=False).reset_index(drop=True)


def find_closed_maximal(fi, n_tx):
    flat_fi = {}
    for d in fi.values():
        flat_fi.update(d)
    closed, maximal = [], []
    for iset, cnt in flat_fi.items():
        k = len(iset)
        is_max = True
        if k+1 in fi:
            for sup in fi[k+1]:
                if iset.issubset(sup):
                    is_max = False
                    break
        if is_max:
            maximal.append({"itemset":iset, "support_count":cnt,
                            "support":cnt/n_tx, "size":k})
        is_closed = True
        if k+1 in fi:
            for sup, scnt in fi[k+1].items():
                if iset.issubset(sup) and scnt == cnt:
                    is_closed = False
                    break
        if is_closed:
            closed.append({"itemset":iset, "support_count":cnt,
                           "support":cnt/n_tx, "size":k})
    return pd.DataFrame(closed), pd.DataFrame(maximal)


with st.sidebar:
    st.markdown("""
    <div style='font-family:Syne,sans-serif;font-size:1.35rem;font-weight:800;
                color:#f5c518;margin-bottom:4px;letter-spacing:-0.01em'>
        MBA Dashboard
    </div>
    <div style='font-family:JetBrains Mono,monospace;font-size:0.7rem;
                color:#64748b;margin-bottom:20px'>CSL7110 · Big Data ML</div>
    """, unsafe_allow_html=True)

    st.markdown("**DATASET**")
    has_groceries = os.path.exists("data/Groceries_dataset.csv")
    has_retail = os.path.exists("data/Online Retail.xlsx")

    if has_groceries:
        st.markdown(
            "<div style='font-family:JetBrains Mono,monospace;font-size:0.78rem;"
            "background:#0d2b1a;border:1px solid #1a4731;border-radius:6px;"
            "padding:10px 12px;margin-bottom:8px'>"
            "<span style='color:#34d399'>CSV detected</span><br>"
            "<span style='color:#64748b'>data/Groceries_dataset.csv</span>"
            "</div>",
            unsafe_allow_html=True
        )
        n_synth = 6000
    elif has_retail:
        st.markdown(
            "<div style='font-family:JetBrains Mono,monospace;font-size:0.78rem;"
            "background:#0d2b1a;border:1px solid #1a4731;border-radius:6px;"
            "padding:10px 12px;margin-bottom:8px'>"
            "<span style='color:#34d399'>XLSX detected</span><br>"
            "<span style='color:#64748b'>data/Online Retail.xlsx</span>"
            "</div>",
            unsafe_allow_html=True
        )
        n_synth = 6000
    else:
        st.markdown(
            "<div style='font-family:JetBrains Mono,monospace;font-size:0.72rem;"
            "color:#f59e0b;margin-bottom:4px'>"
            "No CSV found - using synthetic data"
            "</div>",
            unsafe_allow_html=True
        )
        n_synth = st.slider("Synthetic transactions", 1000, 10000, 6000, 500)

    st.markdown("---")
    st.markdown("**ALGORITHM PARAMETERS**")
    min_sup_pct = st.slider("Min support (%)", 0.5, 15.0, 2.5, 0.5)
    max_k = st.slider("Max itemset size (k)", 2, 5, 4)
    min_conf = st.slider("Min confidence", 0.05, 0.9, 0.20, 0.05)
    min_lift = st.slider("Min lift", 1.0, 5.0, 1.0, 0.1)
    n_buckets = st.selectbox("PCY buckets", [10007, 50021, 100003, 200003], index=2)

    st.markdown("---")
    run_btn = st.button("Run Analysis", use_container_width=True)
    st.markdown("---")

    st.markdown("""
    <div style='font-family:JetBrains Mono,monospace;font-size:0.68rem;
                color:#64748b;line-height:1.7'>
    NAVIGATION<br>
    1. Overview &amp; EDA<br>
    2. A-Priori Results<br>
    3. PCY Algorithm<br>
    4. Association Rules<br>
    5. Network Graph<br>
    6. Co-occurrence<br>
    7. Scalability
    </div>
    """, unsafe_allow_html=True)


with st.spinner("Loading dataset..."):
    raw_df, dataset_name = load_data(
        "data/Groceries_dataset.csv", "data/Online Retail.xlsx", n_synth
    )

raw_df = raw_df.dropna(subset=["item"])
raw_df = raw_df[raw_df["item"].str.strip() != ""]

n_tx = raw_df["transaction_id"].nunique()
min_sup = max(2, int((min_sup_pct / 100) * n_tx))

basket_groups = build_baskets(raw_df)
basket_list = [list(b) for b in basket_groups["basket"]]

if "ap_fi" not in st.session_state or run_btn:
    prog = st.progress(0, "Running A-Priori...")
    ap_fi, ap_stats, ap_timing = run_apriori(
        [tuple(sorted(b)) for b in basket_list], min_sup, max_k
    )
    prog.progress(50, "Running PCY...")
    pcy_fi, pcy_stats, pcy_timing, pcy_buckets_above, pcy_savings = run_pcy(
        [tuple(sorted(b)) for b in basket_list], min_sup, n_buckets, max_k
    )
    prog.progress(80, "Mining rules...")
    ap_fi_ser = serialize_fi(ap_fi)
    rules_df = mine_rules(ap_fi_ser, n_tx, min_conf, min_lift)
    prog.progress(100, "Done.")
    prog.empty()

    st.session_state.update({
        "ap_fi": ap_fi, "ap_stats": ap_stats, "ap_timing": ap_timing,
        "pcy_fi": pcy_fi, "pcy_stats": pcy_stats, "pcy_timing": pcy_timing,
        "pcy_buckets_above": pcy_buckets_above, "pcy_savings": pcy_savings,
        "rules_df": rules_df, "raw_df": raw_df, "n_tx": n_tx,
        "min_sup": min_sup, "dataset_name": dataset_name,
    })

ap_fi = st.session_state["ap_fi"]
ap_stats = st.session_state["ap_stats"]
pcy_fi = st.session_state["pcy_fi"]
pcy_stats = st.session_state["pcy_stats"]
rules_df = st.session_state["rules_df"]
raw_df = st.session_state["raw_df"]
n_tx = st.session_state["n_tx"]
dataset_name = st.session_state["dataset_name"]

itemsets_df = fi_to_df(ap_fi, n_tx)
closed_df, maximal_df = find_closed_maximal(ap_fi, n_tx)
total_fi = sum(len(v) for v in ap_fi.values())
n_items = raw_df["item"].nunique()
avg_basket = raw_df.groupby("transaction_id")["item"].count().mean()
item_freq = (raw_df.groupby("item")["item"].count()
             .rename("count").reset_index()
             .sort_values("count", ascending=False))
item_freq["support"] = item_freq["count"] / n_tx

with st.sidebar:
    st.markdown("---")
    stat_pairs = [
        ("dataset",        dataset_name[:18]),
        ("transactions",   f"{n_tx:,}"),
        ("unique items",   f"{n_items:,}"),
        ("avg basket",     f"{avg_basket:.1f}"),
        ("min support",    f"{min_sup_pct:.1f}% ({min_sup})"),
        ("freq. itemsets", f"{total_fi:,}"),
        ("rules",          f"{len(rules_df):,}"),
        ("pcy savings",    f"{st.session_state.get('pcy_savings', 0):.1f}%"),
    ]
    rows_html = "".join(
        "<div class='stat-row'>"
        "<span class='stat-key'>" + k + "</span>"
        "<span class='stat-val'>" + v + "</span>"
        "</div>"
        for k, v in stat_pairs
    )
    stats_block = (
        "<div style='font-family:JetBrains Mono,monospace;font-size:0.73rem'>"
        + rows_html
        + "</div>"
    )
    st.markdown(stats_block, unsafe_allow_html=True)


st.markdown("""
<div class='page-title'>Market Basket Analysis</div>
<div class='page-sub'>SCALABLE FREQUENT ITEMSET MINING · A-PRIORI &amp; PCY · APACHE SPARK · CSL7110</div>
""", unsafe_allow_html=True)


tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "1. Overview",
    "2. A-Priori",
    "3. PCY",
    "4. Assoc. Rules",
    "5. Network",
    "6. Co-occurrence",
    "7. Scalability",
])


with tab1:
    kpis = [
        ("Transactions",     f"{n_tx:,}",         "#f5c518", f"dataset: {dataset_name}"),
        ("Unique Items",     f"{n_items:,}",        "#38bdf8", "distinct products"),
        ("Avg Basket Size",  f"{avg_basket:.2f}",   "#34d399", "items per transaction"),
        ("Frequent Itemsets",f"{total_fi:,}",       "#a78bfa", f"min sup = {min_sup_pct:.1f}%"),
        ("Rules Generated",  f"{len(rules_df):,}",  "#f87171", f"conf >= {min_conf:.2f}"),
        ("PCY Savings",      f"{st.session_state.get('pcy_savings',0):.1f}%",
                                                    "#fb923c", "vs A-Priori pass 2"),
    ]
    cols = st.columns(6)
    for col, (label, val, color, sub) in zip(cols, kpis):
        col.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>{label}</div>
            <div class='kpi-value' style='color:{color}'>{val}</div>
            <div class='kpi-sub'>{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Dataset Exploration</div>",
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        bs = raw_df.groupby("transaction_id")["item"].count().reset_index()
        bs.columns = ["tid", "size"]
        fig = px.histogram(bs, x="size", nbins=20,
                           color_discrete_sequence=["#38bdf8"],
                           labels={"size":"Items per basket","count":"# Transactions"})
        fig.add_vline(x=bs["size"].mean(), line_dash="dash", line_color="#f5c518",
                      annotation_text=f"mean={bs['size'].mean():.1f}",
                      annotation_font_color="#f5c518")
        fig.add_vline(x=bs["size"].median(), line_dash="dot", line_color="#34d399",
                      annotation_text=f"median={bs['size'].median():.0f}",
                      annotation_font_color="#34d399")
        apply_theme(fig, "Basket Size Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        top20 = item_freq.head(20).copy()
        fig = px.bar(top20.iloc[::-1], x="support", y="item", orientation="h",
                     color="support", color_continuous_scale="Plasma",
                     labels={"support":"Support","item":""})
        fig.update_coloraxes(showscale=False)
        apply_theme(fig, "Top-20 Items by Support")
        fig.update_layout(yaxis=dict(tickfont=dict(size=9)))
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        if "category" in raw_df.columns and raw_df["category"].nunique() > 1:
            cat_counts = (raw_df.groupby("category")["item"].count()
                          .reset_index().sort_values("item", ascending=False))
            fig = px.pie(cat_counts, names="category", values="item",
                         color_discrete_sequence=["#f5c518","#38bdf8","#f87171",
                                                  "#34d399","#a78bfa","#fb923c",
                                                  "#22d3ee","#f472b6"],
                         hole=0.45)
            fig.update_traces(textfont_size=10)
            apply_theme(fig, "Item Category Breakdown", height=380)
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.bar(item_freq.head(10), x="item", y="count",
                         color="count", color_continuous_scale="Viridis")
            apply_theme(fig, "Top-10 Item Counts", height=380)
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        ranks = np.arange(1, len(item_freq)+1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ranks, y=item_freq["support"].values,
                                 mode="markers", name="Items",
                                 marker=dict(color="#f5c518", size=4, opacity=0.6)))
        log_r = np.log(ranks)
        log_s = np.log(item_freq["support"].values + 1e-9)
        slope, intercept = np.polyfit(log_r, log_s, 1)
        fit_y = np.exp(intercept) * ranks**slope
        fig.add_trace(go.Scatter(x=ranks, y=fit_y, mode="lines",
                                 name=f"Power-law a={-slope:.2f}",
                                 line=dict(color="#f87171", width=2)))
        fig.update_xaxes(type="log", title="Item rank (log)")
        fig.update_yaxes(type="log", title="Support (log)")
        apply_theme(fig, "Item Frequency - Power Law Distribution", height=380)
        st.plotly_chart(fig, use_container_width=True)

    if "date" in raw_df.columns:
        try:
            time_df = (raw_df.assign(month=pd.to_datetime(raw_df["date"], errors="coerce")
                                     .dt.to_period("M"))
                       .drop_duplicates("transaction_id")
                       .groupby("month").size().reset_index(name="count"))
            time_df["month"] = time_df["month"].astype(str)
            if len(time_df) > 1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=time_df["month"], y=time_df["count"],
                    mode="lines+markers", fill="tozeroy",
                    line=dict(color="#a78bfa", width=2),
                    fillcolor="rgba(167,139,250,0.12)",
                    marker=dict(color="#a78bfa", size=6)
                ))
                apply_theme(fig, "Monthly Transaction Volume", height=300)
                st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass


with tab2:
    st.markdown("""
    <span class='algo-badge badge-apriori'>A-PRIORI ALGORITHM</span>
    """, unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Pass-by-Pass Statistics</div>",
                unsafe_allow_html=True)

    disp_stats = ap_stats.copy()
    disp_stats["prune_pct"] = (100 * disp_stats["pruned"] /
                                disp_stats["candidates"].clip(lower=1)).round(1)
    disp_stats["time_s"] = disp_stats["time_s"].round(3)
    st.dataframe(
        disp_stats[["k","candidates","frequent","pruned","prune_pct","time_s"]]
        .rename(columns={"k":"Pass k","candidates":"Candidates",
                         "frequent":"Frequent","pruned":"Pruned",
                         "prune_pct":"Prune %","time_s":"Time (s)"}),
        use_container_width=True, hide_index=True
    )

    st.markdown(f"""
    <div class='insight-box'>
    A-Priori ran <b>{len(ap_stats)}</b> passes and found
    <b>{total_fi:,}</b> frequent itemsets total.
    The monotonicity prune is most aggressive at k=2 - once single items are
    filtered, the candidate pair space shrinks dramatically.
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[f"k={k}" for k in ap_stats["k"]],
                             y=ap_stats["candidates"], name="Candidates",
                             marker_color="#64748b"))
        fig.add_trace(go.Bar(x=[f"k={k}" for k in ap_stats["k"]],
                             y=ap_stats["frequent"], name="Frequent",
                             marker_color="#38bdf8"))
        fig.update_layout(barmode="overlay")
        apply_theme(fig, "Candidates Generated vs Frequent Retained")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        prune_pct = 100 * ap_stats["pruned"] / ap_stats["candidates"].clip(lower=1)
        fig = go.Figure(go.Bar(
            x=[f"k={k}" for k in ap_stats["k"]],
            y=prune_pct.round(1),
            text=prune_pct.round(1).astype(str) + "%",
            textposition="outside",
            marker=dict(
                color=prune_pct,
                colorscale="RdYlGn",
                showscale=False
            )
        ))
        apply_theme(fig, "Pruning Efficiency per Pass (%)")
        fig.update_yaxes(range=[0, 110])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-header'>Frequent Itemsets Explorer</div>",
                unsafe_allow_html=True)

    sizes = sorted(itemsets_df["size"].unique())
    sel_k = st.select_slider("Itemset size k", options=sizes, value=sizes[0])
    sub = itemsets_df[itemsets_df["size"] == sel_k].head(30).copy()
    sub["label"] = sub["itemset"].apply(lambda x: " + ".join(sorted(x)))

    col3, col4 = st.columns([2, 1])
    with col3:
        fig = px.bar(sub.iloc[::-1], x="support", y="label", orientation="h",
                     color="support", color_continuous_scale="Plasma",
                     labels={"support":"Support","label":""})
        fig.update_coloraxes(showscale=False)
        apply_theme(fig, f"Top Frequent Itemsets (k={sel_k})", height=500)
        fig.update_layout(yaxis=dict(tickfont=dict(size=8)))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown("<div class='section-header'>Itemset Categories</div>",
                    unsafe_allow_html=True)
        cats = pd.DataFrame({
            "Category": ["All Frequent","Closed","Maximal"],
            "Count": [total_fi, len(closed_df), len(maximal_df)],
            "Pct": [100.0,
                    100*len(closed_df)/max(total_fi, 1),
                    100*len(maximal_df)/max(total_fi, 1)]
        })
        fig = px.funnel(cats, x="Count", y="Category",
                        color="Category",
                        color_discrete_sequence=["#38bdf8","#34d399","#f5c518"])
        apply_theme(fig, "Frequent -> Closed -> Maximal", height=300)
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        for k_v in sizes:
            sub2 = itemsets_df[itemsets_df["size"] == k_v]
            fig2.add_trace(go.Violin(y=sub2["support"], name=f"k={k_v}",
                                     box_visible=True, meanline_visible=True,
                                     fillcolor="rgba(245,197,24,0.15)",
                                     line_color="#f5c518"))
        apply_theme(fig2, "Support Distribution by k", height=300)
        st.plotly_chart(fig2, use_container_width=True)


with tab3:
    pcy_savings = st.session_state.get("pcy_savings", 0)
    pcy_above = st.session_state.get("pcy_buckets_above", 0)
    pcy_timing = st.session_state.get("pcy_timing", {})

    st.markdown("""
    <span class='algo-badge badge-pcy'>PCY ALGORITHM</span>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class='insight-box'>
    PCY Hash Filter: <b>{pcy_above:,}</b> of <b>{n_buckets:,}</b> buckets exceed the
    min-support threshold ({100*pcy_above/n_buckets:.1f}% bitmap density).
    Candidate pairs reduced by <b>{pcy_savings:.1f}%</b> compared to vanilla A-Priori,
    eliminating pairs that hash into sub-threshold buckets before any counting begins.
    </div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.markdown(f"""<div class='kpi-card'>
        <div class='kpi-label'>Buckets above threshold</div>
        <div class='kpi-value' style='color:#34d399'>{pcy_above:,}</div>
        <div class='kpi-sub'>of {n_buckets:,} total</div>
    </div>""", unsafe_allow_html=True)
    col2.markdown(f"""<div class='kpi-card'>
        <div class='kpi-label'>Bitmap density</div>
        <div class='kpi-value' style='color:#38bdf8'>{100*pcy_above/n_buckets:.1f}%</div>
        <div class='kpi-sub'>bits set to 1</div>
    </div>""", unsafe_allow_html=True)
    col3.markdown(f"""<div class='kpi-card'>
        <div class='kpi-label'>PCY candidate savings</div>
        <div class='kpi-value' style='color:#f5c518'>{pcy_savings:.1f}%</div>
        <div class='kpi-sub'>fewer pairs at pass 2</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>A-Priori vs PCY Side-by-Side</div>",
                unsafe_allow_html=True)

    max_k_avail = max(max(ap_stats["k"]), max(pcy_stats["k"]))
    ks = list(range(1, max_k_avail+1))

    def get_stat(df, k, col, default=0):
        row = df[df["k"] == k]
        return row[col].values[0] if len(row) > 0 else default

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[f"k={k}" for k in ks],
                             y=[get_stat(ap_stats, k, "candidates") for k in ks],
                             name="A-Priori", marker_color="#38bdf8"))
        fig.add_trace(go.Bar(x=[f"k={k}" for k in ks],
                             y=[get_stat(pcy_stats, k, "candidates") for k in ks],
                             name="PCY", marker_color="#a78bfa"))
        fig.update_layout(barmode="group")
        apply_theme(fig, "Candidates Generated per Pass")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        ap_t = [st.session_state["ap_timing"].get(k, 0) for k in ks]
        pcy_t = [pcy_timing.get(k, 0) for k in ks]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[f"k={k}" for k in ks], y=ap_t,
                             name="A-Priori", marker_color="#38bdf8"))
        fig.add_trace(go.Bar(x=[f"k={k}" for k in ks], y=pcy_t,
                             name="PCY", marker_color="#a78bfa"))
        fig.update_layout(barmode="group")
        apply_theme(fig, "Runtime per Pass (seconds)")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = go.Figure(go.Pie(
            labels=["Above threshold (bitmap=1)","Below threshold (bitmap=0)"],
            values=[pcy_above, n_buckets-pcy_above],
            hole=0.55,
            marker=dict(colors=["#34d399","#1e2d47"],
                        line=dict(color="#070b14", width=2)),
            textfont=dict(color="#e2e8f0"),
        ))
        apply_theme(fig, f"PCY Bucket Utilisation ({n_buckets:,} buckets)", height=350)
        fig.update_layout(legend=dict(font=dict(size=9)))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        ap_idf = fi_to_df(ap_fi, n_tx)
        pcy_idf = fi_to_df(pcy_fi, n_tx)
        ap_sizes = ap_idf.groupby("size")["itemset"].count().reset_index()
        pcy_sizes = pcy_idf.groupby("size")["itemset"].count().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[f"k={r['size']}" for _, r in ap_sizes.iterrows()],
                             y=ap_sizes["itemset"], name="A-Priori", marker_color="#38bdf8"))
        fig.add_trace(go.Bar(x=[f"k={r['size']}" for _, r in pcy_sizes.iterrows()],
                             y=pcy_sizes["itemset"], name="PCY", marker_color="#a78bfa"))
        fig.update_layout(barmode="group")
        apply_theme(fig, "Frequent Itemsets per k", height=350)
        st.plotly_chart(fig, use_container_width=True)


with tab4:
    st.markdown("<div class='section-header'>Rule Explorer</div>",
                unsafe_allow_html=True)

    if rules_df.empty:
        st.warning("No rules found. Try lowering min support or min confidence.")
    else:
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            conf_range = st.slider("Confidence range",
                                   float(rules_df["confidence"].min()),
                                   float(rules_df["confidence"].max()),
                                   (float(rules_df["confidence"].min()),
                                    float(rules_df["confidence"].max())))
        with fc2:
            lift_range = st.slider("Lift range",
                                   float(rules_df["lift"].min()),
                                   float(rules_df["lift"].max()),
                                   (float(rules_df["lift"].min()),
                                    float(rules_df["lift"].max())))
        with fc3:
            sort_col = st.selectbox("Sort by", ["lift","confidence","support",
                                                "conviction","leverage","kulczynski"])

        filtered = rules_df[
            rules_df["confidence"].between(*conf_range) &
            rules_df["lift"].between(*lift_range)
        ].sort_values(sort_col, ascending=False)

        st.markdown(f"<div class='insight-box'>Showing <b>{len(filtered):,}</b> "
                    f"rules after filters</div>", unsafe_allow_html=True)

        disp_cols = ["antecedent","consequent","support","confidence",
                     "lift","conviction","leverage","kulczynski"]
        st.dataframe(filtered[disp_cols].head(100),
                     use_container_width=True, hide_index=True)

        st.markdown("<div class='section-header'>Metric Visualizations</div>",
                    unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter_3d(
                filtered.head(300),
                x="support", y="confidence", z="lift",
                color="kulczynski", size="support",
                color_continuous_scale="Plasma",
                opacity=0.8,
                hover_data=["antecedent","consequent"],
                labels={"kulczynski":"Kulczynski"},
            )
            fig.update_traces(marker=dict(sizeref=0.01))
            apply_theme(fig, "3-D Rule Space: Support x Confidence x Lift", height=480)
            fig.update_layout(scene=dict(
                bgcolor="#0d1525",
                xaxis=dict(backgroundcolor="#0d1525", gridcolor="#1e2d47",
                           showbackground=True, title_font_color="#e2e8f0"),
                yaxis=dict(backgroundcolor="#0d1525", gridcolor="#1e2d47",
                           showbackground=True, title_font_color="#e2e8f0"),
                zaxis=dict(backgroundcolor="#0d1525", gridcolor="#1e2d47",
                           showbackground=True, title_font_color="#e2e8f0"),
            ))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.scatter(filtered.head(300),
                             x="support", y="confidence",
                             color="lift", size="leverage",
                             color_continuous_scale="Viridis",
                             hover_data=["antecedent","consequent","lift"],
                             size_max=20,
                             labels={"lift":"Lift","leverage":"Leverage"})
            fig.add_hline(y=min_conf, line_dash="dash", line_color="#f87171",
                          annotation_text=f"min_conf={min_conf}")
            apply_theme(fig, "Support vs Confidence (size=Leverage, colour=Lift)", height=480)
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            fig = px.histogram(filtered, x="lift", nbins=30,
                               color_discrete_sequence=["#a78bfa"])
            fig.add_vline(x=1.0, line_dash="dash", line_color="#f87171",
                          annotation_text="Lift=1")
            fig.add_vline(x=filtered["lift"].mean(), line_dash="dot",
                          line_color="#f5c518",
                          annotation_text=f"mean={filtered['lift'].mean():.2f}")
            apply_theme(fig, "Lift Distribution")
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            metric_corr = filtered[["support","confidence","lift",
                                     "conviction","leverage","kulczynski"]].corr()
            fig = px.imshow(metric_corr.round(3),
                            color_continuous_scale="RdBu_r",
                            zmin=-1, zmax=1,
                            text_auto=".2f",
                            aspect="auto")
            apply_theme(fig, "Metric Correlation Matrix")
            fig.update_coloraxes(colorbar=dict(tickfont=dict(color="#e2e8f0")))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("<div class='section-header'>Parallel Coordinates</div>",
                    unsafe_allow_html=True)
        pc_df = filtered.head(150).copy()
        pc_df["conviction_capped"] = pc_df["conviction"].clip(upper=20)
        fig = px.parallel_coordinates(
            pc_df,
            color="lift",
            dimensions=["support","confidence","lift",
                        "conviction_capped","leverage","kulczynski"],
            color_continuous_scale=px.colors.sequential.Plasma,
            labels={"conviction_capped":"Conviction(<=20)"},
        )
        apply_theme(fig, "Parallel Coordinates - Top 150 Rules", height=380)
        fig.update_layout(coloraxis_colorbar=dict(tickfont=dict(color="#e2e8f0")))
        st.plotly_chart(fig, use_container_width=True)


with tab5:
    st.markdown("<div class='section-header'>Association Rule Network</div>",
                unsafe_allow_html=True)

    if rules_df.empty:
        st.info("No rules available for network visualization.")
    else:
        n_top = st.slider("Top N rules to display", 10, 100, 50, 5)
        sort_metric = st.selectbox("Sort top rules by", ["lift","confidence","support"],
                                   key="net_sort")
        top_r = rules_df.nlargest(n_top, sort_metric)

        G = nx.DiGraph()
        for _, row in top_r.iterrows():
            ants = [a.strip() for a in row["antecedent"].split(",")]
            cons = [c.strip() for c in row["consequent"].split(",")]
            for a in ants:
                for c in cons:
                    if G.has_edge(a, c):
                        G[a][c]["weight"] = max(G[a][c]["weight"], row["confidence"])
                    else:
                        G.add_edge(a, c, weight=row["confidence"],
                                   lift=row["lift"], support=row["support"])

        if len(G.nodes()) > 0:
            pos = nx.spring_layout(G, k=2.5, seed=42, iterations=60)

            edge_x, edge_y, edge_text = [], [], []
            edge_colors, edge_widths = [], []
            for (u, v, d) in G.edges(data=True):
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]
                edge_text.append(f"{u} -> {v}<br>conf={d['weight']:.3f} lift={d['lift']:.3f}")
                edge_colors.append(d["lift"])
                edge_widths.append(0.5 + 3.5 * d["weight"])

            node_x = [pos[n][0] for n in G.nodes()]
            node_y = [pos[n][1] for n in G.nodes()]
            degrees = dict(G.degree())
            node_sz = [8 + 6*degrees[n] for n in G.nodes()]
            in_deg = dict(G.in_degree())
            node_colors = [in_deg[n] for n in G.nodes()]

            fig = go.Figure()

            min_lift_v = min(edge_colors) if edge_colors else 1
            max_lift_v = max(edge_colors) if edge_colors else 2
            for i in range(0, len(edge_x)-2, 3):
                seg_lift = edge_colors[i//3] if i//3 < len(edge_colors) else 1
                norm = (seg_lift - min_lift_v) / max(max_lift_v - min_lift_v, 0.01)
                r = int(245*norm + 56*(1-norm))
                g = int(197*norm + 189*(1-norm))
                b = int(24*norm + 248*(1-norm))
                fig.add_trace(go.Scatter(
                    x=edge_x[i:i+3], y=edge_y[i:i+3],
                    mode="lines",
                    line=dict(width=edge_widths[i//3] if i//3 < len(edge_widths) else 1,
                              color=f"rgba({r},{g},{b},0.65)"),
                    hoverinfo="skip", showlegend=False
                ))

            fig.add_trace(go.Scatter(
                x=node_x, y=node_y, mode="markers+text",
                marker=dict(size=node_sz, color=node_colors,
                            colorscale="Plasma",
                            colorbar=dict(
                                title=dict(text="In-degree",
                                           font=dict(color="#e2e8f0")),
                                tickfont=dict(color="#e2e8f0")),
                            line=dict(color="#0d1525", width=1.5)),
                text=list(G.nodes()),
                textposition="top center",
                textfont=dict(size=8, color="#e2e8f0"),
                hovertext=[f"{n}<br>degree={degrees[n]}<br>in={in_deg[n]}"
                           for n in G.nodes()],
                hoverinfo="text",
                name="Items",
            ))

            apply_theme(fig,
                        f"Association Rule Network - Top {n_top} by {sort_metric.capitalize()}",
                        height=620)
            fig.update_layout(
                xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<div class='section-header'>Top Nodes by Degree</div>",
                            unsafe_allow_html=True)
                deg_df = pd.DataFrame({
                    "item": list(G.nodes()),
                    "degree": [degrees[n] for n in G.nodes()],
                    "in_degree": [in_deg[n] for n in G.nodes()],
                    "out_degree": [G.out_degree(n) for n in G.nodes()],
                }).sort_values("degree", ascending=False).head(15)
                st.dataframe(deg_df, hide_index=True, use_container_width=True)
            with col2:
                fig2 = px.bar(deg_df.head(10).iloc[::-1],
                              x="degree", y="item", orientation="h",
                              color="in_degree", color_continuous_scale="Plasma",
                              labels={"degree":"Total degree","in_degree":"In-degree"})
                apply_theme(fig2, "Top-10 Nodes by Connectivity", height=380)
                st.plotly_chart(fig2, use_container_width=True)


with tab6:
    st.markdown("<div class='section-header'>Item Co-occurrence Heatmap</div>",
                unsafe_allow_html=True)

    n_heatmap = st.slider("Top N items for co-occurrence", 10, 40, 20, 5)
    top_items = item_freq["item"].head(n_heatmap).tolist()
    idx = {item: i for i, item in enumerate(top_items)}
    n_c = len(top_items)
    M = np.zeros((n_c, n_c))

    if 2 in ap_fi:
        for iset, cnt in ap_fi[2].items():
            items_l = list(iset)
            if len(items_l) == 2:
                a, b = items_l
                if a in idx and b in idx:
                    M[idx[a], idx[b]] = cnt
                    M[idx[b], idx[a]] = cnt

    cnt_vec = np.array([item_freq[item_freq["item"] == i]["count"].values[0]
                         if i in item_freq["item"].values else 1
                         for i in top_items], dtype=float)
    J = np.zeros_like(M)
    for i in range(n_c):
        for j in range(n_c):
            union = cnt_vec[i] + cnt_vec[j] - M[i, j]
            J[i, j] = M[i, j] / union if union > 0 else 0
    np.fill_diagonal(M, 0)
    np.fill_diagonal(J, 0)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.imshow(M.astype(int), x=top_items, y=top_items,
                        color_continuous_scale="YlOrBr",
                        text_auto=False, aspect="auto")
        apply_theme(fig, "Raw Co-occurrence Counts", height=520)
        fig.update_xaxes(tickangle=45, tickfont_size=8)
        fig.update_yaxes(tickfont_size=8)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.imshow(J.round(3), x=top_items, y=top_items,
                        color_continuous_scale="Plasma",
                        text_auto=False, aspect="auto",
                        zmin=0, zmax=J.max() or 0.1)
        apply_theme(fig, "Jaccard Similarity (normalised)", height=520)
        fig.update_xaxes(tickangle=45, tickfont_size=8)
        fig.update_yaxes(tickfont_size=8)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-header'>Top Item Pair Co-occurrences</div>",
                unsafe_allow_html=True)
    if 2 in ap_fi:
        pairs = [(f"{sorted(list(iset))[0]}  -  {sorted(list(iset))[1]}", cnt)
                 for iset, cnt in ap_fi[2].items() if len(iset) == 2]
        pairs_df = pd.DataFrame(pairs, columns=["pair","count"])\
                     .sort_values("count", ascending=False).head(25)
        pairs_df["support"] = pairs_df["count"] / n_tx
        fig = px.bar(pairs_df.iloc[::-1], x="support", y="pair", orientation="h",
                     color="support", color_continuous_scale="Viridis",
                     labels={"support":"Support","pair":""})
        fig.update_coloraxes(showscale=False)
        apply_theme(fig, "Top-25 Frequent Item Pairs", height=550)
        fig.update_layout(yaxis=dict(tickfont=dict(size=9)))
        st.plotly_chart(fig, use_container_width=True)

    if "category" in raw_df.columns and raw_df["category"].nunique() > 1:
        st.markdown("<div class='section-header'>Cross-Category Co-occurrence</div>",
                    unsafe_allow_html=True)
        cat_item = raw_df[["transaction_id","category"]].drop_duplicates()
        cats = sorted(raw_df["category"].unique())
        cat_idx = {c: i for i, c in enumerate(cats)}
        CM = np.zeros((len(cats), len(cats)))
        for tid, grp in cat_item.groupby("transaction_id"):
            cat_list = sorted(grp["category"].unique())
            for ca, cb in combinations(cat_list, 2):
                CM[cat_idx[ca], cat_idx[cb]] += 1
                CM[cat_idx[cb], cat_idx[ca]] += 1
        fig = px.imshow(CM.astype(int), x=cats, y=cats,
                        color_continuous_scale="Teal",
                        text_auto=True, aspect="auto")
        apply_theme(fig, "Category x Category Co-occurrence Matrix", height=420)
        st.plotly_chart(fig, use_container_width=True)


with tab7:
    st.markdown("<div class='section-header'>Scalability Experiments</div>",
                unsafe_allow_html=True)

    st.markdown("""
    <div class='insight-box'>
    These experiments sample the dataset at varying fractions and re-run both
    algorithms to measure how runtime, frequent itemset count, and rule count scale
    with data volume. Run time varies per machine; relative trends are what matter.
    </div>""", unsafe_allow_html=True)

    run_scale = st.button("Run Scalability Experiments (takes ~30s)")

    if run_scale or "scale_df" in st.session_state:
        if run_scale:
            fracs = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
            scale_results = []
            prog_scale = st.progress(0, "Running scalability experiments...")
            all_baskets = [tuple(sorted(b)) for b in basket_list]
            for step, frac in enumerate(fracs):
                n_sample = max(50, int(frac * len(all_baskets)))
                rng = np.random.default_rng(seed=int(frac*1000))
                sample = [all_baskets[i] for i in
                           rng.choice(len(all_baskets), n_sample, replace=False)]
                minsup = max(2, int((min_sup_pct/100) * n_sample))

                t0 = time.time()
                ap_res, _, _ = _apriori_core(sample, minsup, 3)
                ap_t = time.time() - t0

                t0 = time.time()
                pcy_res, _, _, _, pcy_sav = _pcy_core(sample, minsup, n_buckets, 3)
                pcy_t = time.time() - t0

                rules_res = mine_rules_nocache(
                    serialize_fi(ap_res), n_sample, min_conf, 1.0
                )
                scale_results.append({
                    "frac": frac, "n_baskets": n_sample,
                    "ap_time": round(ap_t, 3), "pcy_time": round(pcy_t, 3),
                    "ap_fi": sum(len(v) for v in ap_res.values()),
                    "pcy_fi": sum(len(v) for v in pcy_res.values()),
                    "n_rules": len(rules_res),
                    "pcy_savings": round(pcy_sav, 1),
                })
                prog_scale.progress(int(100*(step+1)/len(fracs)))
            prog_scale.empty()
            st.session_state["scale_df"] = pd.DataFrame(scale_results)

        scale_df = st.session_state["scale_df"]

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=scale_df["n_baskets"], y=scale_df["ap_time"],
                                     mode="lines+markers", name="A-Priori",
                                     line=dict(color="#38bdf8", width=2),
                                     marker=dict(size=7)))
            fig.add_trace(go.Scatter(x=scale_df["n_baskets"], y=scale_df["pcy_time"],
                                     mode="lines+markers", name="PCY",
                                     line=dict(color="#a78bfa", width=2),
                                     marker=dict(size=7)))
            apply_theme(fig, "Runtime vs Dataset Size", height=360)
            fig.update_layout(xaxis_title="# Baskets", yaxis_title="Time (s)")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            speedup = scale_df["ap_time"] / scale_df["pcy_time"].clip(lower=0.001)
            fig = go.Figure(go.Bar(
                x=scale_df["n_baskets"], y=speedup.round(2),
                text=speedup.round(2), textposition="outside",
                marker=dict(color=speedup, colorscale="RdYlGn", showscale=False)
            ))
            fig.add_hline(y=1, line_dash="dash", line_color="#f87171",
                          annotation_text="Speedup=1")
            apply_theme(fig, "PCY Speedup over A-Priori", height=360)
            fig.update_layout(xaxis_title="# Baskets",
                              yaxis_title="A-Priori time / PCY time")
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=scale_df["n_baskets"], y=scale_df["ap_fi"],
                                     mode="lines+markers", name="A-Priori FI",
                                     fill="tozeroy", fillcolor="rgba(56,189,248,0.1)",
                                     line=dict(color="#38bdf8", width=2)))
            fig.add_trace(go.Scatter(x=scale_df["n_baskets"], y=scale_df["pcy_fi"],
                                     mode="lines+markers", name="PCY FI",
                                     fill="tozeroy", fillcolor="rgba(167,139,250,0.1)",
                                     line=dict(color="#a78bfa", width=2)))
            apply_theme(fig, "Frequent Itemsets vs Dataset Size", height=360)
            fig.update_layout(xaxis_title="# Baskets",
                              yaxis_title="# Frequent Itemsets")
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            fig = go.Figure(go.Scatter(
                x=scale_df["n_baskets"], y=scale_df["n_rules"],
                mode="lines+markers", fill="tozeroy",
                fillcolor="rgba(245,197,24,0.1)",
                line=dict(color="#f5c518", width=2),
                marker=dict(size=7, color="#f5c518")
            ))
            apply_theme(fig, "Rules Count vs Dataset Size", height=360)
            fig.update_layout(xaxis_title="# Baskets", yaxis_title="# Rules")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("<div class='section-header'>Threshold Sensitivity (full dataset)</div>",
                    unsafe_allow_html=True)
        run_thresh = st.button("Run Threshold Sensitivity (takes ~45s)")
        if run_thresh or "thresh_df" in st.session_state:
            if run_thresh:
                sup_levels = [0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
                thresh_rows = []
                prog_thresh = st.progress(0, "Running threshold experiments...")
                all_baskets_full = [tuple(sorted(b)) for b in basket_list]
                for si, sf in enumerate(sup_levels):
                    s = max(2, int(sf * len(all_baskets_full)))
                    t0 = time.time()
                    ap_res, _, _ = _apriori_core(all_baskets_full, s, 3)
                    elapsed = time.time() - t0
                    rules_res = mine_rules_nocache(
                        serialize_fi(ap_res), len(all_baskets_full), min_conf, 1.0
                    )
                    thresh_rows.append({
                        "sup_pct": sf * 100,
                        "n_fi": sum(len(v) for v in ap_res.values()),
                        "n_rules": len(rules_res),
                        "time_s": round(elapsed, 3),
                    })
                    prog_thresh.progress(int(100*(si+1)/len(sup_levels)))
                prog_thresh.empty()
                st.session_state["thresh_df"] = pd.DataFrame(thresh_rows)

            thresh_df = st.session_state["thresh_df"]
            tc1, tc2, tc3 = st.columns(3)
            for plot_col, col_y, title, color in [
                (tc1, "n_fi",    "Frequent Itemsets", "#38bdf8"),
                (tc2, "n_rules", "Rules",              "#f87171"),
                (tc3, "time_s",  "Runtime (s)",        "#a78bfa"),
            ]:
                fig = go.Figure(go.Scatter(
                    x=thresh_df["sup_pct"], y=thresh_df[col_y],
                    mode="lines+markers", fill="tozeroy",
                    fillcolor=f"rgba({','.join(str(int(c*255)) for c in px.colors.hex_to_rgb(color))},0.12)",
                    line=dict(color=color, width=2),
                    marker=dict(size=7, color=color)
                ))
                apply_theme(fig, title)
                fig.update_layout(xaxis_title="Min Support (%)", yaxis_title=title)
                plot_col.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Click the button above to run scalability experiments.")


st.markdown("---")
st.markdown("""
<div style='font-family:JetBrains Mono,monospace;font-size:0.7rem;
            color:#334155;text-align:center;padding:8px 0'>
CSL7110: Machine Learning with Big Data &nbsp;|&nbsp;
Market Basket Analysis - A-Priori &amp; PCY &nbsp;|&nbsp;
Apache Spark · PySpark · Plotly · Streamlit
</div>""", unsafe_allow_html=True)

