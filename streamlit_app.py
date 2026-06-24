"""
FinGuard - Sanctions Screening & Name-Matching Intelligence
===========================================================
Run:  streamlit run app.py

Speed notes
-----------
* The two-name comparison is pure local logic (no network) and returns instantly.
* The watchlist defaults to the bundled OFAC snapshot, which loads in well under a
  second and is cached, so every screen returns in a fraction of a second.
* The live OpenSanctions list is an explicit opt-in (sidebar) for when the latest
  data is wanted; it is fetched once and cached.
"""
import os
import sys
import json
import html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sanctions import (load_watchlist, screen, band, make_test_set, evaluate_thresholds,
                       SCORERS, SCORER_SCALE)
from name_match import compare_names

def _find_data(fname):
    here = os.path.dirname(os.path.abspath(__file__))
    for c in (os.path.join(here, fname),
              os.path.join(here, "data", fname),
              os.path.join(os.path.dirname(here), "data", fname)):
        if os.path.exists(c):
            return c
    return os.path.join(here, fname)
CURVES_PATH = _find_data("curves.json")

NAVY, NAVY2, TEAL, TEAL_D = "#1f3b5c", "#2a5279", "#2a9d8f", "#147d72"
GREEN, RED, AMBER, GREY = "#2e7d52", "#c0392b", "#e08a1e", "#5b6b7f"

st.set_page_config(page_title="FinGuard | Sanctions Screening", page_icon="🛡️", layout="wide")

# ----------------------------------------------------------------------------- style
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, textarea { font-family:'Inter',-apple-system,'Segoe UI',sans-serif; }
.block-container { padding-top:1.3rem; padding-bottom:2rem; max-width:1160px; }
#MainMenu, footer { visibility:hidden; }
/* hero */
.fg-hero { background:linear-gradient(120deg,#1f3b5c 0%,#2a5279 55%,#22708a 100%); border-radius:18px;
  padding:22px 28px; color:#fff; display:flex; align-items:center; gap:18px; box-shadow:0 10px 30px rgba(31,59,92,.22); margin-bottom:18px; }
.fg-logo { width:56px;height:56px;border-radius:14px;background:rgba(255,255,255,.15);display:flex;
  align-items:center;justify-content:center;font-size:31px;border:1px solid rgba(255,255,255,.28); }
.fg-hero h1 { margin:0;font-size:31px;font-weight:800;letter-spacing:.3px;color:#fff;line-height:1.1; }
.fg-hero p { margin:3px 0 0;font-size:14.5px;color:#eaf1f7;opacity:.95; }
.fg-stats { margin-left:auto;text-align:right;font-size:13px;color:#eaf1f7;line-height:1.6; }
.fg-stats b { font-size:19px;color:#fff;font-weight:800; }
/* tabs */
.stTabs [data-baseweb="tab-list"] { gap:6px;border-bottom:2px solid #e2e7ee; }
.stTabs [data-baseweb="tab"] { height:46px;padding:0 18px;font-weight:600;font-size:15px;color:#5b6b7f;border-radius:10px 10px 0 0; }
.stTabs [aria-selected="true"] { color:#1f3b5c !important;background:#eef3f8;border-bottom:3px solid #2a9d8f; }
/* buttons */
.stButton>button, .stDownloadButton>button { background:#1f3b5c;color:#fff;border:none;border-radius:10px;
  padding:.55rem 1.1rem;font-weight:600;font-size:15px;transition:.15s; }
.stButton>button:hover, .stDownloadButton>button:hover { background:#147d72;color:#fff; }
.stButton>button:active, .stButton>button:focus { color:#fff;box-shadow:0 0 0 3px rgba(42,157,143,.35); }
/* inputs */
.stTextInput>div>div>input, .stTextInput>div>div { border-radius:9px; }
.stTextInput label, .stSelectbox label, .stSlider label { font-weight:600;color:#2a5279;font-size:13.5px; }
/* metric cards */
[data-testid="stMetric"] { background:#f7f9fb;border:1px solid #e2e7ee;border-radius:14px;padding:14px 16px;box-shadow:0 1px 2px rgba(16,24,40,.04); }
[data-testid="stMetricValue"] { color:#1f3b5c;font-weight:800;font-size:30px; }
[data-testid="stMetricLabel"] { color:#5b6b7f;font-weight:600; }
/* verdict card */
.fg-verdict { border-radius:15px;padding:18px 22px;margin:8px 0 6px;border-left:7px solid;display:flex;align-items:flex-start;gap:14px; }
.fg-verdict .ico { font-size:26px;line-height:1.1; }
.fg-verdict .t { font-weight:800;font-size:18px;margin:0 0 3px; }
.fg-verdict .s { font-size:14.5px;margin:0;opacity:.95;line-height:1.45; }
.v-match { background:#fcebe8;border-color:#c0392b;color:#7d241a; }
.v-poss  { background:#fdf4e3;border-color:#e08a1e;color:#8a5a12; }
.v-no    { background:#e9f6ee;border-color:#2e7d52;color:#1e5637; }
/* confidence bar */
.fg-confwrap { margin:12px 0 6px; }
.fg-conflabel { display:flex;justify-content:space-between;font-size:13px;color:#5b6b7f;font-weight:600;margin-bottom:5px; }
.fg-conftrack { height:13px;background:#e9edf3;border-radius:8px;overflow:hidden; }
.fg-conffill { height:100%;border-radius:8px;transition:width .4s; }
/* token table */
.fg-table { width:100%;border-collapse:collapse;font-size:14px;margin:4px 0 2px;border-radius:10px;overflow:hidden;box-shadow:0 1px 2px rgba(16,24,40,.05); }
.fg-table th { background:#1f3b5c;color:#fff;text-align:left;padding:10px 13px;font-weight:600;font-size:13px; }
.fg-table td { padding:10px 13px;border-bottom:1px solid #eef2f7;color:#1b2733;background:#fff; }
.fg-table tr:last-child td { border-bottom:none; }
.rel { display:inline-block;padding:3px 11px;border-radius:20px;font-size:12.5px;font-weight:600; }
.rel-exact { background:#dff3e6;color:#1e5637; }
.rel-variant { background:#dce9fb;color:#1f3b5c; }
.rel-spelling { background:#fdeccf;color:#8a5a12; }
.rel-diff { background:#fad9d2;color:#7d241a; }
.sc-strong { background:#fad9d2;color:#7d241a;font-weight:700;border-radius:6px;padding:2px 9px; }
.sc-prob { background:#fdeccf;color:#8a5a12;font-weight:700;border-radius:6px;padding:2px 9px; }
.sc-poss { background:#eef2f7;color:#2a5279;font-weight:700;border-radius:6px;padding:2px 9px; }
.sc-weak { color:#5b6b7f;font-weight:700;padding:2px 9px; }
/* section heading + reasons + chips */
.fg-h { font-weight:700;color:#1f3b5c;font-size:15px;margin:18px 0 7px; }
.fg-reason { display:flex;gap:9px;align-items:flex-start;margin:6px 0;font-size:14.5px;color:#1b2733; }
.fg-reason .dot { color:#2a9d8f;font-weight:800;font-size:15px;line-height:1.3; }
.fg-chiprow { margin:6px 0 2px; }
.fg-chipname { font-weight:700;color:#1f3b5c;margin-right:4px; }
.fg-chip { display:inline-block;background:#eef3f8;border:1px solid #d8e1ea;color:#2a5279;border-radius:18px;padding:3px 11px;margin:3px 4px 0 0;font-size:13px;font-weight:500; }
/* alert + note boxes */
.fg-alert { border-radius:13px;padding:14px 18px;font-weight:600;font-size:15px;border-left:6px solid;margin:6px 0; }
.a-hit { background:#fcebe8;border-color:#c0392b;color:#7d241a; }
.a-clear { background:#e9f6ee;border-color:#2e7d52;color:#1e5637; }
.fg-note { background:#eef3f8;border:1px solid #dbe4ee;border-radius:11px;padding:11px 15px;font-size:13.5px;color:#2a5279;margin-top:12px; }
.fg-sub { color:#5b6b7f;font-size:14.5px;margin:2px 0 6px; }
hr { margin:.6rem 0; }
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "axes.edgecolor": "#c9d2dc",
                     "axes.labelcolor": "#1b2733", "text.color": "#1b2733",
                     "xtick.color": "#5b6b7f", "ytick.color": "#5b6b7f", "axes.titlecolor": NAVY})


def _clean_ax(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#eef2f7", lw=1)
    ax.set_axisbelow(True)


def esc(s):
    return html.escape(str(s))


# ----------------------------------------------------------------------------- data (cached)
@st.cache_data(show_spinner=False)
def get_watchlist(live: bool):
    return load_watchlist(prefer_live=live)


@st.cache_data(show_spinner=False)
def get_curve(live: bool):
    if not live and os.path.exists(CURVES_PATH):
        raw = json.load(open(CURVES_PATH))
        return {n: pd.DataFrame(raw[n]) for n in raw}
    ents, variants, owner, _ = get_watchlist(live)
    pos, neg = make_test_set(ents, n_pos=300, n_neg=300, seed=42)
    return {n: evaluate_thresholds(ents, variants, owner, pos, neg, scorer=sc,
                                   scale=SCORER_SCALE[n], thresholds=range(60, 100, 2))
            for n, sc in SCORERS.items()}


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown(f"<div style='font-size:30px;text-align:center'>🛡️</div>"
                f"<div style='text-align:center;font-weight:800;color:{NAVY};font-size:20px;margin-top:-4px'>FinGuard</div>"
                f"<div style='text-align:center;color:{GREY};font-size:12.5px;margin-bottom:10px'>Sanctions screening &amp; name matching</div>",
                unsafe_allow_html=True)
    st.divider()
    use_live = st.checkbox("Use live OpenSanctions data", value=False,
                           help="Off (recommended): fast bundled OFAC snapshot. "
                                "On: fetch the latest consolidated list from OpenSanctions (needs internet; slower on first load).")
    st.caption("The two-name comparison works fully offline and is unaffected by this setting.")
    st.divider()
    st.markdown(f"<div style='color:{GREY};font-size:12.5px;line-height:1.6'>"
                "FinGuard is a decision-support tool. It explains every verdict; a human makes the final call.<br><br>"
                "Built for a PGDSBA capstone on real, public sanctions data.</div>", unsafe_allow_html=True)

if use_live:
    with st.spinner("Fetching the latest list from OpenSanctions..."):
        ents, variants, owner, source = get_watchlist(True)
else:
    ents, variants, owner, source = get_watchlist(False)

# ----------------------------------------------------------------------------- hero
st.markdown(f"""
<div class="fg-hero">
  <div class="fg-logo">🛡️</div>
  <div>
    <h1>FinGuard</h1>
    <p>Sanctions Screening &amp; Name-Matching Intelligence &nbsp;·&nbsp; {esc(source)}</p>
  </div>
  <div class="fg-stats"><b>{len(ents):,}</b> sanctioned entities<br><b>{len(variants):,}</b> name variants</div>
</div>
""", unsafe_allow_html=True)

t0, t1, t2, t3 = st.tabs(["🆚  Compare Two Names", "🔎  Screen a Name",
                          "📊  Watchlist Dashboard", "🎯  Matching Quality"])

# ============================================================ TAB 0: pairwise compare
with t0:
    st.markdown("<div class='fg-h' style='font-size:18px;margin-top:6px'>Compare a customer name against a watchlist name</div>", unsafe_allow_html=True)
    st.markdown("<div class='fg-sub'>When a screening alert pairs your customer with a sanctioned party, FinGuard decides whether "
                "they are the same identity using name intelligence &mdash; variants, nicknames, particles, initials and reordering &mdash; "
                "so a true first-name difference is flagged even when only one letter changes.</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        name_cust = st.text_input("Customer name", "Mohammed Ali")
        dob_cust = st.text_input("Customer date of birth (optional)", "", placeholder="e.g. 1975 or 12/03/1975")
    with c2:
        name_sdn = st.text_input("Watchlist / SDN name", "Mohammed Al Ali")
        dob_sdn = st.text_input("Watchlist date of birth (optional)", "", placeholder="e.g. 1975 or 12/03/1975")

    go = st.button("Compare names", type="primary", use_container_width=True)
    if go or (name_cust and name_sdn):
        r = compare_names(name_cust, name_sdn, dob_a=dob_cust or None, dob_b=dob_sdn or None)
        v = r["verdict"]
        meta = {"Match": ("v-match", "🔴", RED, "Name match"),
                "Possible match": ("v-poss", "🟡", AMBER, "Possible match"),
                "No match": ("v-no", "🟢", GREEN, "No match")}[v]
        cls, ico, barcol, title = meta
        sub = r["headline"].split(" - ", 1)[1] if " - " in r["headline"] else r["headline"]
        st.markdown(f"<div class='fg-verdict {cls}'><div class='ico'>{ico}</div>"
                    f"<div><p class='t'>{esc(title)}</p><p class='s'>{esc(sub)}</p></div></div>", unsafe_allow_html=True)
        conf = r["confidence"]
        st.markdown(f"<div class='fg-confwrap'><div class='fg-conflabel'><span>Match confidence</span>"
                    f"<span>{conf} / 100</span></div><div class='fg-conftrack'>"
                    f"<div class='fg-conffill' style='width:{conf}%;background:{barcol}'></div></div></div>", unsafe_allow_html=True)

        if r["pairs"]:
            relmap = {"exact": ("Exact match", "rel-exact"), "variant": ("Variant / transliteration", "rel-variant"),
                      "initial": ("Initial", "rel-variant"), "spelling": ("Spelling difference", "rel-spelling"),
                      "different_name": ("Different name", "rel-diff"), "mismatch": ("No match", "rel-diff")}
            rows = ""
            for p in r["pairs"]:
                lab, rc = relmap.get(p["relation"], (p["relation"], ""))
                rows += (f"<tr><td>{esc(p['a'].title())}</td><td>{esc(p['b'].title())}</td>"
                         f"<td><span class='rel {rc}'>{lab}</span></td><td>{int(p['strength']*100)}</td></tr>")
            st.markdown("<div class='fg-h'>Token-by-token comparison</div>"
                        "<table class='fg-table'><thead><tr><th>Customer token</th><th>Watchlist token</th>"
                        f"<th>Relationship</th><th>Strength</th></tr></thead><tbody>{rows}</tbody></table>", unsafe_allow_html=True)

        if r["reasons"]:
            rh = "".join(f"<div class='fg-reason'><span class='dot'>&#9656;</span><span>{esc(x)}</span></div>" for x in r["reasons"])
            st.markdown(f"<div class='fg-h'>Why FinGuard decided this</div>{rh}", unsafe_allow_html=True)

        if r["variants"]:
            chips = ""
            for base, vs in r["variants"].items():
                chips += (f"<div class='fg-chiprow'><span class='fg-chipname'>{esc(base)}</span> also appears as: "
                          + "".join(f"<span class='fg-chip'>{esc(x)}</span>" for x in vs) + "</div>")
            st.markdown(f"<div class='fg-h'>Known name variants to watch for</div>{chips}", unsafe_allow_html=True)

    st.markdown("<div class='fg-note'>Try the hard cases: <b>Mark Thomas vs Mary Thomas</b> (mismatch), "
                "<b>Mohammed Ali vs Mohammed Al Ali</b> (match), <b>Bill Clinton vs William Clinton</b> (nickname), "
                "or <b>Vladimir Putin vs Volodymyr Putin</b> (transliteration).</div>", unsafe_allow_html=True)

# ============================================================ TAB 1: screen a name
with t1:
    st.markdown("<div class='fg-h' style='font-size:18px;margin-top:6px'>Screen a customer or counterparty name</div>", unsafe_allow_html=True)
    st.markdown("<div class='fg-sub'>FinGuard scores the name against every name and alias on the watchlist and returns the "
                "closest matches, ranked, with a score and a band.</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([3, 1.5, 1.5])
    query = c1.text_input("Name to screen", "Havana International Bank")
    algo = c2.selectbox("Matching method", list(SCORERS.keys()), index=0)
    threshold = c3.slider("Alert threshold", 60, 99, 86)
    if st.button("Screen name", type="primary", use_container_width=True) or query:
        res = screen(query, variants, owner, ents, scorer=SCORERS[algo], scale=SCORER_SCALE[algo], limit=6)
        top = res[0]["score"] if res else 0
        if top >= threshold:
            st.markdown(f"<div class='fg-alert a-hit'>&#9888;&nbsp; ALERT &mdash; best match scores "
                        f"{top:.0f} (&ge; {threshold}). Refer for investigation.</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='fg-alert a-clear'>&#10003;&nbsp; No match at or above the threshold "
                        f"(best score {top:.0f}).</div>", unsafe_allow_html=True)
        if res:
            def sc_class(v):
                return ("sc-strong" if v >= 90 else "sc-prob" if v >= 80 else "sc-poss" if v >= 70 else "sc-weak")
            rows = ""
            for m in res:
                bd = band(m["score"]).split(" - ")[0]
                rows += (f"<tr><td><span class='{sc_class(m['score'])}'>{m['score']:.0f}</span></td>"
                         f"<td>{esc(bd)}</td><td>{esc(m['primary_name'])}</td>"
                         f"<td>{esc(m['entity_type'])}</td><td>{m['n_aliases']}</td></tr>")
            st.markdown("<div class='fg-h'>Closest watchlist matches</div>"
                        "<table class='fg-table'><thead><tr><th>Score</th><th>Band</th><th>Watchlist entity</th>"
                        f"<th>Type</th><th>Aliases</th></tr></thead><tbody>{rows}</tbody></table>", unsafe_allow_html=True)
    st.markdown("<div class='fg-note'>Try <b>Havana International Bank</b> (found via an alias), <b>Vladimir Putin</b>, "
                "or a clean name like <b>James Anderson</b> to see the score and alert change.</div>", unsafe_allow_html=True)

# ============================================================ TAB 2: dashboard
with t2:
    st.markdown("<div class='fg-h' style='font-size:18px;margin-top:6px'>The sanctions watchlist at a glance</div>", unsafe_allow_html=True)
    k = st.columns(4)
    k[0].metric("Sanctioned entities", f"{len(ents):,}")
    k[1].metric("Individuals", f"{int((ents.entity_type=='Individual').sum()):,}")
    k[2].metric("Organisations", f"{int((ents.entity_type=='Organization').sum()):,}")
    k[3].metric("Use aliases", f"{(ents.n_aliases>0).mean()*100:.0f}%")
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("<div class='fg-h'>Watchlist composition</div>", unsafe_allow_html=True)
        vc = ents["entity_type"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 3.2))
        bars = ax.bar(vc.index.astype(str), vc.values, color=[NAVY, TEAL], width=0.55)
        for i, val in enumerate(vc.values):
            ax.text(i, val + max(vc.values) * 0.02, f"{val:,}", ha="center", color=NAVY, fontweight="bold")
        _clean_ax(ax)
        ax.set_ylim(0, max(vc.values) * 1.15)
        st.pyplot(fig, use_container_width=True)
    with g2:
        st.markdown("<div class='fg-h'>Aliases per entity (why fuzzy matching is needed)</div>", unsafe_allow_html=True)
        bk = pd.cut(ents["n_aliases"], bins=[-1, 0, 1, 2, 4, 1000],
                    labels=["0", "1", "2", "3-4", "5+"]).value_counts().reindex(["0", "1", "2", "3-4", "5+"])
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.bar(bk.index.astype(str), bk.values, color=NAVY, width=0.6)
        _clean_ax(ax)
        ax.set_xlabel("Number of aliases")
        st.pyplot(fig, use_container_width=True)
    st.markdown(f"<div class='fg-note'>The most-aliased entity carries <b>{int(ents.n_aliases.max())}</b> name variants &mdash; "
                "exactly why exact matching fails and fuzzy matching is required.</div>", unsafe_allow_html=True)

# ============================================================ TAB 3: matching quality
with t3:
    st.markdown("<div class='fg-h' style='font-size:18px;margin-top:6px'>How good is the matching &mdash; and where to set the threshold?</div>", unsafe_allow_html=True)
    st.markdown("<div class='fg-sub'>Evaluated on 300 disguised sanctioned names (which should match) and 300 clean names "
                "(which should not). The threshold trades off catching true hits against the volume of false alerts.</div>", unsafe_allow_html=True)
    curves = get_curve(use_live)
    ts = curves["Token-Set"]
    thr = st.slider("Match threshold", 60, 98, 86, 2, key="thr3")
    row = ts.iloc[(ts.threshold - thr).abs().argmin()]
    m = st.columns(3)
    m[0].metric("Recall (true hits caught)", f"{row.recall*100:.0f}%")
    m[1].metric("False-positive rate", f"{row.false_positive_rate*100:.0f}%")
    m[2].metric("Precision", f"{row.precision*100:.0f}%")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(ts.threshold, ts.recall, marker="o", color=TEAL, lw=2.2, label="Recall (hits caught)")
    ax.plot(ts.threshold, ts.false_positive_rate, marker="s", color=RED, lw=2.2, label="False-positive rate")
    ax.plot(ts.threshold, ts.precision, marker="^", color=NAVY, lw=2.2, label="Precision")
    ax.axvline(thr, color=GREY, ls="--", lw=1.5)
    ax.text(thr + 0.3, 0.04, f"threshold {thr}", color=GREY, fontsize=9)
    _clean_ax(ax)
    ax.set_xlabel("Match threshold")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9, loc="center right")
    st.pyplot(fig, use_container_width=True)

    st.markdown("<div class='fg-h'>Which matching method works best? (at threshold 86)</div>", unsafe_allow_html=True)
    rows = ""
    for n in SCORERS:
        d = curves[n]
        rr = d.iloc[(d.threshold - 86).abs().argmin()]
        tag = " <span style='color:#2a9d8f;font-weight:700'>(chosen)</span>" if n == "Token-Set" else ""
        rows += (f"<tr><td><b>{esc(n)}</b>{tag}</td><td>{rr.recall*100:.0f}%</td>"
                 f"<td>{rr.false_positive_rate*100:.0f}%</td><td>{rr.precision*100:.0f}%</td></tr>")
    st.markdown("<table class='fg-table'><thead><tr><th>Method</th><th>Recall</th><th>False positives</th>"
                f"<th>Precision</th></tr></thead><tbody>{rows}</tbody></table>", unsafe_allow_html=True)
    st.markdown("<div class='fg-note'>Token-Set matching wins: it handles reordered names and aliases, giving the best "
                "recall at a low false-positive rate. This is the method FinGuard uses by default.</div>", unsafe_allow_html=True)
