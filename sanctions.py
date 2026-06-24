"""
FinGuard - sanctions screening core
====================================
FinGuard screens a name against sanctions watchlists, the way a compliance
team does thousands of times a day. The hard part is that sanctioned parties
use many spellings and aliases, so exact matching fails - fuzzy matching is
needed, and the match threshold must balance catching true hits against
drowning investigators in false alerts.

This module provides:
  * load_watchlist()    - the watchlist (live OpenSanctions, else bundled OFAC)
  * screen()            - rank watchlist matches for a query name
  * make_test_set()     - labelled positives/negatives to evaluate matching
  * evaluate_thresholds - recall vs false-positive trade-off by threshold
"""
import os
import re
import io
import urllib.request
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

def _find_data(fname):
    here = os.path.dirname(os.path.abspath(__file__))
    for c in (os.path.join(here, fname),
              os.path.join(here, "data", fname),
              os.path.join(os.path.dirname(here), "data", fname)):
        if os.path.exists(c):
            return c
    return os.path.join(here, fname)
RAW = _find_data("ofac_sdn_raw.csv")          # real OFAC snapshot (parsed)
CLEAN = _find_data("watchlist_clean.csv")      # normalised, app-ready
OPENSANCTIONS_URL = ("https://data.opensanctions.org/datasets/latest/"
                     "sanctions/targets.simple.csv")

ORG_HINT = re.compile(
    r"\b(BANK|LTD|LLC|INC|COMPANY|CORP|GROUP|TRADING|SHIPPING|HOLDING|FUND|"
    r"ENTERPRISE|FACTORY|INDUSTR|OIL|GAS|AVIATION|AIRLINES|FZE|JSC|OAO|OOO|"
    r"PJSC|GMBH|LIMITED|ORGANIZATION|FRONT|ARMY|BRIGADE|COMMITTEE|ASSOCIATION|"
    r"BUREAU|MINISTRY|FORCE|DIVISION|REGIMENT|NETWORK)\b", re.I)

SCORERS = {
    "Token-Set": fuzz.token_set_ratio,                 # 0-100
    "Levenshtein": fuzz.ratio,                          # 0-100
    "Jaro-Winkler": JaroWinkler.normalized_similarity,  # 0-1 (scaled below)
}
SCORER_SCALE = {"Token-Set": 1.0, "Levenshtein": 1.0, "Jaro-Winkler": 100.0}


def normalize(s):
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------- cleaning
def _is_surname_frag(f):
    letters = [c for c in f if c.isalpha()]
    if len(letters) < 2:
        return False
    return sum(1 for c in letters if c.isupper()) / len(letters) > 0.7


def _split_names(s):
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    names, cur = [], []
    for p in parts:
        if _is_surname_frag(p):
            if cur:
                names.append(", ".join(cur))
            cur = [p]
        else:
            cur.append(p)
    if cur:
        names.append(", ".join(cur))
    merged = []
    for n in names:
        if merged and n.upper() in ("LTD", "INC", "LLC", "CO", "SA", "PLC", "LLP"):
            merged[-1] += ", " + n
        else:
            merged.append(n)
    return [n for n in merged if any(c.isalpha() for c in n)]


def _display(name):
    """Turn 'SURNAME, Given' into 'Given Surname' for readability."""
    if "," in name:
        a, b = name.split(",", 1)
        return f"{b.strip()} {a.strip()}".title()
    return name.title() if name.isupper() else name


def build_clean_watchlist():
    """Build the normalised watchlist from the real OFAC snapshot."""
    df = pd.read_csv(RAW)
    df = df.drop_duplicates(subset=["names"]).reset_index(drop=True)
    rows = []
    for i, r in df.iterrows():
        nl = _split_names(r["names"])
        if not nl:
            continue
        etype = "Organization" if ORG_HINT.search(str(r["names"])) else "Individual"
        rows.append({
            "entity_id": f"E{i:05d}",
            "primary_name": _display(nl[0]),
            "all_names": " | ".join(nl),
            "n_aliases": max(0, len(nl) - 1),
            "entity_type": etype,
            "source": "OFAC SDN (bundled snapshot)",
        })
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(CLEAN), exist_ok=True)
    out.to_csv(CLEAN, index=False)
    return out


def _from_opensanctions():
    """Fetch the live OpenSanctions consolidated sanctions list (vectorised for speed)."""
    with urllib.request.urlopen(OPENSANCTIONS_URL, timeout=15) as resp:
        df = pd.read_csv(io.StringIO(resp.read().decode("utf-8", "replace")))
    if "name" not in df.columns:
        return pd.DataFrame()
    name = df["name"].astype(str).str.strip()
    mask = name.ne("") & name.ne("nan")
    df, name = df[mask], name[mask]
    aliases = (df["aliases"] if "aliases" in df.columns else pd.Series("", index=df.index)).fillna("").astype(str)
    alias_lists = aliases.str.split(";").map(lambda xs: [a.strip() for a in xs if a and a.strip()])
    schema = (df["schema"] if "schema" in df.columns else pd.Series("", index=df.index)).astype(str)
    ids = (df["id"] if "id" in df.columns else pd.Series(range(len(df)), index=df.index)).astype(str)
    all_names = [" | ".join([nm] + al) for nm, al in zip(name.tolist(), alias_lists.tolist())]
    return pd.DataFrame({
        "entity_id": ids.values,
        "primary_name": name.values,
        "all_names": all_names,
        "n_aliases": alias_lists.map(len).values,
        "entity_type": np.where(schema.values == "Person", "Individual", "Organization"),
        "source": "OpenSanctions (live)",
    })


def load_watchlist(prefer_live=False):
    """Return (entities_df, variants_list, variant_to_entity_idx, source_label).

    Defaults to the fast, bundled OFAC snapshot. Pass prefer_live=True to fetch
    the latest list from OpenSanctions (slower, needs internet)."""
    df = None
    if prefer_live:
        try:
            df = _from_opensanctions()
            if len(df) < 100:
                df = None
        except Exception:
            df = None
    if df is None:
        if not os.path.exists(CLEAN):
            if os.path.exists(RAW):
                build_clean_watchlist()
            else:
                raise FileNotFoundError("No watchlist data available.")
        df = pd.read_csv(CLEAN)
    df = df.reset_index(drop=True)
    # explode into individual name variants for screening (vectorised for speed)
    exploded = df["all_names"].astype(str).str.split(" | ", regex=False).explode().str.strip()
    keep = exploded.str.len() > 0
    exploded = exploded[keep]
    variants = exploded.map(normalize).tolist()
    owner = np.asarray(exploded.index, dtype=int)
    source = df["source"].iloc[0] if len(df) else "unknown"
    return df, variants, owner, source


# ---------------------------------------------------------------- screening
def screen(query, variants, owner, entities, scorer=fuzz.token_set_ratio, scale=1.0, limit=5):
    """Return top entity matches for a query name."""
    q = normalize(query)
    if not q:
        return []
    hits = process.extract(q, variants, scorer=scorer, limit=120)
    best = {}
    for matched_norm, score, vidx in hits:
        score *= scale
        eidx = int(owner[vidx])
        if eidx not in best or score > best[eidx][0]:
            best[eidx] = (score, matched_norm)
    ranked = sorted(best.items(), key=lambda kv: -kv[1][0])[:limit]
    out = []
    for eidx, (score, matched) in ranked:
        e = entities.iloc[eidx]
        out.append({"score": round(float(score), 1), "primary_name": e["primary_name"],
                    "entity_type": e["entity_type"], "n_aliases": int(e["n_aliases"]),
                    "source": e["source"]})
    return out


def band(score):
    if score >= 90:
        return "Strong match - likely the sanctioned party"
    if score >= 80:
        return "Probable match - investigate"
    if score >= 70:
        return "Possible match - review"
    return "Weak / no match"


# ---------------------------------------------------------------- evaluation
_GIVEN = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael",
          "Linda", "David", "Elizabeth", "Daniel", "Sarah", "Paul", "Karen",
          "Mark", "Nancy", "Steven", "Lisa", "Andrew", "Sandra", "Kevin", "Donna"]
_SUR = ["Anderson", "Thompson", "Roberts", "Mitchell", "Carter", "Phillips",
        "Evans", "Turner", "Parker", "Collins", "Edwards", "Stewart", "Morris",
        "Murphy", "Cook", "Rogers", "Morgan", "Bell", "Bailey", "Cooper", "Reed"]


def _perturb(name, rng):
    """Make a realistic 'dirty' query that should still match the entity."""
    n = name
    mode = rng.integers(0, 4)
    if mode == 0 and len(n) > 4:               # typo: drop a character
        i = rng.integers(1, len(n) - 1)
        n = n[:i] + n[i + 1:]
    elif mode == 1 and len(n) > 5:             # typo: swap two adjacent chars
        i = rng.integers(1, len(n) - 2)
        n = n[:i] + n[i + 1] + n[i] + n[i + 2:]
    elif mode == 2:                             # drop a middle token
        toks = n.split()
        if len(toks) > 2:
            toks.pop(rng.integers(1, len(toks) - 1)); n = " ".join(toks)
    else:                                        # vowel substitution (transliteration)
        n = re.sub("ou", "u", n); n = re.sub("ph", "f", n)
    return n


def make_test_set(entities, n_pos=300, n_neg=300, seed=42):
    rng = np.random.default_rng(seed)
    # positives: real sanctioned entities with a perturbed query
    idx = rng.choice(len(entities), size=min(n_pos, len(entities)), replace=False)
    pos = []
    for i in idx:
        e = entities.iloc[int(i)]
        clean = e["primary_name"]
        pos.append({"query": _perturb(clean, rng), "true_idx": int(i)})
    # negatives: random clean (non-sanctioned) names
    neg = [{"query": f"{rng.choice(_GIVEN)} {rng.choice(_SUR)}", "true_idx": -1}
           for _ in range(n_neg)]
    return pos, neg


def evaluate_thresholds(entities, variants, owner, pos, neg,
                        scorer=fuzz.token_set_ratio, scale=1.0, thresholds=range(60, 100, 3)):
    """Return a DataFrame of recall / false-positive-rate / precision by threshold."""
    def best_match(q):
        m = process.extractOne(normalize(q), variants, scorer=scorer)
        return (int(owner[m[2]]), m[1] * scale) if m else (-1, 0)
    pos_eval = [(best_match(p["query"]), p["true_idx"]) for p in pos]
    neg_eval = [best_match(n["query"])[1] for n in neg]
    rows = []
    for t in thresholds:
        tp = sum(1 for (mi, ms), ti in pos_eval if ms >= t and mi == ti)
        recall = tp / len(pos_eval)
        fp = sum(1 for s in neg_eval if s >= t)
        fpr = fp / len(neg_eval)
        flagged = tp + fp
        precision = tp / flagged if flagged else 1.0
        rows.append({"threshold": t, "recall": round(recall, 3),
                     "false_positive_rate": round(fpr, 3), "precision": round(precision, 3)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    if not os.path.exists(CLEAN):
        build_clean_watchlist()
    ents, variants, owner, src = load_watchlist(prefer_live=False)
    print("watchlist:", len(ents), "entities |", len(variants), "name variants | source:", src)
    for q in ["Ayman al Zawahiri", "Vladimir Putin", "Havana International Bank", "John Anderson"]:
        res = screen(q, variants, owner, ents, limit=2)
        top = res[0] if res else {"score": 0, "primary_name": "-"}
        print(f"  query {q!r:30} -> {top['primary_name'][:34]!r} score {top['score']}  [{band(top['score'])}]")
