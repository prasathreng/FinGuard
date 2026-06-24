"""
FinGuard - pairwise name matching ("is this a real match or a false positive?")
================================================================================
A sanctions investigator's core decision is not "is this name on the list" but
"this alert paired my customer with an SDN - are they actually the same person?"

A plain similarity score fails here:
  * "Mark Thomas" vs "Mary Thomas"   -> one letter, but a NAME MISMATCH
  * "Mohammed Ali" vs "Mohammed Al Ali" -> looks different, but the SAME name
    (only a name particle differs)

This module decides using name intelligence: a built-in knowledge base of
transliteration variants, nicknames, name particles, initials and reordering -
the kind of knowledge a site like behindthename.com provides, bundled in so the
tool always works offline. An optional online lookup can enrich it (see below).
"""
import os
import re
import unicodedata
import urllib.request
import json
from rapidfuzz.distance import JaroWinkler

# --------------------------------------------------------------- knowledge base
# Name particles / connectors that usually are NOT a real difference.
PARTICLES = {
    "al", "el", "ul", "bin", "ben", "ibn", "bint", "abu", "umm", "abd", "abdel",
    "van", "von", "der", "den", "de", "del", "della", "di", "da", "dos", "das",
    "la", "le", "du", "san", "santa", "st", "mac", "mc", "o", "ait", "ould",
}

# Groups of names that are variants / transliterations / nicknames of EACH OTHER.
_VARIANT_GROUPS = [
    # --- Arabic / Muslim transliterations (very common on sanctions lists) ---
    ["mohammed", "muhammad", "mohamed", "mohammad", "muhammed", "mohd", "muhamms", "mahomet"],
    ["ahmed", "ahmad", "achmed"],
    ["abdullah", "abdallah", "abdellah", "abdulla", "abdALLAH".lower()],
    ["abdul", "abdel", "abd"],
    ["ali", "aly"],
    ["hassan", "hasan", "hassane"],
    ["hussein", "hussain", "husayn", "husein", "hossein", "houssein"],
    ["yusuf", "yousef", "youssef", "yousuf", "yusif", "yusuph", "joseph"],
    ["ibrahim", "ebrahim", "ibraheem", "abraham"],
    ["khalid", "khaled", "khalED".lower()],
    ["omar", "umar", "omer"],
    ["ismail", "ismael", "ismayil", "esmail"],
    ["mahmoud", "mahmud", "mehmood", "mahmood"],
    ["said", "sayed", "sayyid", "syed", "saeed", "sayeed"],
    ["fatima", "fatimah", "fatma", "fatemeh"],
    ["aisha", "aysha", "ayesha", "aishah"],
    ["sheikh", "shaykh", "sheik", "shaikh"],
    ["jamal", "jamil", "gamal"],
    ["tariq", "tarek", "tarik"],
    ["rashid", "rasheed", "rachid"],
    ["nasser", "nasir", "naser", "nassir"],
    ["qasim", "kassim", "qassem", "kassem"],
    ["yahya", "yehia", "yahia"],
    ["zaid", "zayd", "zaiyd"],
    # --- Russian / Ukrainian transliterations (sanctions-relevant) ---
    ["vladimir", "volodymyr", "wladimir"],
    ["aleksandr", "alexander", "oleksandr", "aleksander", "alexandr", "alex", "sasha"],
    ["sergei", "sergey", "serhii", "serguei"],
    ["dmitri", "dmitry", "dmitriy", "dmytro"],
    ["mikhail", "michael", "mykhailo", "michail"],
    ["yevgeny", "evgeny", "yevgeni", "evgeni", "eugene"],
    ["nikolai", "nikolay", "mykola", "nicholas"],
    ["andrei", "andrey", "andriy", "andrew"],
    ["pavel", "paul", "pawel"],
    ["ivan", "john", "ioan"],
    ["yuri", "yury", "yuriy", "georgiy"],
    ["viktor", "victor"],
    ["konstantin", "constantine"],
    ["igor", "ihor"],
    # --- Western nicknames <-> formal ---
    ["william", "will", "bill", "billy", "liam", "wm"],
    ["robert", "rob", "bob", "bobby", "bert"],
    ["richard", "rick", "ricky", "dick", "rich"],
    ["james", "jim", "jimmy", "jamie", "jas"],
    ["johnny", "jack", "jackie"],
    ["michael", "mike", "mickey", "mick"],
    ["elizabeth", "liz", "beth", "betty", "eliza", "lizzie", "betsy"],
    ["margaret", "maggie", "peggy", "meg", "marge"],
    ["katherine", "catherine", "kate", "katie", "kathy", "cathy", "kat"],
    ["anthony", "tony"],
    ["charles", "charlie", "chuck", "chas"],
    ["edward", "ed", "eddie", "ted", "ned"],
    ["thomas", "tom", "tommy"],
    ["daniel", "dan", "danny"],
    ["joseph", "joe", "joey"],
    ["david", "dave", "davey"],
    ["andrew", "andy", "drew"],
    ["nicholas", "nick", "nicky"],
    ["christopher", "chris", "kit"],
    ["matthew", "matt"],
    ["benjamin", "ben", "benny"],
    ["samuel", "sam", "sammy"],
    ["patrick", "pat", "paddy"],
    ["stephen", "steven", "steve", "stevie"],
    ["timothy", "tim", "timmy"],
    ["ronald", "ron", "ronnie"],
    ["kenneth", "ken", "kenny"],
    ["theodore", "ted", "teddy"],
    ["francis", "frank", "frankie"],
    ["raymond", "ray"],
    ["lawrence", "larry"],
    ["gregory", "greg"],
    ["jeffrey", "jeff"],
    ["vincent", "vince", "vinny"],
    ["albert", "al", "bert"],
    # --- South Asian common variants ---
    ["mohan", "mohun"],
    ["sanjay", "sanjai"],
    ["rajesh", "rajes"],
    ["krishna", "krishnan", "kishan"],
]

_VAR = {}
for _i, _g in enumerate(_VARIANT_GROUPS):
    for _n in _g:
        _VAR[_n] = _i

# A broad set of recognised given names (so we can tell two DIFFERENT first names
# apart from a mere typo). Built from the variant groups plus common given names.
_COMMON_GIVEN = {
    "mark", "mary", "maria", "marie", "martha", "marc", "marcus", "marco",
    "john", "jane", "joan", "jean", "james", "jacob", "jason", "julia", "june",
    "paul", "peter", "philip", "patricia", "pamela", "rachel", "rebecca",
    "sarah", "sara", "susan", "sandra", "linda", "laura", "lisa", "karen",
    "george", "gary", "gerald", "harold", "henry", "helen", "donald", "donna",
    "kevin", "brian", "bruce", "frank", "frances", "carl", "carol", "diane",
    "amir", "amira", "ahmed", "ali", "hassan", "hussein", "ibrahim", "khalid",
    "omar", "yusuf", "mohammed", "fatima", "aisha", "layla", "leila", "noor",
    "vladimir", "sergei", "dmitri", "ivan", "anna", "olga", "natalia", "boris",
    "raj", "ravi", "amit", "anil", "sunil", "vijay", "ramesh", "suresh",
    "priya", "neha", "pooja", "deepak", "rohit", "rahul", "arjun", "kiran",
    "william", "robert", "richard", "michael", "thomas", "daniel", "joseph",
    "david", "andrew", "nicholas", "christopher", "anthony", "charles", "edward",
}
GIVEN_NAMES = set(_VAR.keys()) | _COMMON_GIVEN


# --------------------------------------------------------------- helpers
def _strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def normalize(s):
    s = _strip_accents(str(s).lower().strip())
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _same_group(a, b):
    return a in _VAR and b in _VAR and _VAR[a] == _VAR[b]


def _token_relation(a, b):
    """Classify how two name tokens relate. Returns (relation, strength 0-1)."""
    if a == b:
        return ("exact", 1.0)
    if (len(a) == 1 and b[:1] == a) or (len(b) == 1 and a[:1] == b):
        return ("initial", 0.95)
    if _same_group(a, b):
        return ("variant", 0.92)
    jw = JaroWinkler.normalized_similarity(a, b)
    # the key rule: two RECOGNISED but DIFFERENT given names = a real mismatch,
    # even if their spelling is close (Mark vs Mary).
    if a in GIVEN_NAMES and b in GIVEN_NAMES and not _same_group(a, b):
        return ("different_name", round(min(jw, 0.40), 3))
    if jw >= 0.88:
        return ("spelling", round(jw, 3))
    return ("mismatch", round(jw, 3))


def _split(name):
    toks = normalize(name).split()
    core = [t for t in toks if t not in PARTICLES]
    parts = [t for t in toks if t in PARTICLES]
    return toks, core, parts


def _year(s):
    m = re.search(r"(18|19|20)\d{2}", str(s))
    return m.group(0) if m else None


# --------------------------------------------------------------- variant lookup
def variants_of(name, online=False):
    """Known variants/transliterations of a single given name (KB + optional online)."""
    n = normalize(name)
    out = set()
    if n in _VAR:
        out = set(_VARIANT_GROUPS[_VAR[n]]) - {n}
    if online:
        try:
            out |= set(online_variants(n))
        except Exception:
            pass
    return sorted(out)


def online_variants(name):
    """OPTIONAL: enrich variants from behindthename.com.
    Off by default. Set environment variable BEHINDTHENAME_KEY to enable.
    Never required - the bundled knowledge base works without any key/internet."""
    key = os.environ.get("BEHINDTHENAME_KEY")
    if not key:
        return []
    url = (f"https://www.behindthename.com/api/related.json?"
           f"name={urllib.parse.quote(name)}&key={key}")
    with urllib.request.urlopen(url, timeout=8) as r:
        data = json.loads(r.read().decode("utf-8", "replace"))
    return [normalize(n) for n in data.get("names", [])]


# --------------------------------------------------------------- the adjudicator
def compare_names(name_a, name_b, dob_a=None, dob_b=None, online=False):
    """Decide whether two names refer to the same identity."""
    toks_a, core_a, part_a = _split(name_a)
    toks_b, core_b, part_b = _split(name_b)
    if not core_a or not core_b:
        return {"verdict": "No match", "confidence": 0, "headline": "Please enter both names.",
                "pairs": [], "reasons": [], "variants": {}}

    # greedy best-strength alignment of core tokens (order-independent)
    cand = []
    for i, ta in enumerate(core_a):
        for j, tb in enumerate(core_b):
            rel, st = _token_relation(ta, tb)
            cand.append((st, i, j, ta, tb, rel))
    cand.sort(key=lambda x: -x[0])
    usedA, usedB, pairs = set(), set(), []
    for st, i, j, ta, tb, rel in cand:
        if i in usedA or j in usedB:
            continue
        usedA.add(i); usedB.add(j)
        pairs.append({"a": ta, "b": tb, "relation": rel, "strength": round(st, 3)})
    leftover_a = [core_a[i] for i in range(len(core_a)) if i not in usedA]
    leftover_b = [core_b[j] for j in range(len(core_b)) if j not in usedB]

    strengths = [p["strength"] for p in pairs]
    has_diffname = any(p["relation"] == "different_name" for p in pairs)
    base = sum(strengths) / len(strengths) if strengths else 0.0
    substantive_extra = [t for t in (leftover_a + leftover_b) if len(t) > 1]
    penalty = sum(0.03 if len(t) == 1 else 0.13 for t in (leftover_a + leftover_b))
    score = max(0.0, base - penalty)

    # optional date-of-birth check (secondary identifier)
    dob_relation = None
    if dob_a and dob_b:
        ya, yb = _year(dob_a), _year(dob_b)
        if ya and yb:
            if ya == yb:
                score = min(1.0, score + 0.05); dob_relation = ("match", ya)
            else:
                dob_relation = ("mismatch", f"{ya} vs {yb}")

    # verdict
    if has_diffname:
        verdict = "No match"
    elif dob_relation and dob_relation[0] == "mismatch" and score < 0.95:
        verdict = "No match"
    elif score >= 0.85 and strengths and all(s >= 0.9 for s in strengths) and not substantive_extra:
        verdict = "Match"
    elif score >= 0.72:
        verdict = "Possible match"
    else:
        verdict = "No match"

    # human-readable reasons
    reasons = []
    label = {"exact": "matches exactly", "variant": "are known variants/transliterations of the same name",
             "initial": "matches as an initial", "spelling": "differ only in spelling (possible typo/transliteration)",
             "different_name": "are DIFFERENT given names, not variants - this is a name mismatch",
             "mismatch": "do not match"}
    for p in pairs:
        connector = "" if p["relation"] in ("matches exactly",) else ""
        reasons.append(f"\u2018{p['a'].title()}\u2019 / \u2018{p['b'].title()}\u2019 {label[p['relation']]}.")
    if part_a or part_b:
        only = part_a if part_a and not part_b else (part_b if part_b and not part_a else [])
        ps = ", ".join(sorted(set(part_a + part_b)))
        if only:
            reasons.append(f"One name includes the particle(s) \u2018{', '.join(sorted(set(only)))}\u2019 "
                           f"(a common connector) - usually not a real difference.")
    if substantive_extra:
        if leftover_a:
            reasons.append(f"Extra name(s) only on the customer side: {', '.join(t.title() for t in leftover_a if len(t)>1)}.")
        if [t for t in leftover_b if len(t) > 1]:
            reasons.append(f"Extra name(s) only on the watchlist side: {', '.join(t.title() for t in leftover_b if len(t)>1)}.")
    if dob_relation:
        reasons.append("Date of birth matches (" + dob_relation[1] + ")." if dob_relation[0] == "match"
                       else "Date of birth differs (" + dob_relation[1] + ") - strong sign of a different person.")

    headline = {
        "Match": "NAME MATCH - the names refer to the same identity. Treat as a genuine hit and escalate.",
        "Possible match": "POSSIBLE MATCH - similar but not certain. Verify with date of birth, nationality or ID before deciding.",
        "No match": "NO MATCH - the names are different. This looks like a false positive that can be cleared.",
    }[verdict]
    if verdict == "No match" and has_diffname:
        dn = next(p for p in pairs if p["relation"] == "different_name")
        headline += f" (First names \u2018{dn['a'].title()}\u2019 and \u2018{dn['b'].title()}\u2019 are different names.)"

    # variant intelligence for the names involved
    variants = {}
    for t in set(core_a + core_b):
        v = variants_of(t, online=online)
        if v:
            variants[t.title()] = [x.title() for x in v][:10]

    return {"verdict": verdict, "confidence": int(round(score * 100)), "headline": headline,
            "pairs": pairs, "reasons": reasons, "variants": variants,
            "dob_relation": dob_relation}


if __name__ == "__main__":
    tests = [
        ("Mark Thomas", "Mary Thomas"),
        ("Mohammed Ali", "Mohammed Al Ali"),
        ("Bill Clinton", "William Clinton"),
        ("J. Smith", "John Smith"),
        ("Thomas, Mark", "Mark Thomas"),
        ("Robert Mueller", "Robert Miller"),
        ("Vladimir Putin", "Volodymyr Putin"),
        ("Ali", "Mohammed Ali"),
        ("Ayman al-Zawahiri", "Ayman Al Zawahiri"),
    ]
    for a, b in tests:
        r = compare_names(a, b)
        print(f"{a:22} vs {b:20} -> {r['verdict']:14} ({r['confidence']:3}) ")
