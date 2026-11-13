#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_translator.py  --  4-tier cascading RBMT + tier-weighted RAG retriever for English <-> Mwaghavul.

Usage:
    python test_translator.py "It was my mother who bought the laptop at the market."
    python test_translator.py --lang mw "Ri se mbiise ni."
    python test_translator.py --json "..."            # machine-readable packet (used by run_eval.py)

Resources (all four uploaded files):
    rbmt_transfer_lexicon.json     lexical transfer, GOLD > SILVER > BRONZE (equivalent-level tiers)
    rbmt_cascading_grammar.json    word order / TAM / negation / valency, GOLD > SILVER > BRONZE > TIER_4_GENERALIZED
    hybrid_llm_parallel_corpus.jsonl  RAG index (3 GOLD + 2 SILVER + 2 high-confidence BRONZE)
    audit_and_filtered_examples.xlsx  (read by run_eval.py only: held-out dictionary examples are excluded from the RAG index)

Blind-evaluation hooks: Engine(exclude_ids=..., exclude_refs=...) removes held-out sentences, their cluster
siblings and near-duplicates from the RAG index AND from the induced closed-class tables.
"""
import argparse, collections, json, math, os, re, sys, unicodedata, difflib
from pathlib import Path

DATA = Path(os.environ.get("MW_DATA", "/mnt/user-data/uploads"))
TIER_RANK = {"GOLD": 0, "SILVER": 1, "BRONZE": 2, "LOANWORD": 3, "TIER_4_GENERALIZED": 3, "OOV": 4}
EVID_W = {"GOLD": 6, "SILVER": 2, "BRONZE": 1}

# ----------------------------------------------------------------------------- text utils
def nrm(s):
    s = unicodedata.normalize("NFC", s)
    return s.replace("'", "’").replace("‘", "’").replace("`", "’").replace("ʼ", "’")

def strip_tone(s):
    d = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC", "".join(c for c in d if unicodedata.category(c) != "Mn"))

def wkey(w):
    return strip_tone(nrm(w).lower()).strip("“”\"'.,?!;:()[]…—")

PUNCT_RE = re.compile(r"[^\s,?!;:—~()“”\"]+")
def mw_tokens(s):
    s = nrm(s)
    s = re.sub(r"\.\s*$", "", s.strip())
    s = s.replace(". ", " ")
    return PUNCT_RE.findall(s)

def crude_stem(w):
    w = w.lower()
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)]
    return w

EN_STOP = set("a an the is are was were am be been being do does did will shall to of".split())
def en_terms(s):
    ws = re.findall(r"[a-zA-Z’']+", s.lower())
    return [crude_stem(w) for w in ws]

def mw_terms(s):
    return [wkey(w.replace("-", "")) for w in mw_tokens(s) if wkey(w.replace("-", ""))]

# ----------------------------------------------------------------------------- closed-class tables
# Forms below were read off the interlinear glosses of GOLD/SILVER rows (see build_closed_class); every
# form is re-verified at start-up against NON-held-out rows, and forms whose only attestation is in held-out
# clusters are dropped (this is what "hidden from closed-class tables" means operationally).
SUBJ_PRON = {"I": "ɨn", "you": "gha", "he": "ri", "she": "ra", "it": "ni", "we": "mun", "they": "mo"}
SUBJ_PRON_ALT = {"you": ["gha (m.)", "yi (f.)", "wu (pl.)"], "we": ["mun (incl.)", "mumun (excl.)"], "I": ["ɨn", "an"]}
OBJ_PRON = {"me": "ɨn", "you": "gha", "him": "ri", "her": "ra", "it": "ni", "us": "mun", "them": "mo"}
DAT_PRON = {"me": "ng’an", "you": "n’wagha", "him": "n’ri", "her": "n’ra", "it": "n’ni", "us": "m’mun", "them": "m’mo"}
POSS_ATTR = {"my": "fɨna", "your": "fwagha", "his": "fɨri", "her": "fɨra", "its": "kɨni", "our": "funu", "their": "furu"}
POSS_INDEP = {"mine": "mɨna", "his": "mɨri", "ours": "munu"}          # predicative possessors (This book is his)
WH = {"who": "we", "what": "me", "where": "nnee", "when": "pɨrang", "why": "mpèe me", "how": "rang", "how many": "góng"}
CLOSED_MW2EN = {
    "ri": "he", "ra": "she", "mo": "they", "mun": "we(incl)", "mu": "we(incl)", "mumun": "we(excl)", "ɨn": "I", "an": "I",
    "wan": "I", "gha": "you(m)", "wagha": "you(m)", "yi": "you(f)", "wu": "you(pl)", "ni": "DEF", "mɨni": "that",
    "ɗiisɨ": "this", "ɗiisɨ̀": "this", "nɗiisɨ̀": "this", "nkaa": "PROG", "kɨ": "with/and", "’ì": "PERF", "ba": "NEG",
    "kas": "NEG.end", "a": "COP/FOC", "nne": "that(COMP)", "ɓe": "then", "ɗí": "REL", "\u0257i": "REL", "taji": "don’t",
    "’e": "Q", "ye": "Q", "we": "who", "me": "what", "nnee": "where", "pɨrang": "when", "góng": "how many",
    "fɨna": "my", "fɨri": "his", "fɨra": "her", "fii": "your(f)", "fwagha": "your(m)", "funu": "our", "furu": "their",
    "mɨna": "mine", "mɨri": "his(indep)", "munu": "ours", "kɨni": "its", "’á": "HAB", "’í": "HAB", "’é": "HAB",
    "doghon": "yesterday", "koo": "every/or", "mee": "one/a", "ɗáng": "and", "ku": "in order to", "sɨ": "there",
    "sɨ̀": "here",
}
MW_POSS2EN = {"fɨna": "my", "fɨri": "his", "fɨra": "her", "fii": "your", "fwagha": "your", "funu": "our", "furu": "their",
              "kɨni": "its", "fughu": "your(pl)"}
CLOSED_CLASS_FORMS = sorted({wkey(v) for v in list(SUBJ_PRON.values()) + list(POSS_ATTR.values()) + list(POSS_INDEP.values())
                             + list(WH.values()) + ["nkaa", "ba", "kas", "taji", "ni", "mɨni", "ɗiisɨ", "nne", "ɗí", "ɓe"]})

# kinship nouns -> inalienable possession (tagged; Mwaghavul fɨ- series, never the independent mɨ- series)
KINSHIP = {"mother", "father", "brother", "sister", "child", "husband", "wife", "son", "daughter", "parent"}

# Permitted loanwords (English + Hausa).  LOANWORD tier sits directly above OOV.
# Native-speaker correction: direct English loans retain STANDARD ENGLISH ORTHOGRAPHY
# (bus, phone, charge) rather than phonetic respellings (bas, fon, caji).
LOAN_EN2MW = {
    "car": "mota", "automobile": "mota", "vehicle": "mota", "hospital": "asibiti", "clinic": "asibiti",
    "wire": "waya", "phone": "phone", "telephone": "phone", "cellphone": "phone", "mobile": "phone", "smartphone": "phone",
    "charge": "charge", "charger": "charge",
    "bus": "bus", "laptop": "laptop", "table": "tebur", "chair": "kujera",
    "school": "makaranta", "doctor": "likita", "radio": "rediyo", "television": "tibi", "tv": "tibi",
    "computer": "kwamfuta",
}
LOAN_MW2EN = {"mota": "car", "asibiti": "hospital", "waya": "wire/phone", "charge": "charge", "chaaji": "charge", "bus": "bus",
              "fon": "phone", "phone": "phone", "laptop": "laptop", "tebur": "table", "kujera": "chair", "makaranta": "school",
              "likita": "doctor", "rediyo": "radio", "tibi": "television", "kwamfuta": "computer"}

# English nouns whose Mwaghavul lemma is known to the native speaker but has not yet been
# induced into the transfer lexicon.  Bridging here prevents an unwanted NOUN_LOAN fall-through
# (e.g. "girl" -> laareep, not the raw English token "girl").
NOUN_BRIDGE = {"girl": "laareep", "phone": "phone", "bus": "bus", "charge": "charge"}

# Electronic-device context for "charge": the native lexicon entry kurum means "to charge"
# (a person / attack), so an electronic "charge" must use the loanword rather than kurum.
ELEC_CHARGE_OBJECTS = {"phone", "telephone", "cellphone", "mobile", "smartphone", "battery", "charger", "cell", "ipad"}

# Reflexive pronouns -> bare Mwaghavul reflexive (sut series).  Standalone reflexives map
# directly; they NEVER take a redundant possessive pronoun (f…
# English -> Mwaghavul reflexive mapping (native-speaker correction H02).
REFLEX_EN2MW = {"myself": "san", "yourself": "sak", "himself": "shin", "herself": "sat",
                "itself": "sut", "ourselves": "sut", "yourselves": "sut", "themselves": "sut"}

PREP_MAP = {  # English preposition -> (Mwaghavul form, fused?, suppress_ni)
    "at": ("n’", True, True), "in": ("n’", True, True), "to": ("n’", True, True), "into": ("n’", True, True),
    "near": ("n’", True, True), "from": ("a", False, False), "on": ("n’kaa", False, False),
    "inside": ("n’ɗɨghɨn", False, False), "with": ("kɨ", False, False), "for": ("mpee", False, False),
    "without": ("nɗin", False, False), "of": ("’ù", False, False),
}
LOCATIVE_PREPS = {"at", "in", "to", "into", "near"}

# Vehicle / enclosed-space nouns: "on/in the bus" means inside the vehicle, so the
# preposition maps to n’ɗɨghɨn ("inside") rather than n’kaa ("on top of").
VEHICLE_NOUNS = {"bus", "car", "taxi", "train", "plane", "airplane", "lorry", "truck", "ship", "boat", "ferry"}

# Temporal "ago": the GOLD elapsed-time relative clause is
#   puus ɗii-<num> ɗi ni ɗel   ("day REL-number REL it pass")
# e.g. "two days ago" -> puus ɗii-vul ɗi ni ɗel ; "three days ago" -> puus ɗii-kun ɗi ni ɗel
# It always sits at the clause periphery (after O and locative P), never between V and O.
# ----------------------------------------------------------------------------- lexicon
def clean_senses(english):
    """'to eat s.t' -> ['eat'];  'mother, old woman' -> ['mother','old woman'];  drops (..) and s.t/s.o/etc."""
    out = []
    for part in re.split(r"[;,]", english):
        g = re.sub(r"\(.*?\)", "", part).strip().lower()
        g = re.sub(r"\b(s\.t|s\.o|sth|sb|etc)\.?\b", "", g).strip(" .")
        g = re.sub(r"^(to|a|an|the)\s+", "", g).strip()
        if g and g != "’":
            out.append(g)
    return out

class Lexicon:
    def __init__(self, path=DATA / "rbmt_transfer_lexicon.json"):
        d = json.load(open(path, encoding="utf-8"))
        self.entries = d["entries"]
        self.oov_cands = {wkey(o["form"]): o for o in d["oov_candidates"]}
        self.mw = collections.defaultdict(list)
        self.en = collections.defaultdict(list)
        plural_forms = {wkey(e["plural"]) for e in self.entries if e.get("plural")}
        for e in self.entries:
            for k in {wkey(e["lemma"]), wkey(e["lemma_normalized"])}:
                self.mw[k].append(e)
            if e.get("plural"):
                self.mw[wkey(e["plural"])].append(e)
            evid = sum(EVID_W[t] * n for t, n in (e.get("evidence_occurrences") or {}).items())
            e["_evid"] = evid
            pen = (0.2 if " " in e["lemma"].strip() else 1.0) * (0.5 if wkey(e["lemma"]) in plural_forms else 1.0)
            for q in e["equivalents"]:
                for g in clean_senses(q["english"]):
                    score = q["weight"] * (1 + math.log1p(evid)) * pen
                    self.en[g].append((TIER_RANK[q["tier"]], -score, -q["weight"], e, q))
        for g in self.en:
            self.en[g].sort(key=lambda x: (x[0], x[1], x[2]))

    # --- English -> Mwaghavul
    def en_lookup(self, lemma, pos=None):
        posmap = {"NOUN": {"N"}, "PROPN": {"PROPN", "N"}, "VERB": {"V"}, "ADJ": {"ADJ"}, "ADV": {"ADV", "LOC"}, "NUM": {"NUM"},
                  "PRON": {"PRON"}, "AUX": {"AUX", "V"}, "ADP": {"LOC", "PREP"}, "INTJ": {"INTERJ"}, "CCONJ": {"CONJ"},
                  "SCONJ": {"CONJ"}, "DET": {"PART", "PRON"}}
        want = posmap.get(pos)
        c = self.en.get(lemma.lower(), [])
        if want:
            cp = [x for x in c if x[3]["pos"] in want]
            c = cp or []
        return c

    # --- Mwaghavul -> English
    def mw_lookup(self, tok):
        k = wkey(tok)
        return self.mw.get(k, [])

    def mw_candidates(self, tok):
        """[(pos, gloss, tier, weight, entry)] ranked GOLD>SILVER>BRONZE then weight; one best gloss per entry."""
        out = []
        for e in self.mw_lookup(tok):
            eq = sorted(e["equivalents"], key=lambda q: (TIER_RANK[q["tier"]], -q["weight"]))
            if not eq:
                continue
            g = [(clean_senses(q["english"]) or [q["english"]])[0] for q in eq[:3]]
            top = max(e["equivalents"], key=lambda q: q["weight"])      # dominant sense decides the entry's tier rank
            out.append({"pos": e["pos"], "gloss": g[0], "alts": g[1:], "tier": eq[0]["tier"], "rtier": top["tier"], "weight": eq[0]["weight"],
                        "evid": e["_evid"], "entry": e["entry_id"], "lemma": e["lemma"]})
        return out

def pick_candidate(cands, prev=None, nxt=None):
    """Context disambiguation for homographs (N vs V, ...). Preference N after preposition / quantifier / determiner."""
    if not cands:
        return None
    n = [c for c in cands if c["pos"] == "N"]
    v = [c for c in cands if c["pos"] == "V"]
    pk = wkey(prev) if prev else ""
    nk = wkey(nxt) if nxt else ""
    subj_pron = {wkey(x) for x in ("ri", "ra", "mo", "mun", "mu", "mumun", "ɨn", "an", "wan", "gha", "wagha", "yi", "wu")}
    if pk in subj_pron and v:
        return sorted(v, key=lambda c: (TIER_RANK[c["rtier"]], -c["evid"], -c["weight"]))[0]
    ctx_n = pk in {"kaa", "mee", "ni", "n’", "a", "koo", "mɨni", "ɗiisɨ", "ɗí"} or (prev and nrm(prev).lower().startswith(("n’", "m’", "ng’"))) \
        or nk == "ni" or (nxt and nrm(nxt).lower() in ("’ù", "’ɨ̀", "kɨ̀"))
    if n and v and ctx_n:
        return sorted(n, key=lambda c: (TIER_RANK[c["rtier"]], -c["evid"]))[0]
    return sorted(cands, key=lambda c: (TIER_RANK[c["rtier"]], -c["evid"], -c["weight"]))[0]

# ----------------------------------------------------------------------------- grammar
class Grammar:
    def __init__(self, path=DATA / "rbmt_cascading_grammar.json"):
        self.g = json.load(open(path, encoding="utf-8"))
        self.cascade = self.g["lookup_cascade"]
        self.minsup = self.g["resolution_policy"]["min_support"]
        self.mindom = self.g["resolution_policy"]["min_dominant_share"]

    def relation_order(self, rel):
        """Walk GOLD>SILVER>BRONZE>TIER_4 and return (order, tier, n, share)."""
        for t in ("GOLD", "SILVER", "BRONZE"):
            r = self.g[t]["word_order"]["relation_order"].get(rel)
            if r and r.get("n", 0) >= self.minsup[t] and r.get("dominant_share", 0) >= self.mindom:
                return r["dominant_order"], t, r["n"], r["dominant_share"]
        r = self.g["TIER_4_GENERALIZED"]["default_word_order"]["relation_order"].get(rel)
        if r:
            return r["order"], "TIER_4_GENERALIZED", None, r.get("weighted_share")
        return None, "TIER_4_GENERALIZED", None, None

    def valency(self, mw_verb):
        """Best verb frame by tier cascade; returns dict or None."""
        for t in ("GOLD", "SILVER", "BRONZE"):
            v = self.g[t]["valency_frames"]["verbs"].get(mw_verb)
            if v and v["n"] >= self.minsup[t]:
                fr = v["frames"][0]
                if fr["n"] / v["n"] >= self.mindom or len(v["frames"]) == 1:
                    return {"tier": t, "verb": mw_verb, "frame": fr["english_roles"], "mw_order": fr["mw_order"], "n": v["n"]}
                return {"tier": t, "verb": mw_verb, "frame": fr["english_roles"], "mw_order": fr["mw_order"], "n": v["n"],
                        "note": "dominance<0.6 (modal frame shown)"}
        return None

    def tam(self, category):
        r = self.g["resolved_rules"]["tam_by_english_category"].get(category)
        if r:
            return r["top_marker"], r["resolved_from"], r["share"], r["n"]
        t4 = self.g["TIER_4_GENERALIZED"]["default_tam_by_english_verb_category"].get(category)
        if t4:
            return t4["top_markers"][0]["marker"], "TIER_4_GENERALIZED", t4["top_markers"][0]["share"], t4["weighted_support"]
        return "<none>", "TIER_4_GENERALIZED", None, None

    def negation(self):
        neg = self.g["GOLD"]["tam"].get("negation")
        return ("ba … kas", "GOLD", neg)

# ----------------------------------------------------------------------------- RAG retriever
class Retriever:
    def __init__(self, path=DATA / "hybrid_llm_parallel_corpus.jsonl", exclude_ids=frozenset(), exclude_pairs=()):
        self.recs = []
        ex_norm_en, ex_norm_mw = set(), set()
        for p in exclude_pairs:
            ex_norm_en.add(" ".join(en_terms(p["english"])))
            ex_norm_mw.add(" ".join(mw_terms(p["mwaghavul"])))
        ex_sets = [set(en_terms(p["english"])) - set(map(crude_stem, EN_STOP)) for p in exclude_pairs]
        ex_mw_sets = [set(mw_terms(p["mwaghavul"].split("~")[0])) for p in exclude_pairs]
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["id"] in exclude_ids:
                    continue
                en, mw = " ".join(en_terms(r["english"])), " ".join(mw_terms(r["mwaghavul"]))
                if r["unit_type"] != "lexeme" and (en in ex_norm_en or mw in ex_norm_mw):
                    continue
                if exclude_pairs and r["unit_type"] != "lexeme":
                    s = set(en_terms(r["english"])) - set(map(crude_stem, EN_STOP))
                    if len(s) >= 2 and any(len(s & x) / max(1, len(s | x)) >= 0.8 for x in ex_sets):
                        continue
                    m = set(mw_terms(r["mwaghavul"]))
                    if len(m) >= 2 and any(len(m & x) / max(1, len(m | x)) >= 0.75 for x in ex_mw_sets):
                        continue
                self.recs.append(r)
        self.n_excluded_note = None
        self.idx = {}
        for side in ("english", "mwaghavul"):
            df = collections.Counter()
            docs = []
            for r in self.recs:
                t = self._terms(side, r[side])
                docs.append(collections.Counter(t))
                df.update(set(t))
            N = len(self.recs)
            idf = {t: math.log((N + 1) / (c + 0.5)) for t, c in df.items()}
            post = collections.defaultdict(list)
            norms = []
            for i, tf in enumerate(docs):
                w = {t: (1 + math.log(c)) * idf[t] for t, c in tf.items()}
                norms.append(math.sqrt(sum(x * x for x in w.values())) or 1.0)
                for t, x in w.items():
                    post[t].append((i, x))
            self.idx[side] = (idf, post, norms)

    def _terms(self, side, s):
        if side == "english":
            t = en_terms(s)
            return t + [a + "_" + b for a, b in zip(t, t[1:])]
        t = mw_terms(s)
        return t + [a + "_" + b for a, b in zip(t, t[1:])]

    def score(self, side, query):
        idf, post, norms = self.idx[side]
        tf = collections.Counter(self._terms(side, query))
        qw = {t: (1 + math.log(c)) * idf[t] for t, c in tf.items() if t in idf}
        qn = math.sqrt(sum(x * x for x in qw.values())) or 1.0
        sc = collections.defaultdict(float)
        for t, wq in qw.items():
            for i, wd in post[t]:
                sc[i] += wq * wd
        return {i: s / (qn * norms[i]) for i, s in sc.items()}

    @staticmethod
    def skeleton_sim(a, b):
        return difflib.SequenceMatcher(None, a, b).ratio() if a and b else 0.0

    def retrieve(self, src_text, lang, skeleton=None, aux_en=None, k=(3, 2, 2), bronze_min=0.8):
        """lang='en' -> compare source with English side; lang='mw' -> Mwaghavul side (+ RBMT English draft on English side)."""
        side = "english" if lang == "en" else "mwaghavul"
        sc = self.score(side, src_text)
        if lang == "mw" and aux_en:
            for i, s in self.score("english", aux_en).items():
                sc[i] = 0.75 * sc.get(i, 0) + 0.25 * s
        res = {"GOLD": [], "SILVER": [], "BRONZE": []}
        for i, s in sc.items():
            r = self.recs[i]
            t = r["tier"]
            if t == "GOLD" and r["unit_type"] != "sentence":
                continue
            if t == "BRONZE" and r["confidence_score"] < bronze_min:
                continue
            if r["unit_type"] == "sentence" and skeleton:
                s = 0.8 * s + 0.2 * self.skeleton_sim(skeleton, (r.get("skeleton_abstract") or "").split())
            s *= {"GOLD": 1.0, "SILVER": 0.9, "BRONZE": 0.8}[t]       # tier weight
            res[t].append((s, r))
        out = []
        for t, n in zip(("GOLD", "SILVER", "BRONZE"), k):
            for s, r in sorted(res[t], key=lambda x: -x[0])[:n]:
                out.append({"tier": t, "score": round(s, 3), "id": r["id"], "mwaghavul": r["mwaghavul"], "english": r["english"],
                            "ref": r["source"].get("ref")})
        return out

# ----------------------------------------------------------------------------- English analysis + EN->MW
CLEFT_RE = re.compile(r"^\s*it\s+(?:is|was|’s|'s)\s+(.+?)\s+(that|who|whom|which)\s+(.+?)\s*[.?!]*\s*$", re.I)
WH_SET = {"who", "what", "where", "when", "why", "how"}

def _verb_obj_text(v, ch):
    """Direct object text of the verb v (for electronic-device 'charge' disambiguation)."""
    for c in ch:
        if c.dep_ == "dobj" and c.lower_ not in WH_SET:
            return c.text
    return ""

class Seg(list):
    pass

class EnglishToMw:
    def __init__(self, lex, gram, nlp):
        self.lex, self.gram, self.nlp = lex, gram, nlp

    # ---- bookkeeping
    def reset(self):
        self.rules, self.flags, self.lex_trace = [], [], []

    def rule(self, name, tier, detail=""):
        self.rules.append({"rule": name, "tier": tier, "detail": detail})

    def flag(self, kind, text):
        self.flags.append(f"{kind}:{text}")

    # ---- lexical lookup with LOANWORD / NOUN_BRIDGE / NOUN_LOAN / OOV tiers
    def lex_en(self, tok, pos=None, obj_context=None):
        lemma = tok.lemma_.lower() if tok.pos_ != "PROPN" else tok.text
        pos = pos or tok.pos_
        if pos == "PROPN":
            self.lex_trace.append((tok.text, "PROPN", "retain"))
            return tok.text
        # electronic-device "charge" -> loanword (native kurum = to charge a person)
        if lemma == "charge" and obj_context and wkey(obj_context) in ELEC_CHARGE_OBJECTS:
            self.flag("LOANWORD", f"{lemma}→{LOAN_EN2MW[lemma]} (electronic-device context)")
            self.lex_trace.append((lemma, "LOANWORD", LOAN_EN2MW[lemma]))
            return LOAN_EN2MW[lemma]
        c = self.lex.en_lookup(lemma, pos)
        if c and lemma in LOAN_EN2MW and tok.pos_ == "NOUN" and not any(x[0] == 0 for x in c):
            pass
        if c:
            best = c[0]
            e, q = best[3], best[4]
            alts = [x[3]["lemma"] for x in c[1:3]]
            self.lex_trace.append((lemma, q["tier"], e["lemma"] + (f" (alts: {', '.join(alts)})" if alts else "")))
            return e["lemma"]
        # native-speaker-verified noun bridge: avoid raw English token fall-through
        if lemma in NOUN_BRIDGE:
            self.flag("NOUN_BRIDGE", f"{lemma}→{NOUN_BRIDGE[lemma]}")
            self.lex_trace.append((lemma, "NOUN_BRIDGE", NOUN_BRIDGE[lemma]))
            return NOUN_BRIDGE[lemma]
        if lemma in LOAN_EN2MW:
            self.flag("LOANWORD", f"{lemma}→{LOAN_EN2MW[lemma]}")
            self.lex_trace.append((lemma, "LOANWORD", LOAN_EN2MW[lemma]))
            return LOAN_EN2MW[lemma]
        if tok.pos_ in ("NOUN",):
            self.flag("NOUN_LOAN", lemma)
            self.lex_trace.append((lemma, "NOUN_LOAN", lemma))
            return lemma
        self.flag("OOV", lemma)
        self.lex_trace.append((lemma, "OOV", "[" + lemma + "]"))
        return "[" + lemma + "]"

    # ---- entry point
    def translate(self, text):
        self.reset()
        doc = self.nlp(text.strip())
        self.doc = doc
        qmark = text.strip().endswith("?")
        m = CLEFT_RE.match(text)
        toks = None
        root = [t for t in doc if t.dep_ == "ROOT"][0]
        if m and doc[0].lower_ == "it":
            toks = self.cleft(doc, m)
        else:
            toks = self.clause(root)
        sent = self.assemble(toks)
        if qmark and not sent.endswith("?"):
            sent += "?"
        elif not sent.endswith(("?", "!", ".")):
            sent += "."
        return sent

    def assemble(self, toks):
        out = ""
        for t in toks:
            if not t:
                continue
            if out and not out.endswith("\u2060"):
                out += " "
            out += t
        out = out.replace("\u2060", "")
        out = re.sub(r"\s+([?!.,])", r"\1", out).strip()
        return out[:1].upper() + out[1:]

    FUSE = "\u2060"
    def fuse(self, clitic, word):
        return clitic + word  # n’ + noun written together (n’lutuk)

    # ---- cleft:  It is/was [focus] that/who [clause]
    def cleft(self, doc, m):
        rel_idx = None
        for t in doc:
            if t.lower_ in ("that", "who", "whom", "which") and t.idx >= m.start(2) - 1 and t.idx == m.start(2):
                rel_idx = t.i
        rel = doc[rel_idx]
        focus_toks = [t for t in doc if m.start(1) <= t.idx < m.end(1)]
        clause_toks = [t for t in doc if t.i > rel_idx]
        # clause head verb
        v = rel.head if rel.dep_ in ("nsubj", "nsubjpass", "dobj", "advmod") and rel.head.pos_ in ("VERB", "AUX") else None
        if v is None:
            vs = [t for t in clause_toks if t.pos_ == "VERB"]
            v = vs[0] if vs else (clause_toks[0] if clause_toks else rel)
        has_subj = any(c.dep_ in ("nsubj", "nsubjpass") and c.i > rel_idx for c in v.children)
        self.rule("it-cleft: strip dummy 'it', front copula 'a' (FOC), no mɨni", "GOLD",
                  "FOC a + focus + clause" + ("" if has_subj else " (subject-focus: clause subject gap)"))
        # focus head = token in focus span whose head is outside the span
        fset = {t.i for t in focus_toks}
        fh = [t for t in focus_toks if t.head.i not in fset] or focus_toks[:1]
        fh = fh[0]
        if fh.dep_ in ("prep",) or fh.pos_ == "ADP":
            f = self.pp(fh, restrict=fset)
        elif fh.pos_ == "ADV":
            f = [self.lex_en(fh, "ADV")]
        else:
            f = self.np(fh, restrict=fset, skip_relcl=True)
        toks = ["a"] + f
        toks += self.clause(v, drop_subj=not has_subj, skip={rel_idx})
        return toks

    # ---- NP
    def np(self, h, restrict=None, loc_governed=False, as_object=False, skip_relcl=False, obj_context=None):
        kids = lambda dep: [c for c in h.children if c.dep_ in dep and (restrict is None or c.i in restrict or True)]
        if h.pos_ == "PRON":
            lem = h.lower_
            if lem in WH_SET:
                return [WH[lem]]
            tab = OBJ_PRON if (h.dep_ in ("dobj", "pobj", "dative", "iobj") or as_object) else SUBJ_PRON
            key = "I" if h.text == "I" else lem
            if key == "you" or key == "we":
                self.flag("AMBIG", f"{key}→{'|'.join(SUBJ_PRON_ALT[key])}")
            self.lex_trace.append((h.text, "CLOSED", tab.get(key, key)))
            return [tab.get(key, key)]
        if h.pos_ == "DET" and h.lower_ in WH_SET | {"what", "this", "that"}:
            return [{"this": "ɗiisɨ̀", "that": "mɨni", "what": "me"}.get(h.lower_, h.text)]
        word = self.lex_en(h, obj_context=obj_context)
        kin = h.lemma_.lower() in KINSHIP
        if kin:
            self.flag("KINSHIP", h.lemma_.lower())
        out = [word]
        pl = h.tag_ in ("NNS", "NNPS")
        # compounds (modifier before head, TIER ruling)
        comp = [c for c in h.children if c.dep_ == "compound"]
        order, tier, n, sh = self.gram.relation_order("compound")
        if comp:
            self.rule("compound", tier, f"{order} (n={n}, share={sh})")
            cw = [self.lex_en(c) for c in comp]
            out = cw + out if order == "DEP_BEFORE_HEAD" else out + cw
        # possessives
        for c in [c for c in h.children if c.dep_ == "poss"]:
            order, tier, n, sh = self.gram.relation_order("poss")
            if c.tag_ == "PRP$":
                p = POSS_ATTR[c.lower_]
                self.rule("possessive pronoun (attributive fɨ- series)" + (" [KINSHIP inalienable]" if kin else ""), "GOLD",
                          f"{c.lower_}→{p} (poss: {order}); never independent mɨ- series")
                out = out + [p]
            else:
                # Noun's Noun2 -> HEAD_BEFORE_DEP : [Noun2] GEN [Noun]
                possr = self.np(c)
                gen = "’ù" if re.search(r"[uo]$", strip_tone(word.lower())) else "’ɨ̀"
                self.rule("Noun's Noun2 → HEAD_BEFORE_DEP (head + GEN + possessor)", tier, f"poss {order} (n={n}); GEN allomorph guess {gen}")
                self.flag("GEN_ALLOMORPH_GUESS", gen)
                out = out + [gen] + possr
        # adjectives (postnominal)
        for c in [c for c in h.children if c.dep_ == "amod"]:
            order, tier, n, sh = self.gram.relation_order("amod")
            self.rule("amod", tier, f"{order or 'HEAD_BEFORE_DEP (postnominal, elicitation ngu jeel ni)'}")
            out = out + [self.lex_en(c, "ADJ")]
        for c in [c for c in h.children if c.dep_ == "nummod"]:
            out = out + [self.lex_en(c, "NUM")]
        # determiners
        defn = False
        for c in [c for c in h.children if c.dep_ == "det"]:
            lw = c.lower_
            if lw == "the":
                defn = True
            elif lw in ("this", "these"):
                out = out + ["ɗiisɨ̀"]
            elif lw in ("that", "those"):
                out = out + ["mɨni"]
            elif lw == "every" or lw == "each":
                out = ["koo", "mee"] + out
            elif lw in ("a", "an"):
                pass
            elif lw in WH_SET or lw == "which":
                out = out + ["me"]
            else:
                out = out + [self.lex_en(c, "DET")] if self.lex.en_lookup(lw) else out
        if pl:
            out = out + ["mo"]
            self.rule("plural", "GOLD", "N + mo (PL particle)")
        # relative clauses
        for c in [c for c in h.children if c.dep_ == "relcl" and not skip_relcl]:
            rel_sub = any(d.dep_ in ("nsubj", "nsubjpass") and d.tag_ in ("WDT", "WP") for d in c.children)
            pr = {"M": "ri", "F": "ra"}.get("M" if h.lower_ in ("man", "boy", "father", "brother", "husband") else "F" if h.lower_ in ("woman", "girl", "mother", "sister", "wife") else None, "ri")
            order, tier, n, sh = self.gram.relation_order("relcl")
            self.rule("relative clause: N ɗí [resumptive pronoun] clause (ni closes the NP)", tier, f"relcl {order} (n={n})")
            body = self.clause(c, drop_subj=rel_sub, skip={d.i for d in c.children if d.tag_ in ("WDT", "WP")})
            out = out + ["ɗí"] + ([pr] if rel_sub else []) + body
            defn = defn or True
        # definiteness + locative suppression
        if defn and not [c for c in h.children if c.dep_ == "poss"]:
            if loc_governed:
                self.rule("locative definiteness: suppress trailing ni after n’", "GOLD", f"{word}: no ni")
            else:
                out = out + ["ni"]
        return out

    def pp(self, p, restrict=None):
        prep = p.lower_
        objs = [c for c in p.children if c.dep_ in ("pobj", "dobj")]
        form, fused, supp = PREP_MAP.get(prep, (None, False, False))
        # Vehicle / enclosed-space container: "on/in the bus" -> n’ɗɨghɨn ("inside")
        # rather than n’kaa ("on top of").
        if form is not None and prep in ("on", "in") and objs:
            obj_text = " ".join(c.lower_ for c in objs)
            if any(w in obj_text for w in VEHICLE_NOUNS):
                form, fused, supp = "n\u2019\u0257\u0268gh\u0268n", False, False
                self.rule("vehicle container: on/in the <vehicle> -> n’ɗɨghɨn", "GOLD",
                          f"{obj_text} -> inside")
        if form is None:
            lw = self.lex_en(p, "ADP")
            form, fused, supp = lw, False, False
        if not objs:
            return [form]
        o = objs[0]
        loc = prep in LOCATIVE_PREPS
        body = self.np(o, loc_governed=(loc and supp))
        order, tier, n, sh = self.gram.relation_order("prep")
        self.rule(f"preposition '{prep}' → {form}", tier, f"prep {order} (n={n}, share={sh})" + (" fused clitic" if fused else ""))
        if fused:
            body[0] = form + body[0]
            return body
        return [form] + body

    # ---- clause
    def clause(self, v, drop_subj=False, skip=frozenset()):
        ch = [c for c in v.children if c.i not in skip]
        neg = any(c.dep_ == "neg" for c in ch)
        auxs = [c for c in ch if c.dep_ in ("aux", "auxpass")]
        al = [a.lemma_.lower() for a in auxs]
        subj = [c for c in ch if c.dep_ in ("nsubj", "nsubjpass")]
        is_cop = v.lemma_ == "be" and v.dep_ in ("ROOT", "relcl", "ccomp", "advcl", "conj")
        # ---- TAM category (English side) and Mwaghavul marker from the cascade
        cat = "simple_past" if v.tag_ == "VBD" else "simple_present"
        if "will" in al or "shall" in al or "’ll" in [a.text for a in auxs]:
            cat = "future"
        elif "have" in al and v.tag_ == "VBN":
            cat = "perfect"
        elif "be" in al and v.tag_ == "VBG":
            cat = "progressive"
        elif "do" in al:
            cat = "do-support" if "will" not in al else cat
        elif any(a.tag_ == "MD" for a in auxs):
            cat = "modal"
        elif any(c.dep_ == "auxpass" for c in ch):
            cat = "passive"
        elif v.tag_ == "VB" and not subj:
            cat = "base/infinitive/imperative"
        marker, mt, sh, n = self.gram.tam(cat)
        pre, fut = None, False
        if cat == "progressive":
            pre = "nkaa" if any(a.tag_ in ("VBZ", "VBP") for a in auxs) or True else "kɨ"
            if any(a.tag_ == "VBD" for a in auxs):
                pre = "kɨ"
        elif cat == "perfect":
            pre = "’ì"
        elif cat == "future":
            fut = True
        elif cat == "modal":
            self.flag("MODAL_UNMAPPED", "/".join(a.lemma_ for a in auxs))
        self.rule(f"TAM: English {cat} → {pre or ('clitic n’' if fut else '∅')}", mt, f"top_marker={marker} (share={sh}, n={n})")
        # ---- wh handling
        wh = [c for c in ch if c.lower_ in WH_SET and c.tag_ in ("WP", "WRB", "WDT", "WP$")]
        wh_subj = [c for c in wh if c.dep_ in ("nsubj", "nsubjpass")]
        wh_obj = [c for c in wh if c.dep_ in ("dobj", "pobj", "attr")]
        wh_adv = [c for c in wh if c.dep_ == "advmod"]
        # ---- subject tokens
        S = []
        if not drop_subj:
            for s in subj:
                if s.lower_ in WH_SET:
                    continue
                S += self.np(s)
        # ---- verb
        V = []
        if is_cop:
            pred = [c for c in ch if c.dep_ in ("attr", "acomp")]
            preps = [c for c in ch if c.dep_ == "prep"]
            adv_loc = [c for c in ch if c.dep_ == "advmod" and c.lower_ in ("here", "there")]
            toks = []
            if pred and pred[0].pos_ == "ADJ":
                V = [self.lex_en(pred[0], "ADJ")]
                self.rule("predicate adjective: stative verb, no copula", "GOLD", "S ADJ")
            elif pred:
                V = ["a"] + self.np(pred[0])
                self.rule("predicate nominal: S a NP (COP a)", "GOLD", "")
            elif preps:
                V = ["a"] + self.pp(preps[0])
                self.rule("locative predicate: S a n’+NP (COP a)", "GOLD", "")
            elif wh_adv and wh_adv[0].lower_ == "where":
                V = ["a", WH["where"]]
            else:
                V = ["a"]
            verb_tok = None
        else:
            vw = self.lex_en(v, "VERB", obj_context=_verb_obj_text(v, ch)) if v.pos_ in ("VERB", "AUX") else self.lex_en(v)
            if fut:
                vw = "n’" + vw
            V = ([pre] if pre else []) + [vw]
            val = self.gram.valency(vw.lstrip("n’"))
            if val:
                self.rule(f"valency frame for '{val['verb']}': {val['frame']} → {val['mw_order']}", val["tier"], f"n={val['n']}")
            else:
                cls_ = self.gram.g["TIER_4_GENERALIZED"]["default_argument_structures"][0]
                self.rule("valency (no verb-specific frame): default arg structure", "TIER_4_GENERALIZED", f"{cls_['english_roles']} → {cls_['default_mw_order']}")
        # ---- objects / datives / obliques / adverbs
        O = []
        dat = [c for c in ch if c.dep_ in ("dative", "iobj")]
        dobj = [c for c in ch if c.dep_ == "dobj" and c.lower_ not in WH_SET]
        for d in dobj:
            # Standalone reflexive pronoun -> bare Mwaghavul reflexive (sut series).
            # sut already encodes "themselves"; it NEVER takes a redundant possessive
            # pronoun (f…
            if d.lower_ in REFLEX_EN2MW:
                self.rule("reflexive: standalone -self/-selves → bare sut reflexive (no possessive)", "GOLD",
                          f"{d.lower_}→{REFLEX_EN2MW[d.lower_]}")
                self.lex_trace.append((d.lower_, "REFLEX", REFLEX_EN2MW[d.lower_]))
                O += [REFLEX_EN2MW[d.lower_]]
                continue
            O += self.np(d, as_object=True, obj_context=d.text)
        for d in dat:
            if d.pos_ == "PRON":
                O += [DAT_PRON.get(d.lower_, "n’" + d.lower_)]
                self.rule("dative recipient: n’+pronoun after O", "GOLD", f"{d.lower_}→{DAT_PRON.get(d.lower_)}")
            else:
                O += ["n’" + self.np(d)[0]] + self.np(d)[1:]
        for w in wh_obj:
            if not is_cop:
                O += [WH[w.lower_]]
        P = []
        for c in ch:
            if c.dep_ == "prep" and not (is_cop and c is [x for x in ch if x.dep_ == "prep"][0] and not [x for x in ch if x.dep_ in ("attr", "acomp")]):
                P += self.pp(c)
            elif c.dep_ == "agent":
                P += self.pp(c)
            elif c.dep_ in ("npadvmod", "advmod") and c.lower_ not in WH_SET | {"not", "n't", "here", "there", "ago"}:
                # Standalone reflexive as a clause-level adverbial ("built the house themselves")
                # -> bare sut reflexive, never a redundant possessive.
                if c.lower_ in REFLEX_EN2MW:
                    self.rule("reflexive: standalone -self/-selves → bare sut reflexive (no possessive)", "GOLD",
                              f"{c.lower_}→{REFLEX_EN2MW[c.lower_]}")
                    self.lex_trace.append((c.lower_, "REFLEX", REFLEX_EN2MW[c.lower_]))
                    P += [REFLEX_EN2MW[c.lower_]]
                else:
                    P += [self.lex_en(c, "ADV")]
            elif c.dep_ in ("xcomp", "ccomp", "advcl", "conj", "cc"):
                if c.dep_ == "cc":
                    P += ["kɨ" if c.lower_ == "and" else self.lex_en(c, "CCONJ")]
                else:
                    P += self.clause(c)
            elif c.dep_ == "oprd":
                P += self.np(c)
        # ---- temporal "ago": elapsed-time frame.  "V N days ago" -> "a m'puus <days> ni" placed
        # immediately before V (matching the GOLD structure … so a m’puus <days> ni <verb>).
        ago = [c for c in ch if c.lower_ == "ago" and c.dep_ == "advmod"]
        TEMP = []
        if ago and not is_cop:
            # GOLD elapsed-time relative clause: puus ɗii-<num> ɗi ni ɗel
            # ("day REL-number REL it pass").  Always at the clause periphery.
            days_toks = []
            for noun in [x for x in ago[0].children if x.pos_ in ("NOUN", "NUM", "PROPN")]:
                for num in [y for y in noun.children if y.dep_ == "nummod" and y.pos_ == "NUM"]:
                    days_toks += self.np(num, as_object=True)
                if not days_toks and noun.pos_ == "NUM":
                    days_toks += self.np(noun, as_object=True)
            self.rule("temporal ago: puus ɗii-<num> ɗi ni ɗel (elapsed-time relative clause)", "GOLD",
                      "… puus ɗii-<num> ɗi ni ɗel")
            TEMP = ["puus", "\u0257ii-" + (days_toks[0] if days_toks else "vul"), "\u0257i", "ni", "\u0257el"]
        # ---- assemble.  Mwaghavul is strictly SVO: S + V + O + locative P + temporal P.
        # Never move the main verb after the direct object.
        toks = []
        if neg and not subj and cat == "base/infinitive/imperative":
            toks += ["taji", "a"] + V + O + P + TEMP + ["kas"]
            self.rule("negative imperative: taji a V … kas", "GOLD", "PROH")
            return toks
        if neg:
            toks += ["ba"]
            self.rule("negation: ba … kas", "GOLD", "clause-final kas")
        if wh_subj:
            toks += ["a", WH[wh_subj[0].lower_]] + V + O + P + TEMP
            toks += ["ye" if wh_subj[0].lower_ == "who" else "ʼ e"]
            self.rule("subject wh-question: FOC a WH V … Q", "GOLD", f"{wh_subj[0].lower_}")
        elif wh_adv and not is_cop:
            toks += S + V + O + P + TEMP + ["a", WH[wh_adv[0].lower_], "ʼ e"]
            self.rule("adverbial wh: S V … a WH ʼ e", "GOLD", wh_adv[0].lower_)
        elif wh_obj:
            toks += S + V + O + P + TEMP + ["ʼ e"]
            self.rule("object wh: S V WH ʼ e (neutral)", "GOLD", "")
        elif is_cop and wh_adv:
            toks += S + ["a", WH[wh_adv[0].lower_], "ʼ e"]
            self.rule("copular wh: S a WH ʼ e", "GOLD", wh_adv[0].lower_)
        else:
            toks += S + V + O + P + TEMP
        if neg:
            toks += ["kas"]
        return toks
# ----------------------------------------------------------------------------- Mwaghavul -> English
class MwToEnglish:
    def __init__(self, lex, gram):
        self.lex, self.gram = lex, gram

    def translate(self, text):
        self.rules, self.flags = [], []
        toks = mw_tokens(text)
        n = len(toks)
        glosses = []     # list of dicts: {surf, gloss, pos, tier, kind}
        for i, t in enumerate(toks):
            prev = toks[i - 1] if i else None
            nxt = toks[i + 1] if i + 1 < n else None
            glosses.append(self.gloss_token(t, prev, nxt, i))
        # ---- reordering rules
        out = self.reorder(glosses)
        return out, glosses

    def gloss_token(self, t, prev, nxt, i):
        k = wkey(t)
        tn = nrm(t)
        # fused clitic n’/m’/ng’ + stem
        mpre = re.match(r"^(n’|m’|ng’|ng’)(.+)$", tn.lower())
        if tn.lower() in ("’ù", "’ɨ̀", "kɨ̀"):
            return {"surf": t, "gloss": "’s(GEN)", "pos": "CL", "tier": "GOLD", "kind": "GEN"}
        if k in CLOSED_MW2EN and not (k == "a" and False):
            kind = "CLOSED"
            if k == "a":
                return {"surf": t, "gloss": "FOC" if i == 0 else "is", "pos": "PART", "tier": "GOLD", "kind": "CLOSED"}
            return {"surf": t, "gloss": CLOSED_MW2EN[k], "pos": "CL", "tier": "GOLD", "kind": kind}
        if k in LOAN_MW2EN:
            return {"surf": t, "gloss": LOAN_MW2EN[k], "pos": "N", "tier": "LOANWORD", "kind": "LOANWORD"}
        if mpre:
            stem = mpre.group(2)
            cands = self.lex.mw_candidates(stem)
            if cands or wkey(stem) in LOAN_MW2EN or wkey(stem) in CLOSED_MW2EN:
                locs = [x for x in cands if x["pos"] in ("LOC", "ADV", "PREP")]
                c = (sorted(locs, key=lambda x: (TIER_RANK[x["rtier"]], -x["evid"]))[0] if locs else pick_candidate(cands, "n’", nxt)) if cands else None
                g = c["gloss"] if c else LOAN_MW2EN.get(wkey(stem), CLOSED_MW2EN.get(wkey(stem)))
                tier = c["tier"] if c else "LOANWORD"
                if c and c["pos"] == "V":
                    return {"surf": t, "gloss": g, "pos": "V", "tier": tier, "kind": "FUT", "alts": c["alts"]}
                return {"surf": t, "gloss": g, "pos": "N", "tier": tier, "kind": "LOC", "alts": (c or {}).get("alts", [])}
        cands = self.lex.mw_candidates(t)
        if cands:
            c = pick_candidate(cands, prev, nxt)
            return {"surf": t, "gloss": c["gloss"], "pos": c["pos"], "tier": c["tier"], "kind": "LEX", "alts": c["alts"],
                    "homographs": [f"{x['pos']}:{x['gloss']}" for x in cands]}
        o = self.lex.oov_cands.get(k)
        if o and o.get("aligned_english"):
            return {"surf": t, "gloss": o["aligned_english"][0]["english"], "pos": o.get("pos_guess") or "?", "tier": "BRONZE",
                    "kind": "OOVCAND"}
        if t[:1].isupper() and i > 0:
            return {"surf": t, "gloss": t, "pos": "PROPN", "tier": "LOANWORD", "kind": "PROPN"}
        return {"surf": t, "gloss": "[" + t + "]", "pos": "?", "tier": "OOV", "kind": "OOV"}

    GEN = {"’ù", "’ɨ̀", "kɨ̀", "’ɨ", "ù"}
    def is_np_head(self, g):
        return (g["kind"] == "LEX" and g["pos"] in ("N", "PROPN", "NUM")) or g["kind"] in ("LOC", "LOANWORD", "PROPN")

    def parse_np(self, gl, i):
        """N [ADJ|POSS|DEM|DEF]* [GEN N-phrase]  ->  (english words, next index)"""
        n = len(gl)
        g = gl[i]
        head, mods, j, defn, poss, dem = g["gloss"], [], i + 1, False, None, None
        while j < n:
            h = gl[j]; kk = wkey(h["surf"])
            if h["kind"] == "LEX" and h["pos"] == "ADJ":
                mods.append(h["gloss"]); j += 1
            elif kk in MW_POSS2EN:
                poss = MW_POSS2EN[kk]; j += 1
            elif kk == "mɨni" or kk.endswith("ɗiisɨ"):
                dem = "that" if kk == "mɨni" else "this"; j += 1
            elif kk == "ni":
                defn = True; j += 1
            else:
                break
        pre = ([poss] if poss else []) + ([dem] if dem else []) + (["the"] if defn and not poss and not dem else [])
        if g["kind"] == "LOC":
            pre = ["at/in/to"] + pre
        words = pre + mods + [head]
        if j + 1 < n and g["kind"] != "LOC" and nrm(gl[j]["surf"]).lower() in self.GEN and (self.is_np_head(gl[j + 1]) or gl[j + 1]["pos"] == "PRON"):
            pw, j2 = self.parse_np(gl, j + 1)
            self.rules.append({"rule": "Noun GEN Noun2 → Noun2’s Noun (HEAD_BEFORE_DEP reversed)", "tier": "GOLD",
                               "detail": f"{g['surf']} {gl[j]['surf']} {gl[j+1]['surf']}"})
            words = pw + ["’s"] + words
            j = j2
        if j > i + 1:
            self.rules.append({"rule": "NP reorder: N ADJ/DEM/POSS/DEF → DET/POSS ADJ N", "tier": "GOLD",
                               "detail": " ".join(x["surf"] for x in gl[i:j])})
        return words, j

    def reorder(self, gl):
        """Mwaghavul order -> English-ordered gloss: NP reorder, GEN reversal, PROG/FUT/PERF/NEG linearised as glossed English."""
        out, i, n = [], 0, len(gl)
        while i < n:
            g = gl[i]
            if self.is_np_head(g):
                w, i = self.parse_np(gl, i); out += w; continue
            kk = wkey(g["surf"])
            if g["kind"] == "FUT":
                out += ["will", g["gloss"]]; i += 1; continue
            if kk == "ba":
                out += ["not"]; i += 1; continue
            if kk == "kas":
                i += 1; continue
            if kk == "nkaa":
                out += ["is/are"]; i += 1
                if i < n and not self.is_np_head(gl[i]):
                    out += [gl[i]["gloss"] + "-ing"]; i += 1
                continue
            if g["surf"] in ("’ì", "’ò", "’à"):
                out += ["has/have"]; i += 1; continue
            if g["gloss"] == "DEF":
                out += ["it"]; i += 1; continue
            out.append(g["gloss"]); i += 1
        return " ".join(out)

# ----------------------------------------------------------------------------- engine facade
class Engine:
    _nlp = None
    def __init__(self, exclude_ids=frozenset(), exclude_pairs=(), need_spacy=True):
        self.lex = Lexicon()
        self.gram = Grammar()
        self.ret = Retriever(exclude_ids=exclude_ids, exclude_pairs=exclude_pairs)
        if need_spacy:
            if Engine._nlp is None:
                import spacy
                Engine._nlp = spacy.load("en_core_web_sm")
            self.nlp = Engine._nlp
        self.e2m = EnglishToMw(self.lex, self.gram, getattr(self, "nlp", None))
        self.m2e = MwToEnglish(self.lex, self.gram)

    @staticmethod
    def detect_lang(s):
        if re.search(r"[ɨɓɗŋ\u0300\u0301\u0304\u0302\u030c]", unicodedata.normalize("NFD", s)):
            return "mw"
        ws = re.findall(r"[A-Za-z’']+", s.lower())
        common = {"the", "is", "are", "was", "were", "he", "she", "it", "we", "they", "i", "you", "a", "an", "to", "of", "and", "in",
                  "at", "my", "his", "her", "not", "did", "will", "who", "what", "where", "this", "that", "with", "for", "has", "have"}
        return "en" if sum(w in common for w in ws) >= max(1, len(ws) // 4) else "mw"

    def skeleton_en(self, doc):
        m = {"PRON": "PRON", "AUX": "AUX", "VERB": "V", "NOUN": "NP", "PROPN": "NP", "DET": "NP", "ADJ": "NP", "NUM": "NP",
             "ADV": "ADV", "ADP": "CL", "CCONJ": "CONJ", "SCONJ": "CONJ", "PART": "PART"}
        seq = []
        for t in doc:
            if t.is_punct:
                continue
            s = m.get(t.pos_, t.pos_)
            if s == "NP" and seq and seq[-1] == "NP":
                continue
            seq.append(s)
        return seq

    def run(self, text, lang=None):
        lang = lang or self.detect_lang(text)
        pk = {"input": text, "lang": lang}
        if lang == "en":
            draft = self.e2m.translate(text)
            doc = self.e2m.doc
            pk.update(direction="EN→MW", draft=draft, flags=self.e2m.flags, rules=self.e2m.rules, lex_trace=self.e2m.lex_trace)
            pk["analysis"] = [(t.text, t.lemma_, t.pos_, t.tag_, t.dep_, t.head.text) for t in doc]
            pk["examples"] = self.ret.retrieve(text, "en", skeleton=self.skeleton_en(doc))
        else:
            draft, glosses = self.m2e.translate(text)
            fl = [f"{g['tier']}:{g['surf']}→{g['gloss']}" for g in glosses if g["tier"] in ("LOANWORD", "OOV", "BRONZE") and g["kind"] != "CLOSED"]
            pk.update(direction="MW→EN", draft=draft, flags=fl, rules=self.m2e.rules,
                      lex_trace=[(g["surf"], g["tier"], g["gloss"] + (f"  [homographs: {', '.join(g['homographs'])}]" if g.get("homographs") and len(g["homographs"]) > 1 else "")) for g in glosses])
            pk["examples"] = self.ret.retrieve(text, "mw", aux_en=draft)
        return pk

def show(pk):
    print(f"\n=== {pk['direction']}  |  {pk['input']}")
    print(f"RBMT symbolic draft : {pk['draft']}")
    print(f"OOV/Loanword flags  : {pk['flags'] or '—'}")
    print("Lexical lookup      :")
    for a, t, b in pk["lex_trace"]:
        print(f"   [{t:8s}] {a} → {b}")
    print("Retrieved rules     :")
    seen = set()
    for r in pk["rules"]:
        key = (r["rule"], r["tier"])
        if key in seen:
            continue
        seen.add(key)
        print(f"   [{r['tier']}] {r['rule']}  {r['detail']}")
    print("RAG examples        :")
    for e in pk["examples"]:
        print(f"   [{e['tier']:6s} {e['score']:.2f}] {e['mwaghavul']}  ||  {e['english']}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sentence")
    ap.add_argument("--lang", choices=["en", "mw"])
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    eng = Engine()
    pk = eng.run(a.sentence, a.lang)
    print(json.dumps(pk, ensure_ascii=False, indent=1) if a.json else "", end="")
    if not a.json:
        show(pk)
