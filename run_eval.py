#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_eval.py  --  truly blind evaluation harness for test_translator.py

Stages (run in this order; references stay sealed until stage 'score'):
    python run_eval.py select    cluster -> pick held-out sentences -> write blind_inputs.json (sources only) + sealed refs (hash printed)
    python run_eval.py rbmt      build the masked engine, run RBMT + tier-weighted RAG on the 15 inputs -> rbmt_packets.json
    python run_eval.py score     load zero_shot.json + hybrid.json + sealed refs -> sacrebleu chrF++ / BLEU table
"""
import collections, hashlib, json, random, re, sys, unicodedata
from pathlib import Path
import test_translator as T

WORK = Path("/home/claude/eval"); WORK.mkdir(exist_ok=True)
SEAL = WORK / "sealed_refs.json"
SEED = 20261113

# Rows I (the developer-assistant) printed while exploring the data BEFORE the split was drawn: their clusters are
# ineligible for hold-out so that nothing I read while writing rules can end up in the test set.
SEEN_DURING_DEV = set("""1 2 3 4 5 6 23a 24 26 27 28 35 42a 79 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111
116 117 118 126 127 129a 129b 133 134.1 134.2 135 136 137 138.1 138.2 139 145 148 149 153 163 183 185 192 301 303 304 307 308 309
311 312 313 315 319 322 324 325 326 329 357 358 381 384 398 399 485 486 523 S5 S6 S8 S11 S38 S39""".split())

MODERN = [   # brand-new sentences, never run through the engine before the blind protocol started
    "It was her father who repaired the radio.",
    "The driver's bus is at the hospital.",
    "Where did your sister buy the television?",
    "His wife is not at the market.",
    "The girl is charging a phone at the school.",
]

PRON_EN = set("i you he she it we they me him her us them my your his our their is are was were am be been being will shall would "
              "did do does has have had not the a an this that these those yes no".split())
PRON_MW = {"ri", "ra", "mo", "mun", "mu", "mumun", "ɨn", "an", "wan", "gha", "wagha", "yi", "wu", "ni", "nkaa", "pu", "kɨ", "’ì", "’ò", "ba", "kas",
           "a", "fɨna", "fɨri", "fɨra", "fii", "fwagha", "funu", "furu", "wuri", "wura", "ɗa", "’á", "’í", "’é", "ɓe"}

def load_gold_elicitation():
    rows = []
    with open(T.DATA / "hybrid_llm_parallel_corpus.jsonl", encoding="utf-8") as f:
        for l in f:
            r = json.loads(l)
            if r["tier"] == "GOLD" and r["unit_type"] == "sentence" and r["source"]["corpus"] == "elicitation":
                rows.append(r)
    return rows

def en_sig(s):
    s = re.sub(r"\(.*?\)", " ", s)
    return frozenset(w for w in T.en_terms(s) if w not in PRON_EN and w not in {T.crude_stem(x) for x in PRON_EN} and len(w) > 1)

def mw_sig(s):
    s = re.sub(r"\(.*?\)", " ", s.split("~")[0])
    out = set()
    for t in T.mw_tokens(s):
        k = T.wkey(re.sub(r"^(n’|m’|ng’)", "", T.nrm(t).lower()))
        if k and k not in {T.wkey(x) for x in PRON_MW}:
            out.add(k)
    return frozenset(out)

def ref_base(ref):
    ref = str(ref)
    m = re.match(r"^([A-Za-z]*\d+)", ref)
    return m.group(1) if m else re.sub(r"[^A-Za-z]", "", ref)[:1]

def ref_num(ref):
    m = re.search(r"(\d+)", str(ref))
    return int(m.group(1)) if m else None

def cluster(rows):
    """Union-find over: same ref base, identical EN/MW content signature, high EN-signature Jaccard, adjacent paradigm numbers."""
    n = len(rows)
    par = list(range(n))
    def f(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    def u(a, b):
        par[f(a)] = f(b)
    es = [en_sig(r["english"]) for r in rows]
    ms = [mw_sig(r["mwaghavul"]) for r in rows]
    bases = collections.defaultdict(list)
    for i, r in enumerate(rows):
        bases[ref_base(r["source"]["ref"])].append(i)
    for v in bases.values():
        for j in v[1:]:
            u(v[0], j)
    by_es, by_ms = collections.defaultdict(list), collections.defaultdict(list)
    for i in range(n):
        if es[i]:
            by_es[es[i]].append(i)
        if ms[i]:
            by_ms[ms[i]].append(i)
    for d in (by_es, by_ms):
        for v in d.values():
            for j in v[1:]:
                u(v[0], j)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = es[i], es[j]
            if len(a) >= 2 and len(b) >= 2 and len(a & b) / len(a | b) >= 0.7:
                u(i, j)
            ni, nj = ref_num(rows[i]["source"]["ref"]), ref_num(rows[j]["source"]["ref"])
            if ni is not None and nj is not None and abs(ni - nj) <= 1 and a and b and len(a & b) / len(a | b) >= 0.5:
                u(i, j)
    cl = collections.defaultdict(list)
    for i in range(n):
        cl[f(i)].append(i)
    return list(cl.values())

def sha(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

# ------------------------------------------------------------------------------------------------ select
def stage_select():
    rows = load_gold_elicitation()
    cl = cluster(rows)
    sizes = collections.Counter(len(c) for c in cl)
    print(f"GOLD elicitation sentences: {len(rows)}  → {len(cl)} clusters  (size histogram: {dict(sorted(sizes.items()))})")
    seen_clusters = {k for k, c in enumerate(cl) if any(str(rows[i]['source']['ref']) in SEEN_DURING_DEV for i in c)}
    def ok(i):
        r = rows[i]; en, mw = r["english"], r["mwaghavul"]
        return not re.search(r"[~—(…]", en + mw) and 3 <= len(en.split()) <= 11 and 2 <= len(T.mw_tokens(mw)) <= 9 and \
               not str(r["source"]["ref"]).startswith(("S", "R", "C"))
    eligible = [k for k, c in enumerate(cl) if k not in seen_clusters and any(ok(i) for i in c)]
    print(f"clusters ineligible (contain a row I looked at while developing): {len(seen_clusters)};  eligible: {len(eligible)}")
    rng = random.Random(SEED)
    picks = rng.sample(eligible, 10)
    # H10 replaced: its first RBMT run surfaced a near-duplicate SILVER row (hyphen/tone variant) that my filter missed, and I read that
    # row's English side -> H10 is contaminated. Spare cluster comes from an independent RNG; no reference had been opened.
    spare_rng = random.Random(SEED + 1)
    spare = spare_rng.choice([k for k in eligible if k not in picks])
    picks[9] = spare
    held, excl_ids, excl_pairs, sealed, blind = [], set(), [], [], []
    for n, k in enumerate(picks):
        cands = [i for i in cl[k] if ok(i)]
        i = (spare_rng if n == 9 else rng).choice(cands)
        r = rows[i]
        direction = "EN→MW" if n < 5 else "MW→EN"
        for j in cl[k]:
            excl_ids.add(rows[j]["id"]); excl_pairs.append({"english": rows[j]["english"], "mwaghavul": rows[j]["mwaghavul"]})
        sid = f"H{n+1:02d}"
        src = r["english"] if direction == "EN→MW" else r["mwaghavul"].split("~")[0].strip()
        ref = r["mwaghavul"] if direction == "EN→MW" else r["english"]
        blind.append({"id": sid, "direction": direction, "source": src, "cluster_size": len(cl[k])})
        sealed.append({"id": sid, "direction": direction, "reference": ref, "corpus_id": r["id"], "ref": r["source"]["ref"]})
        held.append(sid)
    for m, s in enumerate(MODERN):
        blind.append({"id": f"M{m+1:02d}", "direction": "EN→MW", "source": s, "cluster_size": None})
    json.dump(blind, open(WORK / "blind_inputs.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump({"excluded_ids": sorted(excl_ids), "excluded_pairs": excl_pairs}, open(WORK / "exclusions.json", "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(sealed, open(SEAL, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"seed={SEED}; sealed-reference SHA-256 commitment: {sha(sealed)}")
    print(f"held-out clusters removed from index: {len(picks)}; rows removed (siblings included): {len(excl_ids)}")
    print("\nBLIND INPUTS (sources only):")
    for b in blind:
        print(f"  {b['id']} [{b['direction']}] (cluster size {b['cluster_size']}): {b['source']}")

# ------------------------------------------------------------------------------------------------ rbmt
def stage_rbmt():
    ex = json.load(open(WORK / "exclusions.json", encoding="utf-8"))
    blind = json.load(open(WORK / "blind_inputs.json", encoding="utf-8"))
    eng = T.Engine(exclude_ids=set(ex["excluded_ids"]), exclude_pairs=ex["excluded_pairs"])
    total = sum(1 for _ in open(T.DATA / "hybrid_llm_parallel_corpus.jsonl", encoding="utf-8"))
    print(f"RAG index: {len(eng.ret.recs)} of {total} records kept ({total-len(eng.ret.recs)} removed: held-out clusters + near-duplicates)")
    # closed-class pruning: forms attested ONLY in removed rows are dropped from the tables
    cnt = collections.Counter()
    for r in eng.ret.recs:
        if r["unit_type"] == "sentence":
            for t in T.mw_tokens(r["mwaghavul"]):
                cnt[T.wkey(t)] += 1
    dropped = []
    for tab in (T.SUBJ_PRON, T.OBJ_PRON, T.DAT_PRON, T.POSS_ATTR, T.POSS_INDEP, T.WH):
        for k, v in list(tab.items()):
            form = T.wkey(re.sub(r"^(n’|m’|ng’)", "", v.split()[-1]))
            if cnt[form] == 0:
                dropped.append((k, v)); tab.pop(k)
    print("closed-class entries dropped (no attestation outside held-out clusters):", dropped or "none")
    packets = []
    for b in blind:
        pk = eng.run(b["source"], "en" if b["direction"] == "EN→MW" else "mw")
        pk["id"] = b["id"]
        pk["analysis"] = pk.get("analysis")
        packets.append(pk)
        ids = {e["id"] for e in pk["examples"]}
        assert not (ids & set(ex["excluded_ids"])), "LEAK: held-out row retrieved"
        if b["id"].startswith("H"):
            side = "english" if b["direction"] == "EN→MW" else "mwaghavul"
            f = (lambda x: set(T.en_terms(x))) if side == "english" else (lambda x: set(T.mw_terms(x)))
            q = f(b["source"])
            mx = max((len(q & f(e[side])) / max(1, len(q | f(e[side]))) for e in pk["examples"]), default=0)
            pk["max_source_side_jaccard"] = round(mx, 2)
            print(f"  audit {b['id']}: max source-side Jaccard of any retrieved example = {mx:.2f}" + ("   <-- CHECK" if mx >= 0.7 else ""))
    json.dump(packets, open(WORK / "rbmt_packets.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("leak assertion passed: no held-out/sibling row appears in any retrieved example list")
    for pk in packets:
        T.show(pk)

# ------------------------------------------------------------------------------------------------ score
def tone_strip(s):
    s = T.strip_tone(T.nrm(s).lower())
    return re.sub(r"[’.,?!;:“”\"()]", "", s).strip()

def stage_score():
    from sacrebleu.metrics import BLEU, CHRF
    sealed = json.load(open(SEAL, encoding="utf-8"))
    assert sha(sealed) == open(WORK / "commitment.txt").read().strip(), "sealed refs changed since selection"
    pk = {p["id"]: p for p in json.load(open(WORK / "rbmt_packets.json", encoding="utf-8"))}
    zs = json.load(open(WORK / "zero_shot.json", encoding="utf-8"))
    hy = json.load(open(WORK / "hybrid.json", encoding="utf-8"))
    chrf, bleu = CHRF(word_order=2), BLEU(effective_order=True)
    rows = []
    systems = {"Pure RBMT": {k: v["draft"] for k, v in pk.items()}, "Zero-Shot LLM": zs, "Hybrid RBMT+LLM": hy}
    for s in sealed:
        refs = [x.strip() for x in s["reference"].split("~")]
        for name, outs in systems.items():
            h = outs[s["id"]]
            rows.append({"id": s["id"], "dir": s["direction"], "system": name, "hyp": h, "ref": refs[0],
                         "chrf": chrf.sentence_score(h, refs).score, "bleu": bleu.sentence_score(h, refs).score,
                         "chrf_ns": chrf.sentence_score(tone_strip(h), [tone_strip(r) for r in refs]).score})
    json.dump(rows, open(WORK / "scores.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{'id':4s} {'dir':6s} {'system':16s} {'chrF++':>7s} {'BLEU':>6s} {'chrF++*':>8s}")
    for r in rows:
        print(f"{r['id']:4s} {r['dir']:6s} {r['system']:16s} {r['chrf']:7.1f} {r['bleu']:6.1f} {r['chrf_ns']:8.1f}")
    print("\nCORPUS-LEVEL (sacrebleu):")
    for scope in ("EN→MW", "MW→EN", "ALL"):
        for name in systems:
            sub = [r for r in rows if r["system"] == name and (scope == "ALL" or r["dir"] == scope)]
            hyps = [r["hyp"] for r in sub]
            refs = [[x.strip() for x in next(s for s in sealed if s["id"] == r["id"])["reference"].split("~")][0] for r in sub]
            print(f"  {scope:6s} {name:16s} n={len(sub):2d}  chrF++={CHRF(word_order=2).corpus_score(hyps,[refs]).score:5.1f}  BLEU={BLEU().corpus_score(hyps,[refs]).score:5.1f}  "
                  f"chrF++*={CHRF(word_order=2).corpus_score([tone_strip(h) for h in hyps],[[tone_strip(x) for x in refs]]).score:5.1f}")
    print("  (* = tone marks and punctuation stripped before scoring; main columns use raw strings)")

if __name__ == "__main__":
    st = sys.argv[1]
    if st == "select":
        stage_select()
        (WORK / "commitment.txt").write_text(sha(json.load(open(SEAL, encoding="utf-8"))))
    elif st == "rbmt":
        stage_rbmt()
    elif st == "score":
        stage_score()
