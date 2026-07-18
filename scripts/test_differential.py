#!/usr/bin/env python3
"""
test_differential.py - Does the scorer actually work, or does it just produce
a number that looks good?

Three classes of test:

  UNIT      - the arithmetic does what the docstring claims
  BEHAVIOUR - clinically obvious inputs produce clinically obvious outputs
  VALIDITY  - the benchmark measures signal, not an artifact of its own setup

The VALIDITY tests are the ones that matter for a demo. Any pipeline can print
an accuracy; these check the accuracy means something. Run before you quote a
number on stage.

    python scripts/test_differential.py --zip app/data/all_phenopackets.zip
    python scripts/test_differential.py --fast     # skip benchmark-dependent tests
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from collections import Counter

sys.path.insert(0, "app/stages")

from differential import score_diseases, load_db, WEIGHTS
from ground import HPOIndex

PASS, FAIL = "  PASS", "  FAIL"
_results = []


def check(name, condition, detail=""):
    _results.append(bool(condition))
    print(f"{PASS if condition else FAIL}  {name}" + (f"   [{detail}]" if detail else ""))


# ---------------------------------------------------------------- UNIT ----

def unit_tests(db, idx):
    print("\nUNIT - scoring arithmetic")

    marfan = next(d for d in db["diseases"] if d["id"] == "ORPHA:558")
    all_terms = [p["hpo_id"] for p in marfan["phenotypes"]]

    # Feeding a disease its own complete phenotype set must rank it first
    # with coverage 1.0 - if this fails the matching loop is broken.
    r = score_diseases(all_terms, db, idx)
    check("perfect input ranks the source disease first",
          r and r[0].disease_id == "ORPHA:558", f"got {r[0].name if r else 'nothing'}")
    check("perfect input gives coverage == 1.0",
          r and abs(r[0].coverage - 1.0) < 1e-9, f"coverage={r[0].coverage:.3f}" if r else "")

    # Empty input must not crash and must not invent candidates.
    check("empty input returns no candidates", score_diseases([], db, idx) == [])

    # Nonsense HPO ids must score nothing.
    check("unrelated terms return no candidates",
          score_diseases(["HP:0000001"], db, idx) == [] or
          all(c.score < 0.2 for c in score_diseases(["HP:0000001"], db, idx)))

    # Weights must be ordered - an obligate finding cannot be worth less
    # than an occasional one, or the whole ranking is meaningless.
    check("frequency weights are strictly ordered",
          WEIGHTS["obligate"] > WEIGHTS["frequent"] > WEIGHTS["occasional"])

    # Duplicate input must not inflate the score.
    a = score_diseases(all_terms, db, idx)[0].score
    b = score_diseases(all_terms + all_terms, db, idx)[0].score
    check("duplicate terms do not inflate score", abs(a - b) < 1e-9)

    # Adding noise should lower, never raise, the true disease's score.
    noisy = score_diseases(all_terms + ["HP:0001250", "HP:0000365"], db, idx)
    m = next((c for c in noisy if c.disease_id == "ORPHA:558"), None)
    check("unrelated noise does not increase score", m and m.score <= a + 1e-9)

    # Monotonicity: more true findings should not reduce the score.
    half = score_diseases(all_terms[:6], db, idx)
    hm = next((c for c in half if c.disease_id == "ORPHA:558"), None)
    check("more true findings -> higher score", hm and hm.score <= a + 1e-9,
          f"{hm.score:.3f} -> {a:.3f}" if hm else "")


# ----------------------------------------------------------- BEHAVIOUR ----

def behaviour_tests(db, idx):
    print("\nBEHAVIOUR - clinically obvious cases")

    def top(terms):
        r = score_diseases(terms, db, idx)
        return (r[0].name, r[0].score) if r else ("none", 0.0)

    # Fabry: cornea verticillata + acroparesthesia + angiokeratoma + proteinuria
    n, s = top(["HP:0030955", "HP:0031006", "HP:0001075", "HP:0000093"])
    check("Fabry pentad -> Fabry", "Fabry" in n, n)

    # Wilson: KF ring + low ceruloplasmin + dystonia + cirrhosis
    n, s = top(["HP:0200032", "HP:0500009", "HP:0001332", "HP:0001394"])
    check("Wilson triad -> Wilson", "Wilson" in n, n)

    # HHT: recurrent epistaxis + mucosal telangiectasia + pulmonary AVM
    n, s = top(["HP:0004406", "HP:0100582", "HP:0006548"])
    check("HHT triad -> HHT", "telangiectasia" in n.lower(), n)

    # Alport: hematuria + SNHL + anterior lenticonus
    n, s = top(["HP:0000790", "HP:0000407", "HP:0011501"])
    check("Alport triad -> Alport", "Alport" in n, n)

    # NEGATIVE CONTROL - common primary-care findings must not trigger a
    # confident rare-disease call. This is the false-positive guard.
    common = ["HP:0001513",  # Obesity
              "HP:0000822",  # Hypertension
              "HP:0003077",  # Hyperlipidemia
              "HP:0001939",  # Abnormality of metabolism
              "HP:0012531"]  # Pain
    r = score_diseases(common, db, idx)
    worst = r[0].score if r else 0.0
    check("common findings produce no high-confidence call", worst < 0.55,
          f"top score {worst:.3f} ({r[0].name if r else '-'})")

    # A single nonspecific finding must never be high confidence.
    r = score_diseases(["HP:0002650"], db, idx)  # scoliosis alone
    check("isolated scoliosis is not high confidence",
          not r or r[0].confidence() != "high",
          f"{r[0].name} {r[0].confidence()}" if r else "")


# ------------------------------------------------------------ VALIDITY ----

def validity_tests(db, idx, zip_path):
    print("\nVALIDITY - is the benchmark measuring anything real?")
    sys.path.insert(0, "scripts")
    from benchmark import load_cases, evaluate

    cases = load_cases(zip_path)
    if not cases:
        print("  SKIP  no phenopacket cases loaded")
        return

    real = evaluate(cases, db, idx)
    name = {d["id"]: d["name"] for d in db["diseases"]}

    # Beat the trivial baseline. If "always guess the commonest disease" does
    # as well as the model, the model is decorative.
    cnt = Counter(c["truth"] for c in cases)
    maj_id, maj_n = cnt.most_common(1)[0]
    majority = maj_n / len(cases)
    check("beats majority-class baseline", real["top1"] > majority + 0.10,
          f"model {real['top1']:.1%} vs baseline {majority:.1%} ({name[maj_id]})")

    # Permutation test: shuffle which phenotype set belongs to which disease.
    # Real signal must collapse toward chance.
    orig = [d["phenotypes"] for d in db["diseases"]]
    random.seed(0)
    accs = []
    for _ in range(3):
        perm = orig[:]
        random.shuffle(perm)
        for d, p in zip(db["diseases"], perm):
            d["phenotypes"] = p
        accs.append(evaluate(cases, db, idx)["top1"])
    for d, p in zip(db["diseases"], orig):
        d["phenotypes"] = p
    check("permuted labels collapse accuracy", max(accs) < 0.15,
          f"permuted {min(accs):.1%}-{max(accs):.1%} vs real {real['top1']:.1%}")

    # Leakage: if every benchmark term were one we curated, the test would be
    # circular. Most patient terms should be ones we never wrote down.
    byd = {d["id"]: {p["hpo_id"] for p in d["phenotypes"]} for d in db["diseases"]}
    ov = [len(set(c["terms"]) & byd[c["truth"]]) / len(set(c["terms"])) for c in cases]
    check("benchmark is not circular", statistics.mean(ov) < 0.75,
          f"{statistics.mean(ov):.0%} of case terms overlap our curation")

    # Ancestor credit should help, not hurt.
    no_anc = evaluate(cases, db, None)
    check("ancestor credit improves accuracy", real["top1"] > no_anc["top1"],
          f"{no_anc['top1']:.1%} -> {real['top1']:.1%}")

    print(f"\n  n={len(cases)} cases | top-1 {real['top1']:.1%} | top-3 {real['top3']:.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="app/data/rare_diseases.json")
    ap.add_argument("--hpo", default="app/data/hp.json")
    ap.add_argument("--zip", default="app/data/all_phenopackets.zip")
    ap.add_argument("--fast", action="store_true", help="skip benchmark tests")
    a = ap.parse_args()

    db, idx = load_db(a.db), HPOIndex(a.hpo)
    unit_tests(db, idx)
    behaviour_tests(db, idx)
    if not a.fast:
        validity_tests(db, idx, a.zip)

    n, total = sum(_results), len(_results)
    print(f"\n{n}/{total} checks passed")
    sys.exit(0 if n == total else 1)


if __name__ == "__main__":
    main()
