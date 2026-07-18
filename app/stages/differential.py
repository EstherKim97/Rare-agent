#!/usr/bin/env python3
"""
differential.py - Stage 3a of the Rare Disease Diagnostic Agent.

Ranks candidate diseases against a patient's grounded HPO terms.

THERE IS NO LLM IN THIS FILE EITHER. Stage 3b (Claude) writes the *reasoning*
for the candidates this module selects, but it cannot add a disease that did
not score, and it cannot change the ranking. The differential is arithmetic.

Scoring model
-------------
Each disease phenotype carries a weight from its HPO annotation frequency:

    obligate   3.0    present in essentially all patients
    frequent   2.0    present in 30-79%
    occasional 1.0    present in <30%

For a patient term set P and disease phenotype set D:

    matched   = sum of weights of D-terms the patient has
    coverage  = matched / total weight of D          (did we see this disease?)
    precision = matched_count / |P|                  (does it explain the patient?)
    score     = coverage^COV_EXP * precision^PREC_EXP

Coverage alone favours diseases with few annotated terms; precision alone
favours diseases with many. The exponents let you weight the trade-off - they
are tuned empirically against phenopacket-store, not guessed. Run
benchmark.py after changing them.

Optional ancestor credit: if the patient has "Abnormal finger morphology" and
the disease lists "Arachnodactyly", a partial match is granted at
ANCESTOR_CREDIT weight. Requires an HPOIndex; omit for exact matching only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

WEIGHTS = {"obligate": 3.0, "frequent": 2.0, "occasional": 1.0}

IC_EXP = 2.0   # applied to broad tier only; measured worse on curated

# Tuned against phenopacket-store. See benchmark.py.
COV_EXP = 1.0
PREC_EXP = 0.15
ANCESTOR_CREDIT = 0.5
ANCESTOR_DEPTH = 2


@dataclass
class Candidate:
    disease_id: str
    name: str
    score: float
    coverage: float
    precision: float
    matched: list[dict] = field(default_factory=list)     # hpo_id, label, frequency, via
    missing_obligate: list[dict] = field(default_factory=list)
    unexplained: list[str] = field(default_factory=list)  # patient terms this disease misses

    def to_dict(self) -> dict:
        return {
            "disease_id": self.disease_id,
            "name": self.name,
            "score": round(self.score, 4),
            "confidence": self.confidence(),
            "coverage": round(self.coverage, 3),
            "precision": round(self.precision, 3),
            "matched": self.matched,
            "missing_obligate": self.missing_obligate,
            "unexplained_patient_terms": self.unexplained,
        }

    def confidence(self) -> str:
        """Coarse band for display. Never present the raw score as a probability."""
        if self.score >= 0.55:
            return "high"
        if self.score >= 0.30:
            return "moderate"
        return "low"


def load_db(path: str) -> dict:
    return json.load(open(path))


def score_diseases(
    patient_hpo_ids: list[str],
    db: dict,
    hpo_index=None,
    top_n: int = 5,
    min_score: float = 0.05,
    ic=None,
) -> list[Candidate]:
    """Rank diseases in db against the patient's HPO term list."""
    patient = list(dict.fromkeys(patient_hpo_ids))  # de-dupe, keep order
    if not patient:
        return []
    patient_set = set(patient)

    # Precompute patient ancestors once, not per disease.
    anc_map: dict[str, set[str]] = {}
    if hpo_index is not None:
        for pid in patient:
            anc_map[pid] = set(hpo_index.ancestors(pid, ANCESTOR_DEPTH))

    out: list[Candidate] = []

    def w_of(p):
        base = WEIGHTS[p["frequency"]]
        if not ic:
            return base
        return base * (ic["ic"].get(p["hpo_id"], ic["default"]) ** IC_EXP)

    for disease in db["diseases"]:
        phenos = disease["phenotypes"]
        total_w = sum(w_of(p) for p in phenos) or 1.0

        matched_w = 0.0
        matched, missing_obl, explained = [], [], set()

        for p in phenos:
            hid, w = p["hpo_id"], w_of(p)

            if hid in patient_set:
                matched_w += w
                matched.append({**{k: p[k] for k in ("hpo_id", "label", "frequency")},
                                "via": "exact"})
                explained.add(hid)
                continue

            # Ancestor credit: patient term is a parent/child of this phenotype.
            if hpo_index is not None:
                rel = next((pid for pid in patient
                            if hid in anc_map.get(pid, ()) or pid in hpo_index.ancestors(hid, ANCESTOR_DEPTH)),
                           None)
                if rel:
                    matched_w += w * ANCESTOR_CREDIT
                    matched.append({**{k: p[k] for k in ("hpo_id", "label", "frequency")},
                                    "via": f"related:{rel}"})
                    explained.add(rel)
                    continue

            if p["frequency"] == "obligate":
                missing_obl.append({"hpo_id": hid, "label": p["label"]})

        if matched_w == 0:
            continue

        coverage = matched_w / total_w
        precision = len(explained) / len(patient)
        score = (coverage ** COV_EXP) * (precision ** PREC_EXP)

        if score < min_score:
            continue

        matched.sort(key=lambda m: -WEIGHTS[m["frequency"]])
        out.append(Candidate(
            disease_id=disease["id"],
            name=disease["name"],
            score=score,
            coverage=coverage,
            precision=precision,
            matched=matched,
            missing_obligate=missing_obl,
            unexplained=[t for t in patient if t not in explained],
        ))

    out.sort(key=lambda c: -c.score)
    return out[:top_n]


if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="app/data/rare_diseases.json")
    ap.add_argument("--hpo", help="path to hp.json to enable ancestor credit")
    ap.add_argument("terms", nargs="+", help="HPO ids, e.g. HP:0001166 HP:0001083")
    a = ap.parse_args()

    idx = None
    if a.hpo:
        sys.path.insert(0, "app/stages")
        from ground import HPOIndex
        idx = HPOIndex(a.hpo)

    for i, c in enumerate(score_diseases(a.terms, load_db(a.db), idx), 1):
        print(f"{i}. {c.name:<48} {c.score:.3f}  ({c.confidence()}) "
              f"cov={c.coverage:.2f} prec={c.precision:.2f}")
