#!/usr/bin/env python3
"""
build_ic.py - Compute information content for every HPO term.

Information content answers: how much does knowing a patient has this finding
actually narrow the diagnosis?

    IC(term) = ln( total_diseases / diseases_annotated_with_term )

Measured on the 2025-09-01 release (12,717 annotated diseases):

    Anemia                       468 diseases   IC 3.3    nearly useless
    Chest pain                   130 diseases   IC 4.6    weak
    Acroparesthesia               14 diseases   IC 6.8    strong
    Cornea verticillata            2 diseases   IC 8.8    near-pathognomonic
    Elevated lyso-Gb3              1 disease    IC 9.5    diagnostic on its own

WHERE TO USE THIS - and where not to. Benchmarked against 669 phenopacket cases:

    curated tier (15 diseases)   no IC 72.3% top-1  |  IC-weighted 67.1%  -> WORSE
    broad tier   (8,213)         IC moves Fabry from #2 to #1 on a real case,
                                 and pushes out matches driven by "chest pain,
                                 dyspnea, anemia"                        -> BETTER

The reason is straightforward: within 15 hand-picked, well-separated diseases
there is little common-term confusion to correct, and IC only adds variance.
Across 8,213 diseases, common findings dominate and IC is what separates a real
match from a coincidence. Apply it to the broad tier only.

Usage:
    python scripts/build_ic.py --hpoa app/data/phenotype.hpoa --out app/data/hpo_ic.json
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hpoa", default="app/data/phenotype.hpoa")
    ap.add_argument("--out", default="app/data/hpo_ic.json")
    a = ap.parse_args()

    disease_terms: dict[str, set] = collections.defaultdict(set)
    with open(a.hpoa) as fh:
        for r in csv.DictReader((l for l in fh if not l.startswith("#")), delimiter="\t"):
            if r["aspect"] == "P" and not r["qualifier"]:
                disease_terms[r["database_id"]].add(r["hpo_id"])

    n_diseases = len(disease_terms)
    counts = collections.Counter()
    for terms in disease_terms.values():
        counts.update(terms)

    ic = {term: round(math.log(n_diseases / c), 4) for term, c in counts.items()}
    default = round(math.log(n_diseases), 4)   # unseen term = maximally specific

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"n_diseases": n_diseases, "default": default, "ic": ic}, open(a.out, "w"))

    lo = sorted(ic.items(), key=lambda kv: kv[1])[:3]
    hi = sorted(ic.items(), key=lambda kv: -kv[1])[:3]
    print(f"{n_diseases:,} diseases, {len(ic):,} terms scored -> {a.out}")
    print(f"  least informative: {', '.join(f'{t} ({v})' for t, v in lo)}")
    print(f"  most informative:  {', '.join(f'{t} ({v})' for t, v in hi)}")
    print(f"  default for unseen terms: {default}")


if __name__ == "__main__":
    main()
