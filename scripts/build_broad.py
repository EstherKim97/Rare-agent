#!/usr/bin/env python3
"""
build_broad.py - Generate the broad-tier disease database.

Produces a second, much larger DB covering most of the rare disease space, so
the agent is not limited to the 15 hand-curated conditions.

    curated tier   15 diseases     full clinical output: gene panel, red flags,
                                   adjunct workup, discriminating features
    broad tier     ~8,200 diseases ranking only: "this pattern resembles X",
                                   with an Orphanet/OMIM link

WHY THE MINIMUM PHENOTYPE FILTER EXISTS. HPOA contains ~12,700 diseases, but
thousands have only 3-5 annotated phenotypes. Those score spuriously high: match
two of four annotations and coverage is 0.5, beating a well-characterised disease
where you matched 8 of 30. Measured on a Fabry phenotype set:

    no filter  (10,913 diseases)   Fabry not in top 5 - beaten by
                                   "Cardiomyopathy, familial hypertrophic, 20"
    >=10 terms  (8,228 diseases)   Fabry ranked #1

The filter is not arbitrary trimming; it removes entries too sparsely annotated
to rank honestly. Diseases below the threshold are still real - they are just
not rankable from a phenotype list, and pretending otherwise produces confident
nonsense.

Usage:
    python scripts/build_broad.py --min-phenotypes 10
"""

from __future__ import annotations

import argparse
import csv
import json
import os

FREQ_TERMS = {"HP:0040280": "obligate", "HP:0040281": "frequent",
              "HP:0040282": "frequent", "HP:0040283": "occasional",
              "HP:0040284": "occasional", "HP:0040285": "excluded"}


def bucket(raw: str) -> str:
    if not raw:
        return "frequent"
    if raw in FREQ_TERMS:
        return FREQ_TERMS[raw]
    try:
        if raw.endswith("%"):
            v = float(raw.rstrip("%").split("-")[-1])
        elif "/" in raw:
            num, den = raw.split("/")
            v = 100.0 * float(num) / float(den) if float(den) else 0.0
        else:
            return "frequent"
    except (ValueError, ZeroDivisionError):
        return "frequent"
    return "obligate" if v >= 99 else "frequent" if v >= 30 else "occasional"


def load_labels(hp_json: str) -> dict:
    out = {}
    for n in json.load(open(hp_json))["graphs"][0]["nodes"]:
        if "/HP_" in n.get("id", "") and n.get("lbl") \
                and not n.get("meta", {}).get("deprecated"):
            out[n["id"].split("/")[-1].replace("_", ":")] = n["lbl"]
    return out


def external_url(disease_id: str) -> str:
    if disease_id.startswith("OMIM:"):
        return f"https://omim.org/entry/{disease_id[5:]}"
    if disease_id.startswith("ORPHA:"):
        return f"https://www.orpha.net/en/disease/detail/{disease_id[6:]}"
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hp", default="app/data/hp.json")
    ap.add_argument("--hpoa", default="app/data/phenotype.hpoa")
    ap.add_argument("--curated", default="app/data/rare_diseases.json")
    ap.add_argument("--out", default="app/data/rare_diseases_broad.json")
    ap.add_argument("--min-phenotypes", type=int, default=10)
    a = ap.parse_args()

    labels = load_labels(a.hp)
    print(f"[hp.json] {len(labels):,} active terms")

    curated_ids = {d["id"] for d in json.load(open(a.curated))["diseases"]}

    phenos: dict[str, list] = {}
    names: dict[str, str] = {}
    with open(a.hpoa) as fh:
        rows = csv.DictReader((l for l in fh if not l.startswith("#")), delimiter="\t")
        for r in rows:
            if r["aspect"] != "P" or r["qualifier"] or r["hpo_id"] not in labels:
                continue
            did = r["database_id"]
            phenos.setdefault(did, []).append({
                "hpo_id": r["hpo_id"],
                "label": labels[r["hpo_id"]],
                "frequency": bucket(r["frequency"]),
            })
            names[did] = r["disease_name"]

    total = len(phenos)
    kept = {k: v for k, v in phenos.items()
            if len(v) >= a.min_phenotypes and k not in curated_ids}

    order = {"obligate": 0, "frequent": 1, "occasional": 2}
    diseases = []
    for did, ph in kept.items():
        ph.sort(key=lambda p: (order[p["frequency"]], p["label"]))
        diseases.append({
            "id": did,
            "name": names[did],
            "tier": "broad",
            "inheritance": "See source",
            "phenotypes": ph,
            "external_url": external_url(did),
            "source": f"HPO annotation ({did})",
        })
    diseases.sort(key=lambda d: -len(d["phenotypes"]))

    db = {
        "schema_version": "1.0-broad",
        "hpo_release": "2025-09-01",
        "min_phenotypes": a.min_phenotypes,
        "disease_count": len(diseases),
        "note": ("Auto-generated ranking tier. No curated clinical actions. "
                 "Diseases with fewer than min_phenotypes annotations are "
                 "excluded: too sparse to rank honestly."),
        "diseases": diseases,
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(db, open(a.out, "w"))

    size = os.path.getsize(a.out) / 1e6
    print(f"[phenotype.hpoa] {total:,} diseases annotated")
    print(f"  excluded {total - len(kept) - len(curated_ids):,} with "
          f"<{a.min_phenotypes} phenotypes")
    print(f"  excluded {len(curated_ids)} already in the curated tier")
    print(f"\nwrote {a.out}: {len(diseases):,} diseases, "
          f"{sum(len(d['phenotypes']) for d in diseases):,} annotations, {size:.1f} MB")


if __name__ == "__main__":
    main()
