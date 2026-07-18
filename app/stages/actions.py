#!/usr/bin/env python3
"""
actions.py - Stage 4 of the Rare Disease Diagnostic Agent.

Converts a ranked differential into things a clinician can actually do today:
the test to order, the red flags to act on now, and open trials to consider.

NO LLM HERE EITHER. The gene panel comes verbatim from rare_diseases.json, the
trials come from the live ClinicalTrials.gov API. Nothing on this screen was
written by a language model, which is what lets you put a gene list in front of
a geneticist without flinching.

Two correctness traps this module handles explicitly, because getting them
wrong would give a clinician actively wrong advice:

  NOT EVERYTHING IS A GENE PANEL. 22q11.2 deletion syndrome is missed by
  sequencing - it needs microarray, FISH or MLPA. Behcet is not monogenic at
  all; a panel is the wrong test. The DB carries recommended_test.type and this
  module surfaces it rather than assuming "panel".

  SHARED PANELS ARE A FEATURE. Marfan, Loeys-Dietz and vascular EDS are the
  classic confusion set - and they are covered by the SAME aortopathy panel.
  When the top candidates share a panel, we say so: the differential does not
  need to be resolved before the clinician can act.

Trials: live API with an on-disk cache fallback. Venue wifi fails; demos should
not. Build the cache before you present:

    python app/stages/actions.py --build-cache

API: https://clinicaltrials.gov/api/v2/studies  (no key, ~50 req/min)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request

CTGOV = "https://clinicaltrials.gov/api/v2/studies"
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "data", "trial_cache.json")


# ------------------------------------------------------------ trials ------

def fetch_trials(condition: str, max_results: int = 3, timeout: int = 8) -> list[dict]:
    """Query ClinicalTrials.gov for recruiting studies. Raises on failure."""
    params = {
        "query.cond": condition,
        "filter.overallStatus": "RECRUITING,NOT_YET_RECRUITING",
        "pageSize": max_results,
        "format": "json",
    }
    url = f"{CTGOV}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "rare-disease-agent/hackathon"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))

    trials = []
    for study in data.get("studies", []):
        p = study.get("protocolSection", {})
        ident = p.get("identificationModule", {})
        status = p.get("statusModule", {})
        elig = p.get("eligibilityModule", {})
        locs = p.get("contactsLocationsModule", {}).get("locations", []) or []
        trials.append({
            "nct_id": ident.get("nctId", ""),
            "title": ident.get("briefTitle", ""),
            "status": status.get("overallStatus", ""),
            "phase": ", ".join(p.get("designModule", {}).get("phases", []) or []) or "N/A",
            "min_age": elig.get("minimumAge", ""),
            "max_age": elig.get("maximumAge", ""),
            "sex": elig.get("sex", "ALL"),
            "accepts_healthy": elig.get("healthyVolunteers", False),
            "n_locations": len(locs),
            "first_location": (f"{locs[0].get('city','')}, {locs[0].get('country','')}"
                               if locs else ""),
            "url": f"https://clinicaltrials.gov/study/{ident.get('nctId','')}",
        })
    return trials


def load_cache() -> dict:
    try:
        return json.load(open(CACHE_PATH))
    except Exception:
        return {}


def get_trials(condition: str, max_results: int = 3) -> tuple[list[dict], str]:
    """Live lookup with cache fallback. Returns (trials, provenance)."""
    try:
        return fetch_trials(condition, max_results), "live"
    except Exception as e:
        cached = load_cache().get(condition)
        if cached:
            return cached[:max_results], f"cache (live failed: {type(e).__name__})"
        return [], f"unavailable ({type(e).__name__})"


def build_cache(db: dict, max_results: int = 3) -> None:
    """Pre-fetch trials for every disease. Run before demoing."""
    cache, ok = {}, 0
    for d in db["diseases"]:
        q = d["trial_query"]
        try:
            cache[q] = fetch_trials(q, max_results)
            ok += 1
            print(f"  {len(cache[q])} trials  {q}")
        except Exception as e:
            print(f"  FAILED     {q}  ({type(e).__name__})")
        time.sleep(1.2)  # stay well inside the rate limit
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    json.dump(cache, open(CACHE_PATH, "w"), indent=1)
    print(f"\ncached {ok}/{len(db['diseases'])} diseases -> {CACHE_PATH}")


# ------------------------------------------------------------ actions -----

def plan_actions(candidates, db: dict, top_n: int = 3, with_trials: bool = True) -> dict:
    """Build the action plan for the top candidates of a differential."""
    by_id = {d["id"]: d for d in db["diseases"]}
    plans, panels_seen = [], {}

    for cand in candidates[:top_n]:
        d = by_id.get(cand.disease_id)
        if not d:
            continue
        test = d["recommended_test"]

        trials, prov = ([], "skipped")
        if with_trials:
            trials, prov = get_trials(d["trial_query"])

        plans.append({
            "rank": len(plans) + 1,
            "disease": d["name"],
            "disease_id": d["id"],
            "score": round(cand.score, 3),
            "confidence": cand.confidence(),
            "inheritance": d["inheritance"],
            "test": {
                "type": test["type"],
                "name": test["panel_name"],
                "genes": test.get("genes", []),
                "turnaround": test.get("turnaround", ""),
                "adjunct": test.get("adjunct", []),
                "is_sequencing_panel": test["type"].lower().startswith("targeted gene"),
            },
            "red_flags": d.get("red_flags", []),
            "discriminating_features": d.get("discriminating_features", []),
            "commonly_misdiagnosed_as": d.get("commonly_misdiagnosed_as", []),
            "trials": trials,
            "trials_source": prov,
            "source": d.get("source", ""),
        })
        panels_seen.setdefault(test["panel_name"], []).append(d["name"])

    # If the top candidates share a panel, one order covers them all.
    shared = [{"panel": p, "covers": names}
              for p, names in panels_seen.items() if len(names) > 1]

    # Any red flag on any top candidate is worth surfacing immediately.
    urgent = []
    for p in plans:
        for rf in p["red_flags"]:
            urgent.append({"disease": p["disease"], "flag": rf})

    return {
        "plans": plans,
        "shared_panels": shared,
        "urgent_flags": urgent,
        "single_order_resolves": bool(shared and len(shared[0]["covers"]) >= 2),
    }


def render_markdown(plan: dict) -> str:
    """The clinician-facing output - section 3 of the report."""
    L = []
    if plan["urgent_flags"]:
        L.append("### Act on these now\n")
        for f in plan["urgent_flags"][:4]:
            L.append(f"- **{f['disease']}** — {f['flag']}")
        L.append("")

    if plan["shared_panels"]:
        s = plan["shared_panels"][0]
        L.append(f"### One test covers the top candidates\n")
        L.append(f"**{s['panel']}** covers {', '.join(s['covers'])}. "
                 f"The differential does not need to be resolved before ordering.\n")

    L.append("### Recommended next steps\n")
    for p in plan["plans"]:
        L.append(f"**{p['rank']}. {p['disease']}** ({p['confidence']} confidence, "
                 f"{p['inheritance']})")
        L.append(f"- **Order:** {p['test']['name']} — *{p['test']['type']}*")
        if p["test"]["genes"]:
            L.append(f"  - Genes: {', '.join(p['test']['genes'])}")
        if p["test"]["turnaround"]:
            L.append(f"  - Turnaround: {p['test']['turnaround']}")
        for adj in p["test"]["adjunct"]:
            L.append(f"- **Also:** {adj}")
        if p["discriminating_features"]:
            L.append(f"- **Look for:** {p['discriminating_features'][0]}")
        if p["trials"]:
            L.append(f"- **Open trials** ({p['trials_source']}):")
            for t in p["trials"]:
                L.append(f"  - [{t['nct_id']}]({t['url']}) — {t['title'][:70]} "
                         f"({t['status'].lower().replace('_',' ')})")
        elif p["trials_source"] != "skipped":
            L.append(f"- No recruiting trials found ({p['trials_source']})")
        L.append(f"- *Source: {p['source']}*")
        L.append("")

    L.append("---")
    L.append("*Clinical decision support. Not a diagnosis. All gene panels and "
             "trial records are retrieved from curated sources; none are model-generated.*")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--db", default=os.path.join(here, "..", "data", "rare_diseases.json"))
    ap.add_argument("--hpo", default=os.path.join(here, "..", "data", "hp.json"))
    ap.add_argument("--build-cache", action="store_true")
    ap.add_argument("--no-trials", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("terms", nargs="*", help="HPO ids, e.g. HP:0030955 HP:0031006")
    a = ap.parse_args()

    sys.path.insert(0, here)
    from differential import score_diseases, load_db
    db = load_db(a.db)

    if a.build_cache:
        build_cache(db)
        return 0

    if not a.terms:
        ap.error("give HPO ids, or use --build-cache")

    from ground import HPOIndex
    ranked = score_diseases(a.terms, db, HPOIndex(a.hpo))
    if not ranked:
        print("no candidates scored")
        return 1

    plan = plan_actions(ranked, db, with_trials=not a.no_trials)
    print(json.dumps(plan, indent=2) if a.json else render_markdown(plan))
    return 0


if __name__ == "__main__":
    sys.exit(main())
