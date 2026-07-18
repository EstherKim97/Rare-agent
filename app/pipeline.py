#!/usr/bin/env python3
"""
pipeline.py - The whole agent, end to end.

One function, run(), takes raw clinical text and returns everything: the
findings, the grounded codes, the ranked differential, the action plan, and a
timed log of what happened at each step.

This is the ONLY module the web app should import. The four stages stay
independent and separately testable; this is the thing that wires them
together and is the single place where the contract between them lives.

    note text
      -> Stage 1 extract.py       Claude reads prose, emits clinical terms
      -> Stage 2 ground.py        terms -> HPO codes            (no LLM)
      -> Stage 3 differential.py  codes -> ranked diseases      (no LLM)
      -> Stage 4 actions.py       diseases -> tests + trials    (no LLM)

Every stage emits a log event with its own timing, so the UI can show the
agent working rather than freezing for twenty seconds. Judges reward seeing
the machinery; a spinner tells them nothing.

Usage:
    python app/pipeline.py --case case_01
    python app/pipeline.py --file app/data/real_cases/PMC12131641_*.txt
    python app/pipeline.py --case case_01 --json > result.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "stages"))
DATA = os.path.join(HERE, "data")

from ground import HPOIndex                      # noqa: E402
from differential import score_diseases, load_db  # noqa: E402
import actions                                    # noqa: E402
import extract as extract_mod                     # noqa: E402


@dataclass
class Event:
    stage: int
    name: str
    detail: str
    ms: int
    data: dict = field(default_factory=dict)

    def to_dict(self):
        return {"stage": self.stage, "name": self.name,
                "detail": self.detail, "ms": self.ms, "data": self.data}


class Agent:
    """Holds the loaded indexes. Build ONCE at server startup, not per request.

    HPOIndex takes ~3 seconds to build. Constructing it inside a request handler
    turns a 6-second demo into a 9-second one and makes the UI feel broken.
    """

    def __init__(self, data_dir: str = DATA, model: str | None = None):
        t = time.perf_counter()
        self.hpo = HPOIndex(os.path.join(data_dir, "hp.json"))
        self.db = load_db(os.path.join(data_dir, "rare_diseases.json"))
        # Optional broad tier - ranking only, no curated clinical actions.
        broad_path = os.path.join(data_dir, "rare_diseases_broad.json")
        self.broad = load_db(broad_path) if os.path.exists(broad_path) else None
        ic_path = os.path.join(data_dir, "hpo_ic.json")
        self.ic = json.load(open(ic_path)) if os.path.exists(ic_path) else None
        self.model = model or extract_mod.DEFAULT_MODEL
        self.startup_ms = int((time.perf_counter() - t) * 1000)

    def run(self, text: str, with_trials: bool = True, top_n: int = 3) -> dict:
        log: list[Event] = []
        t0 = time.perf_counter()

        def mark(stage, name, detail, since, data=None):
            log.append(Event(stage, name, detail,
                             int((time.perf_counter() - since) * 1000), data or {}))

        # ---- Stage 1: extraction (the only LLM call) --------------------
        t = time.perf_counter()
        ex = extract_mod.extract(text, self.model)
        positives = ex.patient_positive()
        mark(1, "Extract findings",
             f"{len(ex.findings)} findings — {len(positives)} patient-positive, "
             f"{len(ex.negated())} negated, {len(ex.family())} family history",
             t,
             {"model": ex.model, "tokens_in": ex.input_tokens,
              "tokens_out": ex.output_tokens,
              "negated": [f.clinical_term for f in ex.negated()],
              "family": [f.clinical_term for f in ex.family()]})

        # ---- Stage 2: grounding -----------------------------------------
        t = time.perf_counter()
        matches = self.hpo.ground_all([f.clinical_term for f in positives])
        trusted = [m for m in matches if m.trusted]
        rate = len(trusted) / max(len(matches), 1)
        mark(2, "Ground to HPO",
             f"{len(trusted)}/{len(matches)} mapped to verified HPO codes ({rate:.0%})",
             t,
             {"grounding_rate": round(rate, 3),
              "unmatched": [m.query for m in matches if not m.trusted],
              "methods": {k: sum(1 for m in matches if m.method == k)
                          for k in {m.method for m in matches}}})

        # ---- Stage 3: differential --------------------------------------
        t = time.perf_counter()
        ranked = score_diseases([m.hpo_id for m in trusted], self.db, self.hpo, top_n=5)
        tier = "curated"
        # Nothing convincing in the curated 15? Fall back to the broad tier.
        if self.broad and (not ranked or ranked[0].score < 0.0):
            broad_ranked = score_diseases([m.hpo_id for m in trusted],
                                          self.broad, self.hpo, top_n=5, ic=self.ic)
            if broad_ranked and (not ranked or broad_ranked[0].score > ranked[0].score):
                ranked, tier = broad_ranked, "broad"
        mark(3, "Rank differential",
             (f"top: {ranked[0].name} ({ranked[0].confidence()} confidence) "
              f"[{tier} tier]" if ranked else "no candidate scored above threshold"),
             t,
             {"tier": tier,
              "searched": len(self.db["diseases"]) if tier == "curated"
                          else len(self.broad["diseases"]),
              "candidates": [{"name": c.name, "score": round(c.score, 3),
                              "confidence": c.confidence()} for c in ranked]})

        # ---- Stage 4: actions -------------------------------------------
        t = time.perf_counter()
        plan = actions.plan_actions(ranked, self.db, top_n=top_n,
                                    with_trials=with_trials) if (ranked and tier == "curated") else \
            {"plans": [], "shared_panels": [], "urgent_flags": [],
             "single_order_resolves": False}
        n_trials = sum(len(p["trials"]) for p in plan["plans"])
        mark(4, "Plan next steps",
             (f"{len(plan['plans'])} candidate(s), {n_trials} open trial(s)"
              if plan["plans"] else "no actions — nothing scored"),
             t,
             {"trials_source": [p["trials_source"] for p in plan["plans"]],
              "shared_panel": bool(plan["shared_panels"])})

        total_ms = int((time.perf_counter() - t0) * 1000)

        return {
            "ok": bool(ranked),
            "total_ms": total_ms,
            "findings": [f.to_dict() for f in ex.findings],
            "phenotypes": [m.to_dict() for m in matches],
            "grounding_rate": round(rate, 3),
            "differential": [c.to_dict() for c in ranked],
            "actions": plan,
            "report_markdown": self._report(matches, ranked, plan),
            "log": [e.to_dict() for e in log],
        }

    # ---- the three-section clinician-facing report -----------------------

    def _report(self, matches, ranked, plan) -> str:
        L = ["## 1. Standardized phenotypes\n"]
        trusted = [m for m in matches if m.trusted]
        if trusted:
            L.append("| HPO code | Term | Matched via |")
            L.append("|---|---|---|")
            for m in trusted:
                L.append(f"| `{m.hpo_id}` | {m.label} | {m.method} |")
        else:
            L.append("*No findings could be mapped to HPO terms.*")

        unmatched = [m.query for m in matches if not m.trusted]
        if unmatched:
            L.append(f"\n*Not mapped to HPO ({len(unmatched)}): "
                     f"{', '.join(unmatched)}*")

        L.append("\n## 2. Differential\n")
        if ranked:
            L.append("| # | Suspected condition | Confidence | Score | Key overlap |")
            L.append("|---|---|---|---|---|")
            for i, c in enumerate(ranked, 1):
                top = ", ".join(m["label"] for m in c.matched[:3])
                L.append(f"| {i} | {c.name} | {c.confidence()} | {c.score:.2f} | {top} |")
            first = ranked[0]
            if first.missing_obligate:
                L.append(f"\n*Argues against {first.name}: "
                         f"{', '.join(m['label'] for m in first.missing_obligate[:3])} "
                         f"not documented.*")
        else:
            L.append("*No rare disease pattern detected above threshold.*")

        L.append("\n## 3. Next best clinical action\n")
        L.append(actions.render_markdown(plan) if plan["plans"]
                 else "*No action recommended — no candidate met the confidence threshold.*")
        return "\n".join(L)


# ------------------------------------------------------------- CLI --------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", help="id in app/data/demo_notes.json")
    ap.add_argument("--file")
    ap.add_argument("--text")
    ap.add_argument("--model")
    ap.add_argument("--no-trials", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    if a.case:
        notes = json.load(open(os.path.join(DATA, "demo_notes.json")))
        text = next(n for n in notes["notes"] if n["id"] == a.case)["text"]
    elif a.file:
        text = open(a.file).read()
    elif a.text:
        text = a.text
    else:
        text = sys.stdin.read()

    agent = Agent(model=a.model)
    print(f"[index built in {agent.startup_ms} ms]\n", file=sys.stderr)

    result = agent.run(text, with_trials=not a.no_trials)

    if a.json:
        print(json.dumps(result, indent=2))
        return 0

    for e in result["log"]:
        print(f"  [stage {e['stage']}] {e['name']:<20} {e['ms']:>6} ms   {e['detail']}",
              file=sys.stderr)
    print(f"  {'total':<32} {result['total_ms']:>6} ms\n", file=sys.stderr)
    print(result["report_markdown"])
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
