#!/usr/bin/env python3
"""
main.py - Web server for the Rare Disease Diagnostic Agent.

Serves one page and one endpoint. The endpoint streams Server-Sent Events so
the four stages appear in the browser as they finish, instead of the page
freezing for ten seconds and then dumping everything at once.

That is not cosmetic. This agent's credibility rests on being a visible
multi-step pipeline with deterministic grounding, and a spinner communicates
none of that. Streaming the stage log is how the architecture becomes legible
to someone watching a three-minute demo.

    GET  /             the page
    POST /analyze      SSE: stage_begin / stage_end / result / error
    GET  /example      a bundled case for the Load example button
    GET  /health       index sizes and readiness

Run from the project root:
    uvicorn app.main:app --port 8000
"""

from __future__ import annotations

import json
import os
import time
import traceback

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from pydantic import BaseModel

from .pipeline import Agent
from .stages import actions
from .stages.differential import score_diseases
from .stages import extract as extract_mod

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")
DATA = os.path.join(HERE, "data")

app = FastAPI(title="Rare Disease Diagnostic Agent")

# Built once at import. HPOIndex takes ~3s; building it per request would make
# every analysis feel three seconds slower than it is.
AGENT = Agent()


class AnalyzeRequest(BaseModel):
    text: str
    with_trials: bool = True


def sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def run_stream(text: str, with_trials: bool):
    """Yield SSE frames as each stage completes."""
    t0 = time.perf_counter()
    ms = lambda since: int((time.perf_counter() - since) * 1000)

    try:
        yield sse("start", {
            "chars": len(text),
            "curated": len(AGENT.db["diseases"]),
            "broad": len(AGENT.broad["diseases"]) if getattr(AGENT, "broad", None) else 0,
        })

        # ---- Stage 1 -----------------------------------------------------
        yield sse("stage_begin", {"stage": 1, "name": "Reading the note",
                                  "note": "Claude restates findings in clinical terms"})
        t = time.perf_counter()
        ex = extract_mod.extract(text, AGENT.model)
        positives = ex.patient_positive()
        yield sse("stage_end", {
            "stage": 1, "ms": ms(t),
            "detail": f"{len(ex.findings)} findings — {len(positives)} to score, "
                      f"{len(ex.negated())} ruled out, {len(ex.family())} family history",
            "data": {"findings": [f.to_dict() for f in positives],
                     "negated": [f.clinical_term for f in ex.negated()],
                     "family": [f.clinical_term for f in ex.family()],
                     "model": ex.model,
                     "tokens": ex.input_tokens + ex.output_tokens}})

        # ---- Stage 2 -----------------------------------------------------
        yield sse("stage_begin", {"stage": 2, "name": "Mapping to HPO codes",
                                  "note": "Deterministic lookup — no model involved"})
        t = time.perf_counter()
        matches = AGENT.hpo.ground_all([f.clinical_term for f in positives])
        trusted = [m for m in matches if m.trusted]
        rate = len(trusted) / max(len(matches), 1)
        yield sse("stage_end", {
            "stage": 2, "ms": ms(t),
            "detail": f"{len(trusted)} of {len(matches)} mapped to verified codes ({rate:.0%})",
            "data": {"matches": [m.to_dict() for m in matches], "rate": round(rate, 3)}})

        # ---- Stage 3 -----------------------------------------------------
        yield sse("stage_begin", {"stage": 3, "name": "Ranking the differential",
                                  "note": "Weighted phenotype overlap — arithmetic, not inference"})
        t = time.perf_counter()
        ids = [m.hpo_id for m in trusted]
        ranked = score_diseases(ids, AGENT.db, AGENT.hpo, top_n=5)
        tier, searched = "curated", len(AGENT.db["diseases"])
        broad = getattr(AGENT, "broad", None)
        if broad and (not ranked or ranked[0].score < 0.0):
            wide = score_diseases(ids, broad, AGENT.hpo, top_n=5, ic=getattr(AGENT, "ic", None))
            if wide and (not ranked or wide[0].score > ranked[0].score):
                ranked, tier, searched = wide, "broad", len(broad["diseases"])
        yield sse("stage_end", {
            "stage": 3, "ms": ms(t),
            "detail": (f"{ranked[0].name} leads across {searched:,} diseases"
                       if ranked else "nothing scored above threshold"),
            "data": {"tier": tier, "searched": searched,
                     "candidates": [c.to_dict() for c in ranked]}})

        # ---- Stage 4 -----------------------------------------------------
        yield sse("stage_begin", {"stage": 4, "name": "Planning next steps",
                                  "note": "Panels from curated data, trials from ClinicalTrials.gov"})
        t = time.perf_counter()
        if ranked and tier == "curated":
            plan = actions.plan_actions(ranked, AGENT.db, top_n=3, with_trials=with_trials)
        else:
            plan = {"plans": [], "shared_panels": [], "urgent_flags": [],
                    "single_order_resolves": False}
        n_trials = sum(len(p["trials"]) for p in plan["plans"])
        yield sse("stage_end", {
            "stage": 4, "ms": ms(t),
            "detail": (f"{len(plan['plans'])} workup path(s), {n_trials} open trial(s)"
                       if plan["plans"] else
                       "no verified workup — top candidate is outside the curated set"),
            "data": {"trials": n_trials}})

        yield sse("result", {
            "ok": bool(ranked),
            "total_ms": ms(t0),
            "tier": tier,
            "searched": searched,
            "grounding_rate": round(rate, 3),
            "phenotypes": [m.to_dict() for m in matches],
            "differential": [c.to_dict() for c in ranked],
            "actions": plan,
            "negated": [f.clinical_term for f in ex.negated()],
            "family": [f.clinical_term for f in ex.family()]})

    except Exception as e:
        yield sse("error", {"message": f"{type(e).__name__}: {e}",
                            "trace": traceback.format_exc()[-800:]})


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    if not req.text.strip():
        return JSONResponse({"error": "Paste a clinical note first."}, status_code=400)
    return StreamingResponse(
        run_stream(req.text, req.with_trials),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/example")
def example():
    cases = os.path.join(DATA, "real_cases")
    if os.path.isdir(cases):
        for fn in sorted(os.listdir(cases)):
            if fn.endswith(".txt"):
                raw = open(os.path.join(cases, fn)).read()
                body = "\n".join(l for l in raw.splitlines() if not l.startswith("#"))
                return {"label": fn.replace(".txt", ""),
                        "kind": "Published case report — diagnosis redacted",
                        "text": body.strip()}
    demo = os.path.join(DATA, "demo_notes.json")
    if os.path.exists(demo):
        n = json.load(open(demo))["notes"][0]
        return {"label": n["label"], "kind": "Synthetic note", "text": n["text"]}
    return JSONResponse({"error": "No bundled examples found."}, status_code=404)


@app.get("/health")
def health():
    broad = getattr(AGENT, "broad", None)
    return {"ok": True,
            "hpo_terms": len(AGENT.hpo.label_by_id),
            "curated_diseases": len(AGENT.db["diseases"]),
            "broad_diseases": len(broad["diseases"]) if broad else 0,
            "model": AGENT.model,
            "startup_ms": AGENT.startup_ms}


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))
