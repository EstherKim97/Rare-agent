#!/usr/bin/env python3
"""
extract.py - Stage 1 of the Rare Disease Diagnostic Agent.

Turns a raw clinical note or ambient transcript into candidate phenotype
strings that Stage 2 can ground to HPO codes.

This is the ONLY stage where an LLM makes a judgement call, and its job is
narrower than it looks. It does NOT diagnose. It does NOT decide what is
important. It performs one task: read messy prose and re-express each
observed finding in standard clinical terminology.

    "he's tall and skinny with really long fingers"
        -> ["Disproportionate tall stature", "Arachnodactyly"]

That translation step is worth measuring. Benchmarked on case_01, emitting
clinical terminology grounds 15/15 terms (100%) against HPO, while echoing the
note's lay phrasing grounds 5/13 (38%). Same grounder, same note. The prompt
below is written to force the former.

Three failure modes the prompt explicitly guards against:

  NEGATION      "Denies seizures" is not a seizure. Negated findings are
                captured with negated=true and dropped before scoring.
  ATTRIBUTION   "his mother has kidney problems" is family history, not a
                patient finding. Kept separately - it informs inheritance
                but must never enter the patient's phenotype set.
  INFERENCE     The model must not write "Marfan syndrome" or "storage
                disorder". Findings only. Diagnosis is Stage 3's job, and
                Stage 3 is arithmetic.

Usage:
    python app/stages/extract.py --case case_01          # end-to-end acceptance test
    python app/stages/extract.py --text "$(cat note.txt)"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict

from anthropic import Anthropic

from dotenv import load_dotenv
load_dotenv()

DEFAULT_MODEL = "claude-haiku-4-5-20251001" #"claude-sonnet-5"

# The tool schema is the contract. The model cannot return prose; it must
# fill these fields, which is what makes Stage 1 output machine-checkable.
EXTRACT_TOOL = {
    "name": "record_findings",
    "description": "Record every clinical finding observed in the document.",
    "input_schema": {
        "type": "object",
        "properties": {
            "findings": {
                "type": "array",
                "description": "One entry per distinct clinical finding.",
                "items": {
                    "type": "object",
                    "properties": {
                        "clinical_term": {
                            "type": "string",
                            "description": (
                                "The finding in STANDARD CLINICAL TERMINOLOGY, as it "
                                "would appear in the Human Phenotype Ontology. Not the "
                                "patient's words. E.g. 'Arachnodactyly', not 'long "
                                "fingers'; 'Hypohidrosis', not 'never sweats'."
                            ),
                        },
                        "evidence": {
                            "type": "string",
                            "description": "Short verbatim span from the document supporting this finding.",
                        },
                        "organ_system": {
                            "type": "string",
                            "enum": ["cardiovascular", "renal", "neurologic", "ocular",
                                     "cutaneous", "musculoskeletal", "gastrointestinal",
                                     "hematologic", "endocrine", "respiratory",
                                     "auditory", "immunologic", "psychiatric",
                                     "craniofacial", "constitutional", "other"],
                        },
                        "negated": {
                            "type": "boolean",
                            "description": "True if the document states this finding is ABSENT.",
                        },
                        "subject": {
                            "type": "string",
                            "enum": ["patient", "family"],
                            "description": "Who the finding belongs to. Family history is not a patient finding.",
                        },
                        "certainty": {
                            "type": "string",
                            "enum": ["definite", "probable", "possible"],
                        },
                        "onset": {
                            "type": "string",
                            "description": "Age or timing if stated, else empty string.",
                        },
                    },
                    "required": ["clinical_term", "evidence", "organ_system",
                                 "negated", "subject", "certainty", "onset"],
                },
            }
        },
        "required": ["findings"],
    },
}

SYSTEM_PROMPT = """You are a clinical phenotyping assistant. You read clinical \
documents and re-express observed findings in standard clinical terminology so they \
can be mapped to the Human Phenotype Ontology (HPO).

You are NOT diagnosing. Never output a disease name, syndrome name, or diagnostic \
speculation as a finding. Output only observable phenotypic findings.

TRANSLATE, DO NOT ECHO. This is your single most important behaviour. Convert lay \
and colloquial descriptions into the standard clinical term an HPO curator would use:
  "long fingers", "spider fingers"            -> Arachnodactyly
  "tall and thin", "all arms and legs"        -> Disproportionate tall stature
  "never sweats", "doesn't sweat much"        -> Hypohidrosis
  "burning pain in hands and feet"            -> Acroparesthesia
  "gets overheated easily"                    -> Heat intolerance
  "dark red spots on the trunk/groin"         -> Angiokeratoma
  "protein in the urine", "2+ protein on UA"  -> Proteinuria
  "thickened heart muscle", "LVH on echo"     -> Left ventricular hypertrophy
  "mini-stroke", "TIA"                        -> Transient ischemic attack
  "ringing in the ears"                       -> Tinnitus
  "curved spine", "curvature on forward bend" -> Scoliosis
  "sunken chest"                              -> Pectus excavatum
  "flat feet"                                 -> Pes planus
  "lyso-Gb3", "lyso-GL-3", "Lyso-GL3"         -> Elevated circulating lyso-globotriaosylsphingosine concentration
  "enlarged liver" / "enlarged spleen"        -> Hepatomegaly / Splenomegaly
  "loose stools"                              -> Diarrhea
  "nosebleeds"                                -> Epistaxis
  "double-jointed", "very flexible joints"    -> Joint hypermobility
  "nearsighted", "wears glasses for distance" -> Myopia

Prefer the most specific term the document actually supports. If the note says \
"chest wall deformity" without specifying, do NOT guess between pectus excavatum and \
carinatum - use the general term. Never invent specificity that is not in the text.

RULES:
1. NEGATION. "Denies seizures", "no rash", "ANA negative" are findings with \
negated=true. Record them - a confirmed ABSENT finding is diagnostically valuable - \
but mark them correctly.
2. ATTRIBUTION. Findings in relatives get subject="family". "His mother has kidney \
problems" is family history. Never mark it subject="patient".
3. NO DIAGNOSES AS FINDINGS. Prior diagnostic LABELS the patient was given \
(fibromyalgia, IBS) are not phenotypes. Skip them. Extract the underlying findings \
instead if they are described.
4. NORMAL RESULTS. Skip routine normal findings and unremarkable results unless the \
normality is diagnostically notable (e.g. "no murmur" in a cardiac workup).
5. LAB VALUES. Convert to the phenotype they represent. "Cr 1.4, eGFR 62" -> Renal \
insufficiency. "UPCR 1.4 g/g" -> Proteinuria. "HbA1c 5.2" is normal - skip it.
6. GRANULARITY. One finding per entry. Do not bundle "burning pain and heat \
intolerance" into one entry.
7. SOCIAL/ADMIN. Ignore social determinants, insurance, scheduling, and lifestyle \
counselling unless they describe a physical finding.
8. RELEVANCE. Mark certainty="definite" for findings that are unexplained, \
part of the reason for referral, or that the clinician is actively working up. \
Mark certainty="possible" for incidental or long-standing comorbidities that \
already have an established explanation (e.g. reflux esophagitis, degenerative \
joint disease, known coronary disease). Still record them - just mark them.

Be thorough. A missed finding cannot be recovered downstream."""


@dataclass
class Finding:
    clinical_term: str
    evidence: str
    organ_system: str
    negated: bool
    subject: str
    certainty: str
    onset: str

    def to_dict(self):
        return asdict(self)


@dataclass
class Extraction:
    findings: list[Finding]
    model: str
    input_tokens: int
    output_tokens: int

    def patient_positive(self) -> list[Finding]:
        """The only findings that may enter scoring."""
        return [f for f in self.findings
                if not f.negated
                and f.subject == "patient"
                and f.certainty != "possible"]

    def family(self) -> list[Finding]:
        return [f for f in self.findings if f.subject == "family"]

    def negated(self) -> list[Finding]:
        return [f for f in self.findings if f.negated and f.subject == "patient"]

    def to_dict(self):
        return {
            "model": self.model,
            "counts": {
                "total": len(self.findings),
                "patient_positive": len(self.patient_positive()),
                "patient_negated": len(self.negated()),
                "family_history": len(self.family()),
            },
            "tokens": {"in": self.input_tokens, "out": self.output_tokens},
            "findings": [f.to_dict() for f in self.findings],
        }


def extract(text: str, model: str = DEFAULT_MODEL, client: Anthropic | None = None) -> Extraction:
    """Run Stage 1. Raises on API failure - let the caller decide about fallbacks."""
    client = client or Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    resp = client.messages.create(
        model=model,
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        tools=[EXTRACT_TOOL],
        tool_choice={"type": "tool", "name": "record_findings"},
        messages=[{
            "role": "user",
            "content": (
                "Extract every clinical finding from this document. Remember: "
                "standard clinical terminology, not the document's phrasing.\n\n"
                f"<document>\n{text}\n</document>"
            ),
        }],
    )

    findings: list[Finding] = []
    for block in resp.content:
        if block.type == "tool_use" and block.name == "record_findings":
            for raw in block.input.get("findings", []):
                try:
                    findings.append(Finding(
                        clinical_term=raw["clinical_term"].strip(),
                        evidence=raw.get("evidence", "").strip(),
                        organ_system=raw.get("organ_system", "other"),
                        negated=bool(raw.get("negated", False)),
                        subject=raw.get("subject", "patient"),
                        certainty=raw.get("certainty", "definite"),
                        onset=raw.get("onset", ""),
                    ))
                except KeyError:
                    continue  # malformed entry - drop it rather than crash

    return Extraction(findings, model, resp.usage.input_tokens, resp.usage.output_tokens)


# ------------------------------------------------- acceptance test --------

def _acceptance(text: str, expect: str | None, model: str) -> int:
    """End-to-end: extract -> ground -> score. Returns exit code."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ground import HPOIndex
    from differential import score_diseases, load_db

    here = os.path.dirname(os.path.abspath(__file__))
    data = os.path.join(here, "..", "data")

    print("STAGE 1  extracting...")
    ex = extract(text, model)
    pos = ex.patient_positive()
    print(f"  {len(ex.findings)} findings | {len(pos)} patient-positive "
          f"| {len(ex.negated())} negated | {len(ex.family())} family")
    print(f"  tokens in/out: {ex.input_tokens}/{ex.output_tokens}\n")

    print("STAGE 2  grounding...")
    hpo = HPOIndex(os.path.join(data, "hp.json"))
    matches = hpo.ground_all([f.clinical_term for f in pos])
    trusted = [m for m in matches if m.trusted]
    rate = len(trusted) / max(len(matches), 1)
    print(f"  {len(trusted)}/{len(matches)} grounded at trusted=True  ({rate:.0%})")
    for m in matches:
        if not m.trusted:
            print(f"    MISS  {m.query}  ({m.method})")
    print()

    print("STAGE 3  ranking...")
    db = load_db(os.path.join(data, "rare_diseases.json"))
    ranked = score_diseases([m.hpo_id for m in trusted], db, hpo)
    for i, c in enumerate(ranked, 1):
        print(f"  {i}. {c.name:<46} {c.score:.3f}  {c.confidence()}")

    print()
    ok = True
    if rate < 0.80:
        print(f"  FAIL  grounding rate {rate:.0%} below 80% target")
        ok = False
    else:
        print(f"  PASS  grounding rate {rate:.0%}")

    if expect:
        hit = ranked and expect.lower() in ranked[0].name.lower()
        in3 = any(expect.lower() in c.name.lower() for c in ranked[:3])
        if hit:
            print(f"  PASS  expected diagnosis ranked #1 ({expect})")
        elif in3:
            print(f"  WARN  expected diagnosis in top 3 but not #1 ({expect})")
        else:
            print(f"  FAIL  expected diagnosis not in top 3 ({expect})")
            ok = False
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", help="id from app/data/demo_notes.json, e.g. case_01")
    ap.add_argument("--text", help="raw note text")
    ap.add_argument("--file", help="path to a text file")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--json", action="store_true", help="print extraction JSON only")
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    expect = None

    if a.case:
        notes = json.load(open(os.path.join(here, "..", "data", "demo_notes.json")))
        rec = next(n for n in notes["notes"] if n["id"] == a.case)
        text, expect = rec["text"], rec["expected_diagnosis"].split(" (")[0]
    elif a.file:
        text = open(a.file).read()
    elif a.text:
        text = a.text
    else:
        text = sys.stdin.read()

    if a.json:
        print(json.dumps(extract(text, a.model).to_dict(), indent=2))
        return 0
    return _acceptance(text, expect, a.model)


if __name__ == "__main__":
    sys.exit(main())
