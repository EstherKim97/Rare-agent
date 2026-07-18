#!/usr/bin/env python3
"""
fetch_cases.py - Pull real published case reports from PMC Open Access.

Downloads full text via NCBI's BioC API and keeps ONLY the case presentation
sections, dropping abstract, discussion and conclusion.

Why the section filtering matters: a case report states its own answer in the
abstract and discussion. If you feed the whole article to the agent, you are
handing it the answer key and the evaluation is worthless. This script cuts the
article at the point where the authors stop describing and start explaining,
which is the closest thing to what a clinician actually has in front of them.

    API:      https://www.ncbi.nlm.nih.gov/research/bionlp/APIs/BioC-PMC/
    Template: .../pmcoa.cgi/BioC_json/[PMCID]/unicode
    Only covers the PMC Open Access Subset + Author Manuscript Collection.

LICENSING: PMC OA articles carry varying licenses (CC-BY, CC-BY-NC, CC-BY-NC-SA).
This script records the source URL for every file it writes, but it CANNOT
determine the license for you. Check each article's license statement and record
it in SOURCES.md before using anything in a public repo.

Usage:
    python scripts/fetch_cases.py --pmcid PMC10627660 --label fabry_pmr
    python scripts/fetch_cases.py --preset fabry
    python scripts/fetch_cases.py --pmcid PMC10627660 --debug   # dump section map
    python scripts/fetch_cases.py --preset fabry --redact "Fabry,alpha-galactosidase,GLA"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request

API = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode"

# BioC section_type values that describe the patient rather than explain them.
KEEP_SECTIONS = {"CASE", "PRESENTATION", "METHODS", "RESULTS"}
DROP_SECTIONS = {"TITLE", "ABSTRACT", "INTRO", "DISCUSS", "CONCL", "REF",
                 "ACK_FUND", "COMP_INT", "AUTH_CONT", "SUPPL", "APPENDIX",
                 "TABLE", "FIG", "REVIEW_INFO"}

# Fallback: match on the section heading text when section_type is unhelpful.
HEADING_KEEP = re.compile(
    r"case (report|presentation|description|history)|clinical (presentation|course|history)"
    r"|patient (presentation|description)|^history|^presentation|^examination",
    re.I,
)
HEADING_STOP = re.compile(r"^discussion|^conclusion|^comment|^literature review", re.I)

PRESETS = {
    # Curated diagnostic-odyssey case reports. Verify licenses individually.
    "fabry": [
        ("PMC10627660", "fabry_misdx_polymyalgia"),
        ("PMC12131641", "fabry_lateonset_esrd"),
        ("PMC8110900",  "fabry_central_asia"),
        ("PMC11621059", "fabry_misdx_hcm"),
    ],
}


def fetch_bioc(pmcid: str, timeout: int = 60) -> list:
    url = API.format(pmcid=pmcid)
    req = urllib.request.Request(
        url, headers={"User-Agent": "rare-disease-agent/hackathon (contact: local)"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode("utf-8", errors="replace")
    data = json.loads(raw)
    # The API returns either a list of collections or a single object.
    return data if isinstance(data, list) else [data]


def iter_passages(collections: list):
    for coll in collections:
        for doc in coll.get("documents", []):
            for p in doc.get("passages", []):
                yield p


def section_map(collections: list) -> list[tuple[str, str, int]]:
    """(section_type, heading-or-snippet, char_len) for every passage. For --debug."""
    out = []
    for p in iter_passages(collections):
        infons = p.get("infons", {})
        st = infons.get("section_type", "?")
        head = infons.get("title") or infons.get("type", "")
        text = p.get("text", "")
        out.append((st, str(head)[:40], len(text)))
    return out


def extract_case_text(collections: list) -> tuple[str, dict]:
    """Keep case-description passages, stop at the discussion."""
    kept, stats = [], {"kept": 0, "dropped": 0, "stopped_at": None}
    current_heading = ""
    stop = False

    for p in iter_passages(collections):
        infons = p.get("infons", {})
        st = (infons.get("section_type") or "").upper()
        ptype = (infons.get("type") or "").lower()
        text = (p.get("text") or "").strip()
        if not text:
            continue

        if ptype in ("title_1", "title_2", "title_3", "title"):
            current_heading = text
            if HEADING_STOP.search(text):
                stop = True
                stats["stopped_at"] = text
            continue

        if stop:
            stats["dropped"] += 1
            continue

        keep = False
        if st in KEEP_SECTIONS:
            keep = True
        elif st in DROP_SECTIONS:
            keep = False
        elif HEADING_KEEP.search(current_heading):
            keep = True

        # Heading match always wins - some journals tag everything as RESULTS.
        if HEADING_KEEP.search(current_heading):
            keep = True

        if keep and ptype == "paragraph":
            kept.append(text)
            stats["kept"] += 1
        else:
            stats["dropped"] += 1

    return "\n\n".join(kept), stats


def redact(text: str, terms: list[str]) -> tuple[str, int]:
    """Blank out answer-giving terms so the agent cannot read the diagnosis."""
    n = 0
    for t in [x.strip() for x in terms if x.strip()]:
        text, k = re.subn(re.escape(t), "[REDACTED]", text, flags=re.I)
        n += k
    return text, n


def process(pmcid: str, label: str, outdir: str, terms: list[str], debug: bool) -> bool:
    print(f"\n{pmcid}  ({label})")
    try:
        coll = fetch_bioc(pmcid)
    except Exception as e:
        print(f"  ERROR  fetch failed: {e}")
        print("  (article may not be in the PMC Open Access subset)")
        return False

    if debug:
        print("  section map:")
        for st, head, n in section_map(coll)[:60]:
            print(f"    {st:<14}{head:<42}{n:>6} chars")

    text, stats = extract_case_text(coll)
    if not text.strip():
        print("  ERROR  no case sections matched. Re-run with --debug and adjust "
              "KEEP_SECTIONS / HEADING_KEEP for this journal's tagging.")
        return False

    nred = 0
    if terms:
        text, nred = redact(text, terms)

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{pmcid}_{label}.txt")
    header = (f"# SOURCE: https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/\n"
              f"# Retrieved via NCBI BioC API. Case-description sections only;\n"
              f"# abstract, discussion and conclusion removed.\n"
              f"# VERIFY THE LICENSE before redistributing.\n\n")
    open(path, "w").write(header + text + "\n")

    words = len(text.split())
    print(f"  kept {stats['kept']} passages, dropped {stats['dropped']}"
          + (f", stopped at '{stats['stopped_at']}'" if stats["stopped_at"] else ""))
    if nred:
        print(f"  redacted {nred} answer-giving mentions")
    print(f"  wrote {path}  ({words} words)")
    if words < 120:
        print("  WARN  suspiciously short - check with --debug that sections matched")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmcid", help="e.g. PMC10627660")
    ap.add_argument("--label", default="case")
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--outdir", default="app/data/real_cases")
    ap.add_argument("--redact", default="", help="comma-separated terms to blank out")
    ap.add_argument("--debug", action="store_true", help="print the section map")
    a = ap.parse_args()

    terms = a.redact.split(",") if a.redact else []
    jobs = PRESETS[a.preset] if a.preset else ([(a.pmcid, a.label)] if a.pmcid else [])
    if not jobs:
        ap.error("give --pmcid or --preset")

    ok = sum(process(p, l, a.outdir, terms, a.debug) for p, l in jobs)
    print(f"\n{ok}/{len(jobs)} retrieved into {a.outdir}")
    if ok:
        print("\nNext:")
        print(f"  python app/stages/extract.py --file {a.outdir}/<file>.txt")
        print("  Record each source URL and license in SOURCES.md")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
