#!/usr/bin/env python3
"""
ground.py - Stage 2 of the Rare Disease Diagnostic Agent.

Maps free-text symptom strings to canonical Human Phenotype Ontology terms.

THERE IS NO LLM IN THIS FILE. That is the point. Stage 1 (Claude) is allowed to
read a messy note and say "his fingers are really long". This module decides
whether that is HP:0001166 Arachnodactyly. Because the decision is made by a
deterministic lookup against the official HPO release, the agent structurally
cannot invent a phenotype code - it can only select one that exists.

Matching is tried in descending order of confidence:
    1. exact      - the string is a canonical HPO label
    2. synonym    - the string is a registered HPO synonym
    3. normalized - matches after lowercasing / depluralising / de-punctuating
    4. fuzzy      - rapidfuzz similarity above threshold (reported, not trusted)
    5. none       - unmatched, surfaced honestly rather than guessed

Also exposes ancestors() so Stage 3 can give partial credit when a note says
"heart murmur" and the disease is annotated with a more specific cardiac term.

Usage as a library:
    from ground import HPOIndex
    hpo = HPOIndex("app/data/hp.json")
    m = hpo.ground("long slender fingers")
    # Match(hpo_id='HP:0001166', label='Arachnodactyly', method='synonym', score=1.0)

Usage as a self-test:
    python app/stages/ground.py --hp app/data/hp.json --test
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from functools import lru_cache

from rapidfuzz import process, fuzz

# Words that add no diagnostic signal and hurt fuzzy matching.
_STOPWORDS = {
    "the", "a", "an", "of", "with", "and", "his", "her", "their", "patient",
    "patients", "has", "have", "had", "some", "mild", "moderate", "severe",
    "chronic", "acute", "history", "reports", "reported", "noted", "shows",
}

# Clinical shorthand that will never appear in the ontology as written.
_ALIASES = {
    "sob": "dyspnea",
    "doe": "exertional dyspnea",
    "loc": "loss of consciousness",
    "n/v": "nausea and vomiting",
    "ck": "creatine kinase",
    "lvh": "left ventricular hypertrophy",
    "htn": "hypertension",
    "dm": "diabetes mellitus",
    "ckd": "chronic kidney disease",
    "tia": "transient ischemic attack",
    "gi bleed": "gastrointestinal hemorrhage",
    "nose bleeds": "epistaxis",
    "nosebleeds": "epistaxis",
    "hearing loss": "sensorineural hearing impairment",
    "tall and thin": "disproportionate tall stature",
    "long fingers": "arachnodactyly",
    "curved spine": "scoliosis",
    "sunken chest": "pectus excavatum",
    "enlarged liver": "hepatomegaly",
    "enlarged spleen": "splenomegaly",
    "protein in urine": "proteinuria",
    "blood in urine": "hematuria",
}


@dataclass
class Match:
    """One grounding decision, carrying its own provenance."""
    query: str
    hpo_id: str | None
    label: str | None
    method: str          # exact | synonym | normalized | alias | fuzzy | none
    score: float         # 1.0 for deterministic hits, 0-1 for fuzzy
    trusted: bool        # False for fuzzy - show it, but flag it in the UI

    def to_dict(self) -> dict:
        return asdict(self)


def normalize(text: str) -> str:
    """Lowercase, strip punctuation and stopwords, crudely singularize."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s/+-]", " ", text)
    tokens = []
    for tok in text.split():
        if tok in _STOPWORDS:
            continue
        if len(tok) > 4 and tok.endswith("ies"):
            tok = tok[:-3] + "y"
        elif len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        tokens.append(tok)
    return " ".join(tokens)


class HPOIndex:
    """In-memory index over an HPO release. Build once at app startup (~3s)."""

    def __init__(self, hp_json_path: str, fuzzy_threshold: float = 88.0):
        self.fuzzy_threshold = fuzzy_threshold
        self.label_by_id: dict[str, str] = {}
        self._exact: dict[str, str] = {}       # canonical label -> id
        self._synonym: dict[str, str] = {}     # synonym         -> id
        self._normalized: dict[str, str] = {}  # normalized form -> id
        self._parents: dict[str, list[str]] = {}

        graph = json.load(open(hp_json_path))["graphs"][0]

        for node in graph["nodes"]:
            nid = node.get("id", "")
            if "/HP_" not in nid or not node.get("lbl"):
                continue
            meta = node.get("meta", {})
            if meta.get("deprecated"):
                continue

            hp_id = nid.split("/")[-1].replace("_", ":")
            label = node["lbl"]
            self.label_by_id[hp_id] = label

            self._exact.setdefault(label.lower(), hp_id)
            self._normalized.setdefault(normalize(label), hp_id)

            for syn in meta.get("synonyms", []):
                val = syn.get("val", "")
                if not val:
                    continue
                self._synonym.setdefault(val.lower(), hp_id)
                self._normalized.setdefault(normalize(val), hp_id)

        for edge in graph["edges"]:
            if edge.get("pred") != "is_a":
                continue
            sub = edge["sub"].split("/")[-1].replace("_", ":")
            obj = edge["obj"].split("/")[-1].replace("_", ":")
            if sub in self.label_by_id and obj in self.label_by_id:
                self._parents.setdefault(sub, []).append(obj)

        # Frozen key list for fuzzy search.
        self._fuzzy_keys = list(self._normalized.keys())

    # -- grounding ---------------------------------------------------------

    def ground(self, text: str) -> Match:
        """Resolve one free-text symptom to an HPO term."""
        raw = text.strip()
        low = raw.lower()

        # A raw HP id passes straight through.
        if raw.upper() in self.label_by_id:
            hid = raw.upper()
            return Match(raw, hid, self.label_by_id[hid], "exact", 1.0, True)

        if low in self._exact:
            hid = self._exact[low]
            return Match(raw, hid, self.label_by_id[hid], "exact", 1.0, True)

        if low in self._synonym:
            hid = self._synonym[low]
            return Match(raw, hid, self.label_by_id[hid], "synonym", 1.0, True)

        if low in _ALIASES:
            expanded = _ALIASES[low]
            hid = self._exact.get(expanded) or self._normalized.get(normalize(expanded))
            if hid:
                return Match(raw, hid, self.label_by_id[hid], "alias", 1.0, True)

        norm = normalize(raw)
        if norm in self._normalized:
            hid = self._normalized[norm]
            return Match(raw, hid, self.label_by_id[hid], "normalized", 1.0, True)

        # Fuzzy is a last resort. WRatio will happily score a short key like
        # "pain" at 90 against "burning pain in hands and feet" because it is a
        # substring, which grounds specific complaints to useless generic terms.
        # Take several candidates and require comparable length before trusting.
        for key, score, _ in process.extract(
            norm, self._fuzzy_keys, scorer=fuzz.WRatio,
            score_cutoff=self.fuzzy_threshold, limit=5,
        ):
            ratio = len(key) / max(len(norm), 1)
            if ratio < 0.6 and score < 97:
                continue  # short-key substring inflation - skip it
            hid = self._normalized[key]
            return Match(raw, hid, self.label_by_id[hid], "fuzzy", round(score / 100, 3), False)

        return Match(raw, None, None, "none", 0.0, False)

    def ground_all(self, texts: list[str]) -> list[Match]:
        """Ground a list, dropping duplicate HPO ids but keeping unmatched items."""
        seen, out = set(), []
        for t in texts:
            m = self.ground(t)
            if m.hpo_id and m.hpo_id in seen:
                continue
            if m.hpo_id:
                seen.add(m.hpo_id)
            out.append(m)
        return out

    # -- ontology traversal ------------------------------------------------

    @lru_cache(maxsize=8192)
    def ancestors(self, hpo_id: str, max_depth: int = 2) -> tuple[str, ...]:
        """Ancestor ids up to max_depth. Used for partial-credit scoring."""
        seen: set[str] = set()
        frontier = [hpo_id]
        for _ in range(max_depth):
            nxt = []
            for node in frontier:
                for parent in self._parents.get(node, []):
                    if parent not in seen:
                        seen.add(parent)
                        nxt.append(parent)
            frontier = nxt
            if not frontier:
                break
        return tuple(seen)

    def related(self, a: str, b: str, max_depth: int = 2) -> bool:
        """True if a and b are the same term or within max_depth of each other."""
        if a == b:
            return True
        return b in self.ancestors(a, max_depth) or a in self.ancestors(b, max_depth)

    def stats(self) -> dict:
        return {
            "terms": len(self.label_by_id),
            "labels": len(self._exact),
            "synonyms": len(self._synonym),
            "searchable_strings": len(self._normalized),
            "is_a_edges": sum(len(v) for v in self._parents.values()),
        }


# -- self test -------------------------------------------------------------

_TESTS = [
    "Arachnodactyly",                    # exact
    "long slender fingers",              # synonym
    "Long fingers",                      # alias
    "spider fingers",                    # synonym
    "aortic root aneurysm",              # exact, lowercase
    "dislocated lenses",                 # fuzzy / synonym
    "recurrent nosebleeds",              # alias-ish
    "burning pain in hands and feet",    # hard - expect fuzzy or none
    "whorl-like corneal opacity",        # hard - cornea verticillata
    "his skin was unusually translucent",# noisy phrasing
    "Kayser-Fleischer ring",             # exact
    "enlarged spleen",                   # alias
    "seizures",                          # plural
    "purple monkey dishwasher",          # must return none
]


def _self_test(path: str) -> None:
    hpo = HPOIndex(path)
    s = hpo.stats()
    print(f"index: {s['terms']:,} terms | {s['searchable_strings']:,} searchable "
          f"| {s['is_a_edges']:,} is_a edges\n")
    width = max(len(t) for t in _TESTS)
    for t in _TESTS:
        m = hpo.ground(t)
        flag = "" if m.trusted else "  <-- unverified"
        got = f"{m.hpo_id} {m.label}" if m.hpo_id else "NO MATCH"
        print(f"  {t:<{width}}  {m.method:<11} {m.score:<6} {got}{flag}")

    print("\nancestor walk for HP:0001166 (Arachnodactyly), depth 2:")
    for a in list(hpo.ancestors("HP:0001166", 2))[:6]:
        print(f"  {a}  {hpo.label_by_id[a]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hp", default="app/data/hp.json")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--query", help="ground a single string and exit")
    a = ap.parse_args()

    if a.query:
        print(json.dumps(HPOIndex(a.hp).ground(a.query).to_dict(), indent=2))
    else:
        _self_test(a.hp)
