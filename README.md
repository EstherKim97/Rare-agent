# Phenotype Agent

Rare disease patients wait five to seven years for a diagnosis. The striking part
of that number is that it rarely reflects an unknowable answer. It reflects clues
spread across four different clinicians' notes over a decade. The cardiologist
saw a thick heart wall. The nephrologist saw protein in the urine. A
rheumatologist saw joint pain and recorded fibromyalgia. Each reading was
reasonable in isolation. Nothing ever placed the three findings on the same page.

Phenotype Agent puts them on the same page. Given a clinical note, it returns
standardized phenotypes, a ranked differential, and the specific diagnostic test
to order next.

Built in one day for the Abridge × Anthropic × Lightspeed hackathon, July 2026.

---

## The decision the system rests on

**Claude reads. It never decides.**

Stage 1 is the only place a language model touches the problem, and its role is
narrower than it first appears: read messy prose and re-express each finding in
standard clinical terminology. "Burning pain in his hands since he was a kid"
becomes `Acroparesthesia`. The prompt explicitly forbids naming a disease.

Everything after that is a lookup:

```
clinical note
  → Stage 1  extract.py       Claude → findings, in clinical terms      [LLM]
  → Stage 2  ground.py        findings → HPO codes                      [no LLM]
  → Stage 3  differential.py  codes → ranked diseases                   [no LLM]
  → Stage 4  actions.py       diseases → tests, red flags, trials       [no LLM]
```

The practical consequence is that the system *structurally* cannot invent a
phenotype code, a gene name, or a clinical trial. This is not a matter of
prompting against it — there is no path to it. Stage 2 can only return terms
present in the ontology file. Stage 4 reads gene panels verbatim from a curated
JSON. That property is what makes displaying a gene panel defensible at all.

---

## Evaluation

Testing only on self-authored inputs would prove very little, so the ranking
engine is benchmarked against **phenopacket-store**: 669 real published patients
with confirmed molecular diagnoses, curated by the Monarch Initiative from the
literature.

**72.3% top-1. 91.2% top-3.** The median case carries only five phenotypes.

A benchmark result nobody has attacked is weak evidence, so the number was
stress-tested four ways:

- **Baseline.** Always guessing the most common disease scores 44.5%. Without
  this check, nearly half the headline number would have been unexamined class
  imbalance.
- **Permutation test.** Shuffling which phenotype set belongs to which disease
  collapses accuracy to 0.1–4.0%, confirming the signal lives in the mapping
  rather than the experimental setup.
- **Leakage.** Only 49% of each benchmark case's terms appear in the curated set.
  A figure near 100% would have made the evaluation circular.
- **Ablation.** Removing the ontology ancestor walk costs 3.7 points, so that
  component earns its place rather than being architecture for its own sake.

One point deserves precision: phenopacket phenotypes arrive already HPO-coded, so
this benchmarks **Stage 3 in isolation**. It measures neither extraction nor
grounding. Describing the system as "91.2% accurate" would overstate the result.

### Specificity

The more important risk is not sensitivity but false positives. A tool that cries
wolf across a health system's notes will be switched off within a week, and
should be.

Common primary-care findings — obesity, hypertension, hyperlipidemia, pain —
score **0.075** against a 0.55 confidence threshold. Abridge's 25 synthetic
encounters, spanning annual physicals, prenatal intakes, and skilled-nursing
admissions, produce no confident rare-disease call. Both behaviors are enforced
by automated tests rather than assumed: `python scripts/test_differential.py`
runs 18 checks covering scoring correctness, clinical behavior, false-positive
control, and benchmark validity.

---

## What went wrong, and what it changed

**Curating by label rather than by ID.** The first build failed on two terms
because HPO had renamed them between releases — "Elevated…level" became
"Increased…concentration". HPO IDs are permanent; display strings are not. The
build script now accepts raw `HP:` identifiers and exits non-zero rather than
silently shipping a code that does not exist.

**Over-redacting the evidence.** Testing against published case reports requires
stripping the answer from the text first, or the exercise only measures reading
comprehension. An initial pass blanked every mention of the disease *and* its
enzyme, which removed "reduced α-galactosidase A activity" — the most diagnostic
line in the paper. The system performed reasonably given what remained, but what
remained was considerably less than a real clinician would have had.

**More diseases made results worse.** A broad tier generated from all 12,717
HPOA diseases returned four near-identical hypertrophic cardiomyopathy subtypes
and a pleural mesothelioma. Diseases carrying only three to five annotations
score spuriously high: matching two of four yields 50% coverage. Filtering to a
minimum of ten annotations (8,213 diseases) corrected the ranking. The fix was a
threshold, not more data.

**A standard technique helped in one place and hurt in another.**
Information-content weighting — weighting each HPO term by how many diseases carry
it — is common practice in phenotype matching. `Anemia` appears in 468 diseases;
`cornea verticillata` in two. Benchmarking produced an unexpected split:

- Curated tier (15 diseases): no measurable improvement, and a slight decline in top-1.
- Broad tier (8,213 diseases): moves the correct disease from #2 to #1.

The explanation follows from scale. Among 15 hand-picked, well-separated
diseases there is little common-term confusion to correct, so the weighting adds
only variance. Across 8,213, terms like "chest pain" and "anemia" dominate, and
information content is the only thing distinguishing a real match from a
coincidence. It therefore ships to the broad tier alone. The measurement was more
valuable than a better headline number would have been.

---

## Limitations

- **Case reports are not clinical notes.** Validation uses published PMC case
  reports: real patients and real physician writing, but composed retrospectively
  by authors who already knew the diagnosis. Real documentation is fragmentary and
  contradictory, and the grounding rate reported here should be treated as
  optimistic.
- **Atypical presentations defeat it.** Late-onset cardiac Fabry presenting as
  polymyalgia rheumatica in a 79-year-old lacks every classic feature, and the
  system ranks a cardiomyopathy first. This is the genuine ceiling of phenotype
  matching, and it is the same ceiling clinicians encounter — which is precisely
  why that patient went undiagnosed for years.
- **Uneven performance across curated diseases.** Eight of the fifteen have cases in
  the benchmark; the remainder are validated only by construction. Performance also
  varies significantly: Marfan and hereditary hemorrhagic telangiectasia reach 100%
  top-1, while Loeys-Dietz reaches 30% — it is misranked as Marfan in 140 of 235
  cases. That confusion is expected: the two conditions were considered a single
  entity until 2005, share aortic, skeletal, and ocular features, and the findings
  that separate them (bifid uvula, hypertelorism, arterial tortuosity) are
  frequently absent from a published phenotype list. Both conditions are covered by
  the same thoracic aortic aneurysm panel, so the top-3 result still produces the
  correct order.
- **The broad tier is unbenchmarked.** Its results are labeled ranking-only, with
  no test recommendation attached.
- This is decision support, not diagnosis. It surfaces candidates for clinician
  confirmation, and the interface states so.

---

## Running it

```bash
python3 -m venv venv && source venv/bin/activate
pip install anthropic fastapi "uvicorn[standard]" requests rapidfuzz python-dotenv

# reference data, ~55 MB, gitignored
curl -L -o app/data/hp.json        http://purl.obolibrary.org/obo/hp.json
curl -L -o app/data/phenotype.hpoa http://purl.obolibrary.org/obo/hp/phenotype.hpoa

python scripts/build_db.py      # 15-disease curated DB, every code verified
python scripts/build_ic.py      # information content table
python scripts/build_broad.py   # optional 8,213-disease ranking tier

echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
uvicorn app.main:app --port 8000
```

Command line, tests, and benchmark:

```bash
python app/pipeline.py --file app/data/real_cases/PMC12131641_fabry_lateonset_esrd.txt
python scripts/test_differential.py          # 18 checks
python scripts/benchmark.py                  # 669 real cases
python app/stages/actions.py --build-cache   # cache trials before demoing offline
```

---

## Layout

```
app/
  pipeline.py        four-stage orchestration (CLI)
  main.py            FastAPI + server-sent events
  static/index.html  single page, live agent trace
  stages/
    extract.py       Claude tool use; negation and family-history handling
    ground.py        HPO index, five-tier matching, ancestor walk
    differential.py  weighted overlap scoring
    actions.py       gene panels, red flags, ClinicalTrials.gov
scripts/
  build_db.py  build_broad.py  build_ic.py
  benchmark.py  test_differential.py  fetch_cases.py
```

`ground.py` deserves a note. Its five matching tiers — exact, synonym, alias,
normalized, fuzzy — each tag their result with the route taken. Fuzzy matches are
marked untrusted and render in amber in the interface, and unmapped findings are
displayed rather than quietly dropped. An interface that hides which of its own
matches are shaky is harder to trust than one that admits it.

---

## Data

HPO 2025-09-01 supplies the ontology and disease annotations. Orphanet and OMIM
supply the curated clinical layer — gene panels, red flags, discriminating
features — which no public file provides directly and which is what makes the
output actionable. phenopacket-store 0.1.27 supplies the benchmark.
ClinicalTrials.gov API v2 supplies trials live, backed by a local cache because
venue wifi is not worth betting a demo on. PMC Open Access case reports supply
end-to-end validation, with sources and licenses recorded in `SOURCES.md`.
Abridge's synthetic encounters serve as a negative control.

Every HPO code in the curated database is verified against `hp.json` at build
time. If a label fails to resolve to a real term, the build exits non-zero. A
broken build is preferable to a plausible-looking code that does not exist.

---

## Built during the hackathon

All four stages, the curated database, the scoring model, the benchmark and test
suites, the web application, and the case-report fetcher were written during the
event. Reference ontologies and public datasets are downloaded from their original
sources by the scripts above; nothing is vendored.

Clinical decision support, not a diagnosis. No gene, code, or panel shown in the
interface is model-generated.
