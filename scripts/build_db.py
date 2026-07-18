#!/usr/bin/env python3
"""
build_db.py - Build rare_diseases.json for the Rare Disease Diagnostic Agent.

Merges two things:
  1. CURATED clinical metadata (hand-written below): the gene panel to order,
     discriminating features, common misdiagnoses, red flags. This is NOT in
     any public file - it is the part that makes the agent's output actionable.
  2. GROUND TRUTH phenotypes pulled from the real HPO release:
       hp.json         -> canonical HPO id + label for every term
       phenotype.hpoa  -> which HPO terms belong to which disease, + frequency

Every HPO id in the output is verified against hp.json. If a curated label does
not resolve to a real term, the build FAILS LOUDLY rather than shipping a made
up code. That guarantee is the whole anti-hallucination story of this project.

Usage:
    python build_db.py --hp data/hp.json --hpoa data/phenotype.hpoa \
                       --out data/rare_diseases.json
"""

import argparse, csv, json, sys
from collections import defaultdict

# --------------------------------------------------------------------------
# CURATED LAYER
# phenotypes = HPO term LABELS chosen for diagnostic signal, not completeness.
# Ordered roughly most-discriminating first. Frequencies are filled in from
# HPOA at build time, so do not hand-write them here.
# --------------------------------------------------------------------------
CURATED = [
{
 "id": "ORPHA:558", "omim": "154700", "name": "Marfan syndrome",
 "inheritance": "Autosomal dominant", "prevalence": "1-5 / 10,000",
 "organ_systems": ["cardiovascular", "ocular", "skeletal"],
 "phenotypes": ["Ectopia lentis","Aortic root aneurysm","Aortic dissection",
   "Disproportionate tall stature","Arachnodactyly","Pectus excavatum",
   "Pectus carinatum","Scoliosis","Joint hypermobility","Mitral valve prolapse",
   "Myopia","Dural ectasia","High, narrow palate","Spontaneous pneumothorax",
   "Striae distensae","Pes planus"],
 "discriminating_features": ["Ectopia lentis (upward lens dislocation)",
   "Aortic root dilatation at the sinuses of Valsalva","Dural ectasia on MRI"],
 "commonly_misdiagnosed_as": ["Loeys-Dietz syndrome","Vascular Ehlers-Danlos syndrome",
   "Homocystinuria","Familial thoracic aortic aneurysm"],
 "recommended_test": {"type":"Targeted gene panel","panel_name":"Thoracic Aortic Aneurysm / Connective Tissue Panel",
   "genes":["FBN1","TGFBR1","TGFBR2","SMAD3","ACTA2","MYH11","COL3A1"],
   "turnaround":"2-4 weeks",
   "adjunct":["Transthoracic echocardiogram with aortic root Z-score","Slit-lamp exam with dilated pupils"]},
 "red_flags": ["Aortic root Z-score >2 or rapid growth -> urgent cardiothoracic referral",
   "Acute chest/back pain in a Marfanoid patient = dissection until proven otherwise"],
 "trial_query": "Marfan Syndrome",
 "source": "Orphanet ORPHA:558 / OMIM 154700 / HPO 2025-09-01",
},
{
 "id": "ORPHA:60030", "omim": "609192", "name": "Loeys-Dietz syndrome",
 "inheritance": "Autosomal dominant", "prevalence": "<1 / 1,000,000",
 "organ_systems": ["cardiovascular", "craniofacial", "skeletal", "cutaneous"],
 "phenotypes": ["Arterial tortuosity","Hypertelorism","Bifid uvula","Cleft palate",
   "Aortic aneurysm","Aortic dissection","Arterial dissection","Craniosynostosis",
   "Blue sclerae","Translucent skin" ,"Pectus carinatum","Scoliosis",
   "Joint hypermobility","Talipes equinovarus","Atypical scarring of skin"],
 "discriminating_features": ["Bifid uvula or cleft palate (absent in Marfan)",
   "Arterial tortuosity on head-to-pelvis CTA","Hypertelorism",
   "Dissection at smaller aortic diameters than Marfan"],
 "commonly_misdiagnosed_as": ["Marfan syndrome","Vascular Ehlers-Danlos syndrome"],
 "recommended_test": {"type":"Targeted gene panel","panel_name":"Thoracic Aortic Aneurysm / Connective Tissue Panel",
   "genes":["TGFBR1","TGFBR2","SMAD3","TGFB2","TGFB3","FBN1"],
   "turnaround":"2-4 weeks",
   "adjunct":["Head-to-pelvis CT or MR angiography","Echocardiogram"]},
 "red_flags": ["Aneurysms rupture at smaller diameters - lower the surgical threshold",
   "Whole-arterial-tree imaging required, not just the aortic root"],
 "trial_query": "Loeys-Dietz Syndrome",
 "source": "Orphanet ORPHA:60030 / OMIM 609192 / HPO 2025-09-01",
},
{
 "id": "ORPHA:286", "omim": "130050", "name": "Vascular Ehlers-Danlos syndrome (vEDS)",
 "inheritance": "Autosomal dominant", "prevalence": "1-9 / 100,000",
 "organ_systems": ["cardiovascular", "gastrointestinal", "cutaneous", "obstetric"],
 "phenotypes": ["Dermal translucency","Arterial dissection","Gastrointestinal infarctions",
   "Uterine rupture","Bruising susceptibility","Thin skin","Prematurely aged appearance",
   "Cigarette-paper scars","Aortic dissection","Pneumothorax","Joint hypermobility",
   "Varicose veins","Deeply set eye","Thin vermilion border","Talipes equinovarus"],
 "discriminating_features": ["Translucent skin with visible subcutaneous veins",
   "Spontaneous arterial, bowel or uterine rupture","Characteristic thin, pinched facies",
   "Hypermobility often limited to small joints only"],
 "commonly_misdiagnosed_as": ["Hypermobile EDS","Marfan syndrome","Child abuse (in paediatrics)"],
 "recommended_test": {"type":"Targeted gene panel","panel_name":"EDS / Connective Tissue Panel",
   "genes":["COL3A1","COL1A1","TGFBR1","TGFBR2"],
   "turnaround":"2-3 weeks",
   "adjunct":["Non-invasive vascular imaging (avoid arteriography)"]},
 "red_flags": ["AVOID invasive arteriography and elective vascular procedures - high rupture risk",
   "Acute abdominal or chest pain warrants immediate imaging",
   "Pregnancy carries substantial maternal mortality - refer to high-risk obstetrics"],
 "trial_query": "Vascular Ehlers-Danlos Syndrome",
 "source": "Orphanet ORPHA:286 / OMIM 130050 / HPO 2025-09-01",
},
{
 "id": "ORPHA:324", "omim": "301500", "name": "Fabry disease",
 "inheritance": "X-linked", "prevalence": "1-9 / 100,000",
 "organ_systems": ["renal", "cardiovascular", "neurologic", "cutaneous", "ocular"],
 "phenotypes": ["Cornea verticillata","Angiokeratoma","Acroparesthesia","Hypohidrosis",
   "Proteinuria","Renal insufficiency","Left ventricular hypertrophy",
   "Hypertrophic cardiomyopathy","Stroke","Transient ischemic attack",
   "Heat intolerance","Abdominal pain","Sensorineural hearing impairment",
   "Decreased alpha-galactosidase A activity","Tinnitus","Fatigue"],
 "discriminating_features": ["Cornea verticillata on slit lamp (whorl-like corneal opacity)",
   "Burning acroparesthesias of hands and feet since childhood",
   "Angiokeratomas in the bathing-trunk distribution",
   "Unexplained LVH plus proteinuria plus early stroke in one patient"],
 "commonly_misdiagnosed_as": ["Fibromyalgia","Rheumatic fever","Multiple sclerosis",
   "Hypertrophic cardiomyopathy (isolated)","Growing pains"],
 "recommended_test": {"type":"Enzyme assay then targeted sequencing",
   "panel_name":"Alpha-galactosidase A activity (males) + GLA sequencing (all)",
   "genes":["GLA"], "turnaround":"1-2 weeks (enzyme), 2-3 weeks (sequencing)",
   "adjunct":["Plasma lyso-Gb3","Urine protein/creatinine ratio","Echocardiogram","Slit-lamp exam"]},
 "red_flags": ["Enzyme assay is UNRELIABLE in heterozygous females - sequence GLA regardless",
   "Disease-specific therapy exists (ERT / chaperone) - diagnosis changes management immediately"],
 "trial_query": "Fabry Disease",
 "source": "Orphanet ORPHA:324 / OMIM 301500 / HPO 2025-09-01",
},
{
 "id": "ORPHA:905", "omim": "277900", "name": "Wilson disease",
 "inheritance": "Autosomal recessive", "prevalence": "1-9 / 100,000",
 "organ_systems": ["hepatic", "neurologic", "psychiatric", "ocular", "hematologic"],
 "phenotypes": ["Kayser-Fleischer ring","Decreased circulating ceruloplasmin concentration",
   "Sunflower cataract","Dystonia","Tremor","Dysarthria","Cirrhosis",
   "Elevated circulating hepatic transaminase concentration","Acute hepatic failure",
   "Hemolytic anemia","Psychosis","Personality changes","Chorea","Hepatomegaly","Jaundice"],
 "discriminating_features": ["Kayser-Fleischer rings on slit lamp",
   "Low serum ceruloplasmin with high 24h urinary copper",
   "Coombs-negative haemolytic anaemia with acute liver failure",
   "Neuropsychiatric symptoms in a young patient with abnormal LFTs"],
 "commonly_misdiagnosed_as": ["Autoimmune hepatitis","Non-alcoholic fatty liver disease",
   "Primary psychiatric disorder","Essential tremor","Juvenile Parkinsonism"],
 "recommended_test": {"type":"Biochemical then targeted sequencing",
   "panel_name":"Serum ceruloplasmin + 24h urinary copper, then ATP7B sequencing",
   "genes":["ATP7B"], "turnaround":"3-7 days (biochem), 2-3 weeks (sequencing)",
   "adjunct":["Slit-lamp exam for KF rings","Liver biopsy copper quantification if equivocal"]},
 "red_flags": ["TREATABLE - chelation prevents irreversible neurological injury; do not delay",
   "Screen all first-degree relatives once a proband is confirmed"],
 "trial_query": "Wilson Disease",
 "source": "Orphanet ORPHA:905 / OMIM 277900 / HPO 2025-09-01",
},
{
 "id": "ORPHA:77259", "omim": "230800", "name": "Gaucher disease type 1",
 "inheritance": "Autosomal recessive", "prevalence": "1-9 / 100,000",
 "organ_systems": ["hematologic", "hepatic", "skeletal"],
 "phenotypes": ["Erlenmeyer flask deformity of the femurs","Splenomegaly","Hepatosplenomegaly",
   "Thrombocytopenia","Anemia","Avascular necrosis","Bone pain","Pathologic fracture",
   "Decreased beta-glucocerebrosidase level","Osteopenia","Bruising susceptibility",
   "Growth delay","Pulmonary arterial hypertension","HP:0032640","Fatigue"],
 "discriminating_features": ["Massive splenomegaly with thrombocytopenia and no clear cause",
   "Erlenmeyer flask deformity of the distal femur on plain film",
   "Bone crises misread as osteomyelitis"],
 "commonly_misdiagnosed_as": ["Immune thrombocytopenic purpura","Haematological malignancy",
   "Osteomyelitis","Juvenile idiopathic arthritis"],
 "recommended_test": {"type":"Enzyme assay then targeted sequencing",
   "panel_name":"Beta-glucocerebrosidase activity (dried blood spot) + GBA1 sequencing",
   "genes":["GBA1"], "turnaround":"1-2 weeks",
   "adjunct":["Plasma glucosylsphingosine (lyso-Gb1)","MRI femur for marrow burden"]},
 "red_flags": ["TREATABLE with enzyme replacement / substrate reduction therapy",
   "Avoid splenectomy before diagnosis - it accelerates bone and liver disease"],
 "trial_query": "Gaucher Disease",
 "source": "Orphanet ORPHA:77259 / OMIM 230800 / HPO 2025-09-01",
},
{
 "id": "ORPHA:365", "omim": "232300", "name": "Pompe disease (acid maltase deficiency)",
 "inheritance": "Autosomal recessive", "prevalence": "1-9 / 100,000",
 "organ_systems": ["muscular", "respiratory", "cardiovascular"],
 "phenotypes": ["Decreased circulating acid maltase activity","Progressive proximal muscle weakness",
   "Diaphragmatic weakness","Respiratory insufficiency due to muscle weakness","Orthopnea",
   "HP:0003236","Gowers sign","Hyperlordosis",
   "Difficulty climbing stairs","Floppy infant","Hypertrophic cardiomyopathy","Macroglossia",
   "Sleep apnea","EMG: myopathic abnormalities","Exercise intolerance"],
 "discriminating_features": ["Respiratory failure OUT OF PROPORTION to limb weakness",
   "Orthopnea / nocturnal hypoventilation as a presenting complaint",
   "Cardiomegaly with short PR interval in the infantile form"],
 "commonly_misdiagnosed_as": ["Limb-girdle muscular dystrophy","Polymyositis",
   "Spinal muscular atrophy","Idiopathic hyperCKemia"],
 "recommended_test": {"type":"Enzyme assay then targeted sequencing",
   "panel_name":"GAA enzyme activity (dried blood spot) + GAA sequencing",
   "genes":["GAA"], "turnaround":"1-2 weeks",
   "adjunct":["Upright and supine forced vital capacity","CK","ECG/echocardiogram"]},
 "red_flags": ["TREATABLE with enzyme replacement therapy - earlier start, better outcome",
   "Check supine FVC; a normal seated FVC can hide diaphragmatic failure"],
 "trial_query": "Pompe Disease",
 "source": "Orphanet ORPHA:365 / OMIM 232300 / HPO 2025-09-01",
},
{
 "id": "ORPHA:465508", "omim": "235200", "name": "HFE-related hereditary hemochromatosis",
 "inheritance": "Autosomal recessive", "prevalence": "1-5 / 10,000",
 "organ_systems": ["hepatic", "endocrine", "cardiovascular", "musculoskeletal"],
 "phenotypes": ["Elevated transferrin saturation","Increased circulating ferritin concentration",
   "Generalized bronze hyperpigmentation","Diabetes mellitus","Cirrhosis","Hepatomegaly",
   "Arthropathy","Stiff interphalangeal joints","Cardiomyopathy","Hypogonadotropic hypogonadism",
   "Erectile dysfunction","Fatigue","Arthralgia","Hepatocellular carcinoma","Amenorrhea"],
 "discriminating_features": ["Transferrin saturation >45% with raised ferritin",
   "2nd and 3rd MCP joint arthropathy ('iron fist')",
   "Bronze skin plus diabetes plus hepatomegaly"],
 "commonly_misdiagnosed_as": ["Alcoholic liver disease","Non-alcoholic fatty liver disease",
   "Rheumatoid arthritis","Type 2 diabetes (isolated)"],
 "recommended_test": {"type":"Biochemical then targeted genotyping",
   "panel_name":"Fasting transferrin saturation + ferritin, then HFE genotyping (C282Y/H63D)",
   "genes":["HFE","HJV","HAMP","TFR2","SLC40A1"],
   "turnaround":"1 week (biochem), 1-2 weeks (genotype)",
   "adjunct":["Hepatic MRI T2* for iron quantification","Liver fibrosis staging if ferritin >1000"]},
 "red_flags": ["TREATABLE with phlebotomy - cirrhosis and cardiomyopathy are preventable if caught early",
   "Ferritin >1000 ng/mL warrants fibrosis assessment"],
 "trial_query": "Hemochromatosis",
 "source": "Orphanet ORPHA:465508 / OMIM 235200 / HPO 2025-09-01",
},
{
 "id": "ORPHA:774", "omim": "187300", "name": "Hereditary hemorrhagic telangiectasia (Osler-Weber-Rendu)",
 "inheritance": "Autosomal dominant", "prevalence": "1-9 / 100,000",
 "organ_systems": ["vascular", "pulmonary", "hepatic", "neurologic", "gastrointestinal"],
 "phenotypes": ["Spontaneous, recurrent epistaxis","Mucosal telangiectasiae","Lip telangiectasia",
   "Tongue telangiectasia","Pulmonary arteriovenous malformation","Cerebral arteriovenous malformation",
   "Hepatic arteriovenous malformation","Gastrointestinal hemorrhage","Anemia",
   "Telangiectasia of the skin","Migraine","Cerebral hemorrhage","Transient ischemic attack",
   "Pulmonary arterial hypertension","Hemoptysis"],
 "discriminating_features": ["Recurrent spontaneous epistaxis since childhood plus family history",
   "Mucocutaneous telangiectases on lips, tongue and fingertips",
   "Paradoxical embolic stroke or brain abscess from a pulmonary AVM",
   "Curacao criteria: 3 of 4 = definite"],
 "commonly_misdiagnosed_as": ["Idiopathic epistaxis","Iron deficiency anaemia of unknown cause",
   "Cryptogenic stroke"],
 "recommended_test": {"type":"Targeted gene panel","panel_name":"HHT Panel",
   "genes":["ENG","ACVRL1","SMAD4","GDF2"], "turnaround":"2-3 weeks",
   "adjunct":["Contrast (bubble) transthoracic echocardiography to screen for pulmonary AVM",
     "Brain MRI","Ferritin/CBC"]},
 "red_flags": ["Untreated pulmonary AVM causes brain abscess and paradoxical stroke - screen everyone",
   "SMAD4 variants imply juvenile polyposis - add GI surveillance"],
 "trial_query": "Hereditary Hemorrhagic Telangiectasia",
 "source": "Orphanet ORPHA:774 / OMIM 187300 / HPO 2025-09-01",
},
{
 "id": "ORPHA:63", "omim": "301050", "name": "Alport syndrome",
 "inheritance": "X-linked (85%), autosomal recessive or dominant",
 "prevalence": "1-9 / 100,000",
 "organ_systems": ["renal", "auditory", "ocular"],
 "phenotypes": ["Anterior lenticonus","Glomerular basement membrane lamellation",
   "Sensorineural hearing impairment","Hematuria","Microscopic hematuria","Proteinuria",
   "Renal insufficiency","Stage 5 chronic kidney disease","Retinal flecks","Hypertension",
   "Posterior subcapsular cataract","Recurrent corneal erosions","Nephritis",
   "Thin glomerular basement membrane","Focal segmental glomerulosclerosis"],
 "discriminating_features": ["Persistent microscopic haematuria plus sensorineural deafness",
   "Anterior lenticonus is essentially pathognomonic",
   "Basket-weave GBM lamellation on electron microscopy",
   "Family history of renal failure in young males"],
 "commonly_misdiagnosed_as": ["IgA nephropathy","Thin basement membrane nephropathy",
   "Focal segmental glomerulosclerosis (idiopathic)"],
 "recommended_test": {"type":"Targeted gene panel","panel_name":"Hereditary Nephropathy / Alport Panel",
   "genes":["COL4A5","COL4A3","COL4A4"], "turnaround":"2-3 weeks",
   "adjunct":["Audiometry","Ophthalmology exam","Urine protein/creatinine ratio"]},
 "red_flags": ["Early ACE inhibition delays renal failure by years - diagnosis is time-critical",
   "Do not dismiss isolated microscopic haematuria in a child with deafness"],
 "trial_query": "Alport Syndrome",
 "source": "Orphanet ORPHA:63 / OMIM 301050 / HPO 2025-09-01",
},
{
 "id": "ORPHA:805", "omim": "191100", "name": "Tuberous sclerosis complex",
 "inheritance": "Autosomal dominant", "prevalence": "1-9 / 100,000",
 "organ_systems": ["neurologic", "cutaneous", "renal", "cardiac", "pulmonary"],
 "phenotypes": ["Cortical tubers","Subependymal nodules","Subependymal giant-cell astrocytoma",
   "Hypomelanotic macule","Shagreen patch","Angiofibromas","Ungual fibroma",
   "Renal angiomyolipoma","Cardiac rhabdomyoma","Infantile spasms","Focal-onset seizure",
   "Autism","Intellectual disability","Pulmonary lymphangiomyomatosis","Retinal hamartoma"],
 "discriminating_features": ["Hypomelanotic macules seen under Wood's lamp",
   "Facial angiofibromas often misdiagnosed as acne",
   "Cardiac rhabdomyoma detected antenatally",
   "Infantile spasms plus hypopigmented macules"],
 "commonly_misdiagnosed_as": ["Idiopathic epilepsy","Acne vulgaris","Idiopathic autism",
   "Sporadic renal angiomyolipoma"],
 "recommended_test": {"type":"Targeted gene panel","panel_name":"TSC Panel",
   "genes":["TSC1","TSC2"], "turnaround":"2-3 weeks",
   "adjunct":["Brain MRI","Renal MRI/ultrasound","Echocardiogram","Wood's lamp skin exam","EEG"]},
 "red_flags": ["mTOR inhibitors (everolimus) treat SEGA, angiomyolipoma and LAM - diagnosis unlocks therapy",
   "Renal angiomyolipoma >3cm carries haemorrhage risk"],
 "trial_query": "Tuberous Sclerosis Complex",
 "source": "Orphanet ORPHA:805 / OMIM 191100 / HPO 2025-09-01",
},
{
 "id": "ORPHA:636", "omim": "162200", "name": "Neurofibromatosis type 1",
 "inheritance": "Autosomal dominant", "prevalence": "1-5 / 10,000",
 "organ_systems": ["cutaneous", "neurologic", "ocular", "skeletal"],
 "phenotypes": ["Multiple cafe-au-lait spots","Lisch nodules","Axillary freckling",
   "Inguinal freckling","Plexiform neurofibroma","Optic nerve glioma","Spinal neurofibroma",
   "Macrocephaly","Specific learning disability","Attention deficit hyperactivity disorder",
   "Scoliosis","Short stature","Pheochromocytoma","Slender long bone","Hypertension"],
 "discriminating_features": ["Six or more cafe-au-lait macules plus skinfold freckling",
   "Lisch nodules on slit lamp (present in most adults)",
   "Tibial dysplasia / pseudarthrosis"],
 "commonly_misdiagnosed_as": ["Legius syndrome","McCune-Albright syndrome",
   "Constitutional mismatch repair deficiency","Isolated ADHD"],
 "recommended_test": {"type":"Targeted gene panel","panel_name":"NF / RASopathy Panel",
   "genes":["NF1","SPRED1","NF2","LZTR1"], "turnaround":"3-4 weeks",
   "adjunct":["Slit-lamp exam for Lisch nodules","Whole-body MRI if plexiform tumour suspected",
     "Annual blood pressure (phaeochromocytoma / renal artery stenosis)"]},
 "red_flags": ["SPRED1 testing distinguishes Legius syndrome - it has no tumour risk, so this changes surveillance",
   "New pain or rapid growth in a plexiform neurofibroma suggests malignant transformation"],
 "trial_query": "Neurofibromatosis Type 1",
 "source": "Orphanet ORPHA:636 / OMIM 162200 / HPO 2025-09-01",
},
{
 "id": "ORPHA:550", "omim": "540000", "name": "MELAS (mitochondrial encephalomyopathy, lactic acidosis, stroke-like episodes)",
 "inheritance": "Mitochondrial (maternal)", "prevalence": "1-9 / 100,000",
 "organ_systems": ["neurologic", "muscular", "endocrine", "auditory", "cardiovascular"],
 "phenotypes": ["Stroke-like episode","Increased circulating lactate concentration",
   "Elevated brain lactate level by MRS","Ragged-red muscle fibers","Sensorineural hearing impairment",
   "Type II diabetes mellitus","Short stature","Migraine","Recurrent paroxysmal headache",
   "Seizure","Encephalopathy","Exercise intolerance","Hypertrophic cardiomyopathy",
   "Muscle weakness","Dementia","Increased CSF lactate"],
 "discriminating_features": ["Stroke-like lesions that do NOT respect vascular territories",
   "Diabetes plus sensorineural deafness in a maternal inheritance pattern",
   "Raised lactate in blood or CSF","Short stature with migraine and seizures"],
 "commonly_misdiagnosed_as": ["Ischaemic stroke","Viral encephalitis","Complex migraine",
   "Idiopathic epilepsy","Type 2 diabetes (isolated)"],
 "recommended_test": {"type":"Mitochondrial DNA testing",
   "panel_name":"mtDNA targeted analysis (m.3243A>G) then full mtDNA sequencing",
   "genes":["MT-TL1","MT-ND5","MT-TK"], "turnaround":"2-4 weeks",
   "adjunct":["Blood AND urine sediment sampling (heteroplasmy is low in blood)",
     "Serum/CSF lactate","Brain MRI with MR spectroscopy"]},
 "red_flags": ["Test URINE or buccal cells - a negative blood test does not exclude MELAS",
   "AVOID valproate and metformin; both can precipitate crises",
   "Maternal relatives are at risk - offer family testing"],
 "trial_query": "MELAS Syndrome",
 "source": "Orphanet ORPHA:550 / OMIM 540000 / HPO 2025-09-01",
},
{
 "id": "OMIM:109650", "omim": "109650", "name": "Behcet syndrome",
 "inheritance": "Complex / multifactorial", "prevalence": "1-9 / 100,000",
 "organ_systems": ["mucocutaneous", "ocular", "vascular", "musculoskeletal"],
 "phenotypes": ["Oral ulcer","Genital ulcers","Iridocyclitis","Hypopyon","Chorioretinitis",
   "Erythema nodosum","Superficial thrombophlebitis","Arthritis","Iritis",
   "Patchy alopecia","Raynaud phenomenon","Epididymitis","Erythema","Irritability"],
 "discriminating_features": ["Recurrent oral ulceration at least 3x/year plus genital ulceration",
   "Hypopyon uveitis","Pathergy reaction at needle-prick sites",
   "Venous thrombosis in a young patient with mucosal ulcers"],
 "commonly_misdiagnosed_as": ["Recurrent aphthous stomatitis","Inflammatory bowel disease",
   "Herpes simplex","Reactive arthritis"],
 "recommended_test": {"type":"Clinical criteria plus supportive testing",
   "panel_name":"ISG/ICBD clinical criteria; HLA-B51 typing is supportive, not diagnostic",
   "genes":["HLA-B51"], "turnaround":"1-2 weeks",
   "adjunct":["Pathergy test","Ophthalmology review within days if ocular symptoms","ESR/CRP"]},
 "red_flags": ["Ocular involvement threatens sight - same-week ophthalmology referral",
   "Not a monogenic disorder; gene panel is NOT the right test here"],
 "trial_query": "Behcet Syndrome",
 "source": "OMIM 109650 / HPO 2025-09-01",
},
{
 "id": "ORPHA:567", "omim": "192430", "name": "22q11.2 deletion syndrome (DiGeorge / velocardiofacial)",
 "inheritance": "Autosomal dominant (usually de novo)", "prevalence": "1-5 / 10,000",
 "organ_systems": ["cardiovascular", "immunologic", "endocrine", "craniofacial", "neurodevelopmental"],
 "phenotypes": ["Hypoplasia of the thymus","Hypocalcemia","Hypoparathyroidism","Tetralogy of Fallot",
   "Truncus arteriosus","Ventricular septal defect","Abnormal aortic arch morphology","Cleft palate",
   "Hypernasal speech","Immunodeficiency","Global developmental delay","Schizophrenia",
   "Long face","Prominent nasal bridge","Specific learning disability","Recurrent otitis media"],
 "discriminating_features": ["Conotruncal cardiac defect plus hypocalcaemia plus recurrent infection",
   "Velopharyngeal insufficiency with hypernasal speech",
   "Markedly elevated lifetime risk of psychotic illness"],
 "commonly_misdiagnosed_as": ["Isolated congenital heart disease","Idiopathic developmental delay",
   "Primary immunodeficiency (isolated)","Primary schizophrenia"],
 "recommended_test": {"type":"Chromosomal microarray / targeted deletion analysis",
   "panel_name":"Chromosomal microarray (CMA) or FISH/MLPA for 22q11.2",
   "genes":["TBX1","COMT","22q11.2 region"], "turnaround":"2-3 weeks",
   "adjunct":["Ionised calcium and PTH","Lymphocyte subsets / T-cell counts","Echocardiogram","Renal ultrasound"]},
 "red_flags": ["Sequencing panels MISS this - a deletion needs microarray, FISH or MLPA",
   "Give irradiated blood products until T-cell function is confirmed",
   "Check calcium before any surgery"],
 "trial_query": "22q11.2 Deletion Syndrome",
 "source": "Orphanet ORPHA:567 / OMIM 192430 / HPO 2025-09-01",
},
]

# HPO frequency subontology -> our three buckets
FREQ_TERMS = {"HP:0040280":"obligate","HP:0040281":"frequent","HP:0040282":"frequent",
              "HP:0040283":"occasional","HP:0040284":"occasional","HP:0040285":"excluded"}

def bucket(raw):
    """HPOA frequency (HP term, n/m, or pct) -> obligate | frequent | occasional."""
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

def load_hpo_labels(path):
    """hp.json -> ({label_lower: HP:id}, {HP:id: canonical label})"""
    nodes = json.load(open(path))["graphs"][0]["nodes"]
    by_label, by_id = {}, {}
    for n in nodes:
        if "/HP_" not in n.get("id", "") or not n.get("lbl"):
            continue
        hid = n["id"].split("/")[-1].replace("_", ":")
        if n.get("meta", {}).get("deprecated"):
            continue
        by_id[hid] = n["lbl"]
        by_label[n["lbl"].lower()] = hid
        for syn in n.get("meta", {}).get("synonyms", []):
            by_label.setdefault(syn["val"].lower(), hid)
    return by_label, by_id

def load_hpoa(path, wanted):
    """phenotype.hpoa -> {disease_id: {HP:id: frequency_bucket}}"""
    out = defaultdict(dict)
    with open(path) as fh:
        rows = csv.DictReader((l for l in fh if not l.startswith("#")), delimiter="\t")
        for r in rows:
            if r["database_id"] in wanted and r["aspect"] == "P" and not r["qualifier"]:
                out[r["database_id"]][r["hpo_id"]] = bucket(r["frequency"])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hp",   default="data/hp.json")
    ap.add_argument("--hpoa", default="data/phenotype.hpoa")
    ap.add_argument("--out",  default="data/rare_diseases.json")
    a = ap.parse_args()

    by_label, by_id = load_hpo_labels(a.hp)
    print(f"[hp.json]        {len(by_id):,} active terms, {len(by_label):,} searchable strings")

    annots = load_hpoa(a.hpoa, {d["id"] for d in CURATED})
    print(f"[phenotype.hpoa] annotations for {len(annots)} of {len(CURATED)} diseases")

    diseases, errors = [], []
    for d in CURATED:
        resolved, ann = [], annots.get(d["id"], {})
        for label in d["phenotypes"]:
            # Accept a raw HP:####### id (stable) or a label (renamed between releases)
            hid = label if label.startswith("HP:") and label in by_id \
                  else by_label.get(label.lower())
            if not hid:
                errors.append(f"{d['name']}: no HPO term matches '{label}'")
                continue
            resolved.append({"hpo_id": hid,
                             "label": by_id[hid],
                             "frequency": ann.get(hid, "occasional"),
                             "in_hpoa": hid in ann})
        order = {"obligate": 0, "frequent": 1, "occasional": 2}
        resolved.sort(key=lambda p: (order[p["frequency"]], p["label"]))
        rec = dict(d); rec["phenotypes"] = resolved
        diseases.append(rec)
        unverified = sum(1 for p in resolved if not p["in_hpoa"])
        flag = f"  ({unverified} not in HPOA for this disease)" if unverified else ""
        print(f"  {d['id']:<16} {d['name'][:44]:<46} {len(resolved):>2} terms{flag}")

    if errors:
        print("\nUNRESOLVED LABELS - fix these, do not ship invented codes:", file=sys.stderr)
        for e in errors:
            print("  " + e, file=sys.stderr)
        sys.exit(1)

    db = {"schema_version": "1.0",
          "hpo_release": "2025-09-01",
          "disease_count": len(diseases),
          "note": "Hackathon demo subset. Curated clinical metadata; phenotypes verified against HPO.",
          "diseases": diseases}
    json.dump(db, open(a.out, "w"), indent=2)
    total = sum(len(d["phenotypes"]) for d in diseases)
    print(f"\nWrote {a.out}: {len(diseases)} diseases, {total} verified phenotype terms")

if __name__ == "__main__":
    main()
