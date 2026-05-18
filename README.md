# IMLJD — Indian Matrimonial Litigation Judgment Dataset

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![HuggingFace](https://img.shields.io/badge/🤗-Dataset-yellow)](https://huggingface.co/datasets/joyboseroy/imljd)

A computational dataset of Indian matrimonial litigation judgments for legal NLP, knowledge graph construction, and procedural fairness research.

**Dataset:** 3,613 cases | **Coverage:** Supreme Court 2000–2024, Karnataka HC 2018–2024 | **Focus:** IPC 498A, DV Act, CrPC 482 quashing petitions

---

## Key Findings

| Finding | Value |
|---------|-------|
| SC quash success rate | **57.6%** (462 of 802 quash petitions) |
| Karnataka HC quash success rate | **39.7%** (849 of 2,136 petitions) |
| Differential (SC vs HC) | **+17.9 percentage points** |
| Judicial criticism of misuse (SC) | **10.0%** of cases |
| Relatives accused (SC) | **7.9%** of cases |
| Settlement at quash stage (SC) | **15.2%** (122 of 802) |

The 17.9-point quash success differential between SC and HC levels suggests systematic pre-filtering: cases reaching the SC are pre-selected for stronger quashing arguments.

---

## Dataset Structure

```
data/
├── parquet/
│   ├── sc_enriched.parquet       # 1,474 SC matrimonial cases 2000-2024
│   ├── hc_29_3.parquet           # 2,136 Karnataka HC 482 petitions 2018-2024
│   ├── sc_enriched.csv           # SC cases (no full text)
│   └── hc_matrimonial.csv        # HC cases
├── extracted/
│   └── sc_<path>.txt             # Full judgment text (SC cases)
└── kg/
    ├── imljd_graph.gexf          # Knowledge graph (Gephi)
    ├── imljd_graph.json          # Knowledge graph (D3/web)
    └── kg_stats.json             # Machine-readable statistics
```

---

## Schema

### Supreme Court (`sc_enriched.parquet`)

| Field | Type | Description |
|-------|------|-------------|
| case_id | str | Stable identifier |
| title | str | Case title |
| petitioner / respondent | str | Party names |
| decision_date | str | Date of judgment |
| year | str | Year |
| disposal_nature | str | Raw disposal (e.g. "Appeal(s) allowed") |
| case_type | str | quash / appeal / maintenance / bail / other |
| outcome | str | quashed / allowed / dismissed / settled / disposed / partly_allowed |
| statutes | str | Pipe-delimited: "IPC 498A \| CrPC 482" |
| mediation_mentioned | bool | Mediation discussed |
| settlement_mentioned | bool | Settlement / compromise mentioned |
| omnibus_vague_language | bool | "omnibus/vague allegations" language |
| relatives_accused | bool | In-laws / extended family named |
| judicial_criticism_misuse | bool | Court criticises abuse of process |
| arnesh_kumar_cited | bool | Arnesh Kumar guidelines cited |
| rajesh_sharma_cited | bool | Rajesh Sharma cited |
| cited_cases | json | Citation list |
| word_count | int | Full text word count |
| allegations_text | str | Extracted facts section |
| judicial_observations | str | Extracted court observations |

### Karnataka HC (`hc_29_3.parquet`)

| Field | Type | Description |
|-------|------|-------------|
| title | str | Case title (CRL.P/NNNNN/YYYY format) |
| description | str | Judgment header text |
| judge | str | Presiding judge |
| pdf_link | str | eCourts PDF path |
| decision_date | str | Date |
| disposal_nature | str | ALLOWED / DISMISSED / DISPOSED / Partly Allowed |
| outcome | str | Mapped: quashed / dismissed / disposed / partly_allowed |
| _year | int | Year |
| _court_name | str | Karnataka High Court |

---

## Data Sources

| Source | URL | Access |
|--------|-----|--------|
| SC judgments | s3://indian-supreme-court-judgments/ | Public, no auth |
| HC judgments | s3://indian-high-court-judgments/ | Public, no auth |

Both are AWS Open Data Registry datasets. No credentials required:
```bash
aws s3 ls s3://indian-supreme-court-judgments/ --no-sign-request
```

---

## Reproducing the Dataset

```bash
git clone https://github.com/joyboseroy/imljd
cd imljd
pip install -r requirements.txt

# SC metadata (fast)
python3 scripts/03downloadfixed.py --sc --years $(seq 2000 2024)
python3 scripts/enrichsc.py
python3 scripts/fixoutcomes.py

# SC full text (slow — downloads tar archives)
python3 scripts/fetchsclean.py --years 2015 2017 2019 2021 2023

# Karnataka HC 482 petitions
python3 scripts/hcextractv3.py --courts 29_3 --years 2018 2019 2020 2021 2022 2023 2024

# Build knowledge graph
python3 scripts/buildkg.py
```

---

## Requirements

```
boto3
pyarrow
pandas
pdfplumber
pymupdf
tqdm
rank_bm25
networkx
```

---

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{boseroy2026imljd,
  title     = {IMLJD: Indian Matrimonial Litigation Judgment Dataset},
  author    = {Bose, Joy},
  year      = {2026},
  url       = {https://github.com/joyboseroy/imljd},
  note      = {3,610 cases, Supreme Court 2000-2024 and Karnataka HC 2018-2024}
}
```

This work extends the legal reasoning framework from:
```bibtex
@article{boseroy2026falkor,
  title  = {FalkorDB-IRAC: Graph-Grounded Legal Reasoning},
  author = {Bose, Joy},
  year   = {2026},
  url    = {https://arxiv.org/abs/2605.14665}
}
```

---

## Ethics

- Public court judgments only — no private communications or FIR data
- Names of parties present as in original public records
- Anonymisation pass recommended before any downstream NLP training
- Framing: procedural fairness research, not case outcome prediction
- Signal flags are descriptive, not diagnostic
- Not suitable for "false case detection" — ground truth does not exist cleanly in this domain

---

## Related Work

- [ILDC Dataset](https://arxiv.org/abs/2105.13562) — 35k SC cases with annotations
- [LawSum](https://arxiv.org/abs/2110.01188) — 10k+ judgments with summaries  
- [InLegalNLP](https://arxiv.org/abs/2209.06049) — Indian legal NLP benchmark

IMLJD complements these by focusing specifically on matrimonial litigation with outcome labels, signal flags, and a knowledge graph optimised for procedural fairness analysis.
