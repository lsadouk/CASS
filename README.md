# CASS: Context-Aware Semantic Similarity

**A lightweight safety-aware retrieval framework for specialized child-facing applications.**

[![HuggingFace QESC](https://img.shields.io/badge/🤗%20Dataset-QESC-blue)](https://huggingface.co/datasets/lsadouk1111/QESC) 
[![HuggingFace PhonEx](https://img.shields.io/badge/🤗%20Dataset-PhonEx-orange)](https://huggingface.co/datasets/lsadouk1111/PhonEx)
[![Python](https://img.shields.io/badge/Python-3.8+-green)](https://python.org)
[![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey)](https://creativecommons.org/licenses/by-nc/4.0/)
---

## What is CASS?

CASS extends cosine similarity with pluggable domain-specific constraint functions:

```
CASS(q, c) = α · cos(φ(q), φ(c))  +  Σᵢ βᵢ · Cᵢ(q, c)
```

where:
- `φ(·)` — pretrained multilingual sentence encoder (MiniLM-L12)
- `Cᵢ` — domain-specific constraint functions (Gaussian or categorical)
- `α + Σβᵢ = 1` — weights sum to 1

**No model training required.** CASS works off-the-shelf with any multilingual encoder.

---

## Repository Contents

```
cass/
├── README.md
├── CASS_Instantiation1_QESC.ipynb       # Full notebook — Islamic emotional education
├── CASS_Instantiation2_PhonEx.ipynb     # Full notebook — Dyslexia reading support
├── QESC_v1.0.json                       # QESC corpus (100 entries)
└── PHONICS_CORPUS_v1.0.json             # PhonEx corpus (100 entries)
```

Each notebook contains all code cells in order:
- Corpus loading and embedding generation
- Constraint function definitions
- Detection functions (EI, RT, DIFF, ERR)
- Weight optimization (Dirichlet search)
- CASS retrieval function
- Safety gate demonstrations
- Ablation study (BM25 + 4 CASS variants)

---

## Two Instantiations

### Instantiation 1 — Islamic Children's Emotional Education

**Notebook:** `CASS_Instantiation1_QESC.ipynb`  
**Corpus:** `QESC_v1.0.json`

A child types a free-text emotional expression in English, French, or Moroccan Darija.  
CASS retrieves the most appropriate Quranic prophet situational scene.

| Parameter | Value |
|---|---|
| Encoder | paraphrase-multilingual-MiniLM-L12-v2 |
| C₁ | C_AEA — Emotional Intensity (Gaussian, λ=0.2) |
| C₂ | C_RES — Resolution Type (Gaussian, μ=2.0) |
| α, β₁, β₂ | 0.724, 0.164, 0.112 |
| MHR (validation) | 1.739 / 3.000 |

---

### Instantiation 2 — French Dyslexia Reading Support

**Notebook:** `CASS_Instantiation2_PhonEx.ipynb`  
**Corpus:** `PHONICS_CORPUS_v1.0.json`

A parent or teacher types a free-text description of a child's reading difficulty.  
CASS retrieves the most appropriate phonics remediation exercise.

| Parameter | Value |
|---|---|
| Encoder | paraphrase-multilingual-MiniLM-L12-v2 |
| C₁ | C_DIFF — Phonological Difficulty (Gaussian, λ=0.2) |
| C₂ | C_ERR — Error Type (Categorical) |
| α, β₁, β₂ | 0.305, 0.186, 0.510 |
| MHR (validation) | 1.900 / 3.000 |

---

## Quick Start

### Requirements

```bash
pip install sentence-transformers numpy rank_bm25
```

### Run in Google Colab

1. Upload `QESC_v1.0.json` or `PHONICS_CORPUS_v1.0.json` to your Google Drive
2. Open the corresponding notebook in Colab
3. Run all cells from top to bottom

### Example queries

```python
# Instantiation 1 — Islamic emotional education
retrieve("I feel sad and nobody understands me")
retrieve("Je me sens abandonné par mes amis")
retrieve("ma kaynch had li yfahmni")

# Instantiation 2 — Dyslexia reading support
retrieve_phonex("My child confuses b and d when reading")
retrieve_phonex("Elle saute des syllabes dans les mots longs")
retrieve_phonex("Weldi ma iqrach mezyan, kayqra harf harf")
```

---

## Safety Gate

CASS prevents emotionally and pedagogically dangerous retrievals.

**Instantiation 1 — across 5 diagnostic queries:**
- 27 of 28 dangerous entries suppressed
- Mean rank drop: 37 positions
- Maximum rank drop: 62 positions
- 3 of 5 queries had a dangerous entry ranked 1st by cosine → corrected by CASS

**Instantiation 2 — across 6 diagnostic queries:**
- All 28 dangerous entries suppressed
- C_DIFF mean rank drop: 25.3 positions
- C_ERR mean rank drop: 55.7 positions
- 3 of 6 queries had a dangerous entry ranked 1st by cosine → corrected by CASS

---

## Corpora

### QESC — Quranic Emotional Situation Corpus (`QESC_v1.0.json`)

| Property | Value |
|---|---|
| Entries | 100 |
| Quranic figures | 26 (prophets + key figures including Hagar) |
| EI levels | 1–5 (~19–24 entries per level) |
| RT levels | 1–3 (27 / 29 / 23 entries) |
| Languages | English descriptions, multilingual paraphrases |
| HuggingFace | [lamyaa/QESC](https://huggingface.co/datasets/lamyaa/QESC) |

### PhonEx — French Phonics Exercise Corpus (`PHONICS_CORPUS_v1.0.json`)

| Property | Value |
|---|---|
| Entries | 100 |
| Error types | 5 (visual_confusion, vowel_substitution, syllable_omission, letter_reversal, blending_difficulty) |
| Difficulty levels | 1–5 (20 entries per level) |
| Balance | 4 entries per error type × difficulty combination |
| Target | Moroccan French primary school, ages 6–9 |

---

## Ablation Results

### Instantiation 1 (QESC)

| System | MHR | P@1 |
|---|---|---|
| BM25 baseline | 1.000 | 0.00 |
| Cosine-only (α=1) | 1.391 | 0.09 |
| CASS + C_AEA only | 1.391 | 0.13 |
| CASS + C_RES only | 1.435 | 0.09 |
| **Full CASS** | **1.739** | **0.22** |

### Instantiation 2 (PhonEx)

| System | MHR | P@1 |
|---|---|---|
| BM25 baseline | 1.450 | 0.10 |
| Cosine-only (α=1) | 1.350 | 0.10 |
| CASS + C_DIFF only | 1.400 | 0.05 |
| CASS + C_ERR only | 1.500 | 0.15 |
| **Full CASS** | **1.850** | **0.30** |

---

## Citation

@unpublished{sadouk2025cass,
  title  = {CASS: A Context-Aware Semantic Similarity Framework
             for Safe Retrieval in Child-Facing Educational Applications},
  author = {Sadouk, Lamyaa and Gadi, Taoufiq},
  note   = {Manuscript submitted for publication},
  year   = {2025}
}

---

## License

CC BY-NC 4.0 — Creative Commons Attribution-NonCommercial 4.0 International

---

*Built for children in Morocco and beyond.*

---

## License Details

This work is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

**You are free to:**
- Share — copy and redistribute the material
- Adapt — remix, transform, and build upon the material

**Under the following terms:**
- **Attribution** — You must give appropriate credit and cite the paper above
- **NonCommercial** — You may not use the material for commercial purposes

For commercial licensing inquiries, contact the author.
