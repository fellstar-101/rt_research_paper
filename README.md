# Neural Machine Translation Quality Estimation

**Can round-trip translation metrics predict translation quality without human references?**

This repository contains the research paper 📄 from my graduate Deep Learning Systems course at the University of Chicago (Fall 2025), co-authored with Paulo Rodrigues.

---

## Overview

We investigated two questions in neural machine translation:

1. **Pivot languages**: When (if ever) should translation systems route through intermediate languages?
2. **Quality estimation**: Can round-trip (RT) translation metrics serve as reference-free quality signals?

Using a 120B parameter LLM on the FLORES-200 benchmark, we found that:
- Modern LLMs perform best with **direct translation** — pivot languages in most cases introduce information loss that outweighs potential benefits
- RT metrics function as effective **"smoke detectors"** (AUC = 0.94 for flagging failures) but add minimal signal beyond COMETkiwi for fine-grained quality ranking

---

## My Contribution

I designed and owned the statistical analysis framework for **RQ3 (Round-Trip Consistency Analysis)**, including:

- **ROC AUC analysis** across quality thresholds to test discriminative power
- **Stratified correlation analysis** to test the "smoke detector" hypothesis—whether RT metrics work better at detecting failures than distinguishing good from excellent
- **Spearman ranking within quartiles** to assess fine-grained ranking ability
- **Multiple regression** to test whether RT adds predictive value beyond COMETkiwi
- **Threshold adjustment methodology** to handle class imbalance across target languages (NPI)

---

## Key Findings

| Finding | Implication |
|---------|-------------|
| RT metrics show strong correlation in low-quality stratum but near-zero in high-quality stratum | RT is a coarse classifier, not a fine ranker |
| RT achieves AUC = 0.94 for distinguishing good vs. catastrophically bad translations | Effective as a "safety rail" for flagging translations needing human review |
| RT adds little beyond COMETkiwi in multiple regression | Minimal additional value over COMETkiwi, but may be valuable for if other quality indicators are used e.g. LaBSE consine similarity |
| Linguistically similar pivots degrade less than distant ones | If pivoting is unavoidable, choose pivots from the same language family |

---

## Methods

- **Model**: 120B parameter LLM (OSS) via CELS academic API
- **Dataset**: FLORES-200 benchmark (996 parallel sentences across 200+ languages)
- **Quality metrics**: COMET (semantic), chrF (surface)
- **RT metrics**: LaBSE cosine similarity for semantic round-trip consistency
- **Target languages (RQ3)**: Turkish, Japanese, Nepali
- **Pivot languages (RQ3)**: French, German, Russian, Mandarin

---

## Citation

If you reference this work:
```
Rodrigues, P., & Li, J. (2025). Translation: When should AI take a detour? 
University of Chicago, CMSC 35200 Deep Learning Systems.
```

---

## Acknowledgments

- Co-author: Paulo Rodrigues.
- Course: CMSC 35200 Deep Learning Systems, University of Chicago, Professor Rick Stevens.
- Compute: Professor Rick Stevens, UChicago CS Department, and Argonne ALCF for API access to OSS 120B.
