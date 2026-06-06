# Culturally Adaptive Retrieval-Based Framework for Multilingual Information Disorder Assessment

A retrieval-augmented in-context learning framework for culturally adaptive LLM assessment of multilingual information disorder. Developed as part of a Master's thesis in Language Technologies and Digital Humanities at the University of Turin (2026).

What is it about?
Extended a multilingual baseline study on the InDor corpus to investigate where current LLMs exhibit cultural blindness and produce culturally misaligned outputs in information disorder assessment. Designed and evaluated a retrieval-based in-context learning framework that connects model reasoning to community-annotated exemplars dynamically retrieved through semantic similarity at inference time. Evaluated across Persian (Farsi) and Italian using a combination of automated metrics and native-speaker human evaluation.

---

## Thesis

**Title:** Culturally Adaptive Explainable LLM Assessment for Multilingual Information Disorder

**Author:** Maziar Kianimoghadam Jouneghani

**Institution:** University of Turin

**Related papers:** 
1) Thesis summary paper as an ongoing study, introducing the full framework: [arXiv:2603.27356](https://arxiv.org/abs/2603.27356). presented at the InDor26 Workshop, LREC-COLING 2026.
2) Full thesis. Implementation and pilot evaluation of this framework on Italian and Persian (Farsi) information disorder. To be made available in the near future.
---

## Repository Structure

```
culturally-adaptive-prompting/
│
├── pipelines/               # prompt pipelines for all four experimental conditions
│   ├── b0_farsi.py          # zero-shot baseline, Farsi
│   ├── b0_italian.py        # zero-shot baseline, Italian
│   ├── b1_farsi.py          # static few-shot baseline, Farsi
│   ├── b1_italian.py        # static few-shot baseline, Italian
│   ├── m1.py                # dynamic retrieval, English instructions (both languages)
│   ├── a1_farsi.py          # dynamic retrieval, Farsi instructions
│   └── a1_italian.py        # dynamic retrieval, Italian instructions
│
├── preprocessing/           # InDor corpus cleaning pipeline
│   └── cleaning_indor.py    # 8-step cleaning, splitting, and bank construction
│
├── retrieval/               # exemplar bank indexing
│   └── build_bge_m3_index.py  # encodes the exemplar bank using BGE-M3
│
├── evaluation/              # automated metric computation and win analysis
│   ├── compute_metrics.py     # severity F1, token overlap F1, BERTScore
│   ├── win_analysis.py        # rationale-level win/loss analysis across conditions
│   └── evaluation_exclusions.txt  # log of items excluded from evaluation
│
├── human_evaluation/        # native speaker evaluation
│   ├── rationale_extraction.py    # selects paired rationales for A/B evaluation
│   ├── compute_human_results.py   # computes stratum 1 and stratum 2 results
│   ├── farsi_form.pdf             # evaluation form shown to Farsi evaluators
│   └── italian_form.pdf           # evaluation form shown to Italian evaluators
│
└── resources/
    └── dataset_landscape.csv  # curated review of 108 information disorder datasets
```

---

## Getting Started

### Requirements

```bash
pip install pandas numpy scikit-learn langdetect evaluate sentence-transformers matplotlib seaborn
```

Or install all dependencies at once:

```bash
pip install -r requirements.txt
```

### Input Data

This framework is built on the [InDor corpus](https://lrec.elra.info/lrec2026-main-515). You will need the raw Italian and Farsi JSONL files (`indor_it.jsonl`, `indor_fa.jsonl`). Place them in a `data/` folder before running the cleaning pipeline.

Pipeline result raw CSVs and human evaluation form responses are not included in this repository. See the README files in each folder for the expected file naming convention.

---

## How to Run

Run the steps in the following order:

1. **Clean the corpus** — `preprocessing/cleaning_indor.py`
   Produces the master clean pool, pilot test set, and exemplar bank for each language.

2. **Build the retrieval index** — `retrieval/build_bge_m3_index.py`
   Encodes the exemplar bank using BGE-M3 and saves the indexed pickle files.

3. **Run the pipelines** — `pipelines/`
   Run the relevant script for each condition, language, and model. Set your OpenRouter API key as an environment variable: `export OPENROUTER_API_KEY=your_key_here`.

4. **Evaluate** — `evaluation/compute_metrics.py`
   Computes severity classification (macro F1), span detection (token overlap F1), and rationale quality (BERTScore F1) across all conditions.

5. **Win analysis** — `evaluation/win_analysis.py`
   Computes item-level detection and quality advantage of each adaptive condition over the static baseline.

6. **Human evaluation** — `human_evaluation/`
   Use `rationale_extraction.py` to select rationale pairs from the best-performing model, then `compute_human_results.py` to compute stratum 1 and stratum 2 results from the form responses.

---

## Human Evaluation

The human evaluation asked native speakers of Persian and Italian community to rate rationale pairs from B1 and M1 on a 1–4 scale of cultural appropriateness in a blind A/B test, and to validate asymmetric detection cases. Seven evaluators participated per language. The evaluation forms, including extracted rationales to compare and body news texts in both languages are included in `human_evaluation/` as PDFs. Raw form responses are not published. See `human_evaluation/README.md` for details on the expected input format.

---

## Dataset Landscape

`resources/dataset_landscape.csv` contains a curated review of 108 fake news and information disorder datasets, covering modality, language, annotation type, accessibility, and annotation depth such as availability of spans and rationales. This is a community resource intended to support researchers navigating the dataset landscape in this domain.

---

## Ethical Considerations

This framework uses data from the InDor corpus, which was collected and annotated under an institutional ethics committee approval. All InDor annotator identifiers in the original files are numeric and the mapping to personal identities is held exclusively by the Aequa-Tech, srl. research consortium. Human evaluators in this study participated voluntarily and anonymously. No personal data was collected or stored. For full details see Chapter 7 of the thesis.

The evaluation forms in `human_evaluation/` and the in-context examples used in the pipelines are drawn from the InDor corpus, which may contain content that is politically sensitive, offensive, or otherwise disturbing, including racist, sexist, or violent language. This material is included strictly for research transparency and reproducibility purposes.

---

## Citation

If you use this framework, the dataset landscape sheet, or the evaluation methodology, please cite the study proposal summary paper or the full thesis to be made publically available soon:

```bibtex
@misc{jouneghani2026culturallyadaptiveexplainablellm,
  title         = {Culturally Adaptive Explainable {LLM} Assessment for Multilingual Information Disorder: A Human-in-the-Loop Approach},
  author        = {Kianimoghadam Jouneghani, Maziar},
  year          = {2026},
  eprint        = {2603.27356},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2603.27356},
  note          = {Presented at the InDor26 Workshop, LREC 2026}
}
```


---

## Get in Touch

If you find any issues with the code, the dataset landscape sheet, or anything else in this repository, feel free to open an issue or reach out directly. Feedback and contributions are welcome.

**Website:** [maziarkiani.github.io](https://maziarkiani.github.io)

---

## License

This repository is released for academic and non-commercial use. The InDor corpus is licensed for non-commercial use only. Please refer to the original InDor repository for its full license terms.
