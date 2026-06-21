# Culturally Adaptive Retrieval-Based Framework for Multilingual Information Disorder Assessment

A retrieval-augmented in-context learning framework for culturally adaptive LLM assessment of multilingual information disorder. Built as part of a Master's thesis in Language Technologies and Digital Humanities at the University of Turin (2026).

## What's this about?

Information disorder doesn't look the same everywhere. The same manipulative framing that's obvious to a native speaker can slip right past a model trained mostly on English data, a problem this thesis calls *cultural blindness*. And even when a model does flag something as problematic, the explanation it gives often doesn't reflect how the target community actually reasons about it, what the thesis calls *culturally misaligned rationales*.

This project starts from a multilingual baseline study on the [InDor corpus](https://lrec.elra.info/lrec2026-main-515), where evaluation of two MoE LLMs under zero-shot and static few-shot prompting showed exactly these limitations. A fixed set of few-shot examples just can't cover the thematic and cultural range of content these models are asked to assess. So the framework built here goes one step further: instead of a static prompt, it builds a community-annotated Exemplar Bank per language, encodes it with the multilingual retrieval model BGE-M3, and dynamically retrieves the most semantically similar items with human-written rationales for each unseen article at inference time. The aim is to align model's reasoning more deeply in real community reasoning and explanations on manipulative content.

Everything is evaluated on Persian (Farsi) and Italian, using a mix of automated metrics (severity F1, span overlap F1, rationale BERTScore F1) and native-speaker human evaluation on the rationales.

---

## Thesis

**Title:** Culturally Adaptive Explainable LLM Assessment for Multilingual Information Disorder

**Author:** Maziar Kianimoghadam Jouneghani

**Institution:** University of Turin

**Related papers:**
1. Thesis summary paper, presenting the framework as an ongoing study: [arXiv:2603.27356](https://arxiv.org/abs/2603.27356). Presented at the InDor26 Workshop, LREC-COLING 2026.
2. Full thesis, with the complete implementation and pilot evaluation on Italian and Persian (Farsi) information disorder. To be made available soon.

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
├── evaluation/               # automated metric computation and win analysis
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

Each folder has its own README going into more detail on what each folder has and how it maps to the thesis chapters.

---

## Getting Started

### Requirements

```bash
pip install pandas numpy scikit-learn langdetect evaluate sentence-transformers matplotlib seaborn
```

Or just:

```bash
pip install -r requirements.txt
```

### Input Data

This framework is built on the [InDor corpus](https://lrec.elra.info/lrec2026-main-515). You'll need the raw Italian and Farsi JSONL files (`indor_it.jsonl`, `indor_fa.jsonl`) placed in a `data/` folder before running the cleaning pipeline.

Pipeline result CSVs and human evaluation form responses aren't included in this repository for size and privacy reasons. Check the README in each folder for the expected file naming convention if you're plugging in your own data.

---

## How to Run

Run things in this order:

1. **Clean the corpus** — `preprocessing/cleaning_indor.py`
   Takes the raw InDor JSONL files and produces the clean pool, the pilot test set, and the exemplar bank for each language.

2. **Build the retrieval index** — `retrieval/build_bge_m3_index.py`
   Encodes the exemplar bank with BGE-M3 and saves the indexed pickle files that the M1 and A1 pipelines need.

3. **Run the pipelines** — `pipelines/`
   Run whichever script matches the condition, language, and model you want. Set your OpenRouter API key first: `export OPENROUTER_API_KEY=your_key_here`.

4. **Evaluate** — `evaluation/compute_metrics.py`
   Computes severity classification (macro F1), span detection (token overlap F1), and rationale quality (BERTScore F1) across all conditions.

5. **Win analysis** — `evaluation/win_analysis.py`
   Goes item by item to see how often each adaptive condition actually beats the static baseline, rather than just comparing averages.

6. **Human evaluation** — `human_evaluation/`
   Run `rationale_extraction.py` first to pick which rationale pairs go in front of evaluators, then `compute_human_results.py` once the form responses are in.

---

## Human Evaluation

Native speakers of Persian and Italian rated rationale pairs from B1 and M1 on a 1-4 scale of cultural appropriateness in a blind A/B test, and separately validated cases where only one condition detected a problem. Seven evaluators took part per language. The forms themselves, including the rationales and news text shown to evaluators, are included as PDFs in `human_evaluation/`. See `human_evaluation/README.md` for the full breakdown.

---

## Dataset Landscape

`resources/dataset_landscape.csv` is a curated review of 108 fake news and information disorder datasets, covering modality, language, annotation type, accessibility, and whether they include span-level or rationale-level annotations. It's released as a standalone resource for anyone else trying to navigate this space, not just as supporting material for this thesis.

---

## Ethical Considerations

This framework uses data from the InDor corpus, collected and annotated under institutional ethics committee approval. All InDor annotator identifiers in the original files are numeric, and the mapping to personal identities is held exclusively by the Aequa-Tech srl research consortium. Human evaluators in this study took part voluntarily and anonymously, and no personal data was collected or stored. Full details are in Chapter 7 of the thesis.

Worth flagging directly: the evaluation forms in `human_evaluation/` and the in-context examples used throughout the pipelines are drawn from the InDor corpus, which can contain content that's politically sensitive, offensive, or otherwise disturbing, including racist, sexist, or violent language. This is included strictly for research transparency and reproducibility.

---

## Citation

If you use this framework, the dataset landscape sheet, or the evaluation methodology, please cite the study proposal summary paper for now; the full thesis citation will be added once it's publicly available:

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

Found an issue with the code, the dataset landscape sheet, or anything else here? Feel free to open an issue or reach out directly. Feedback and contributions are genuinely welcome.

**Website:** [maziarkiani.github.io](https://maziarkiani.github.io)

---

## License

This repository is released for academic and non-commercial use. The InDor corpus itself is licensed for non-commercial use only; check the original InDor repository for its full license terms.
