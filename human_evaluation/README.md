# human_evaluation/

This folder covers the native-speaker evaluation from Chapter 6, Section 6.6: how items were selected for the blind A/B test, and how the Google Form responses were turned into the numbers reported in the thesis.

> [!WARNING]
> The evaluation forms below contain in-context examples drawn from the InDor corpus, which may include content that is politically sensitive, offensive, or otherwise disturbing, including racist, sexist, or violent language. This material is presented strictly for research transparency and reproducibility.

---

## `rationale_extraction.py`

Builds the item pool used for the blind A/B evaluation, comparing Maverick's B1 and M1 rationales. Selection logic is described in Chapter 6, Section 6.6.

Items are split into two strata:
- **Stratum 1:** items where both B1 and M1 produced a rationale on the same article, and the human-reference label is actually Problematic (not None). That last part matters: if both conditions generated a rationale for an article that humans labeled as clean, that's a false alarm on both sides, not a meaningful comparison, so those get dropped too. After this full filtering chain, the eligible pool was 45 items for Farsi and 23 for Italian, and 10 per language were randomly sampled (seed 42) for the actual paired A/B rating, with display order randomized so evaluators don't know which option is which.
- **Stratum 2:** items where only one of the two conditions produced a rationale at all, and the human reference confirms the article really is Problematic. This isolates genuine detection divergence rather than cases where staying silent was the correct call. All eligible Stratum 2 items are shown to evaluators with no further sampling: 11 for Farsi, 6 for Italian.

When picking which rationale to show for a Stratum 1 item, the script follows three fallback rules: first try to match B1 and M1 rationales that point to overlapping spans, then fall back to whichever rationale's span best matches the human-reference annotation, and finally just take the first rationale generated under each condition if neither of the above applies.

**Output:** one CSV per language (`{lang}_human_eval_mapping.csv`) listing each selected item, its stratum, the news text, and which rationale goes in option A vs option B.

**To run it:** point `BASE_DIR` at your pipeline result CSVs, then:
```bash
python rationale_extraction.py
```

---

## `compute_human_results.py`

Takes the raw Google Form export and the mapping file from `rationale_extraction.py`, and computes the actual evaluation results: mean scores per setup, gender breakdowns, and stratum 2 validation rates.

For Stratum 1, it reshapes the form responses into a long format (one row per evaluator per item per setup) and reports mean 1-4 ratings overall and by gender, plus the gender gap as the absolute difference between male and female mean scores per setup.

For Stratum 2, it checks each evaluator's answer against the expected validation response and reports how often each setup's detection was confirmed by native speakers.

This script expects two files per language that aren't included in the repo for privacy reasons: the mapping CSV from `rationale_extraction.py`, and the raw form responses exported from Google Sheets. See the note in the script header for the expected column layout.

**To run it:** set `BASE_DIR` to the folder with your mapping and form response files, then:
```bash
python compute_human_results.py
```

---

## `farsi_form.pdf` and `italian_form.pdf`

These are the actual Google Forms shown to evaluators, exported as PDF, with the news text and rationale pairs exactly as native speakers saw and rated them.

---

## Notes

- 7 native speakers per language participated.
- Raw form responses are not published here to protect evaluator privacy. Only the blank forms above are included.
- See Chapter 6, Section 6.6 for the full rubric and procedure.
