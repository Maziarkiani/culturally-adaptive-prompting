# preprocessing/

This is the cleaning pipeline that turns the raw InDor annotations into the clean pool, the pilot test set, and the Exemplar Bank used everywhere else in the repo. See Chapter 6, Section 6.1 for the full reasoning behind each step.

## `cleaning_indor.py`

Runs eight cleaning steps in sequence on the raw `indor_it.jsonl` and `indor_fa.jsonl` files, then splits the result into a stratified test set and exemplar bank per language.

| Step | What it does | Italian | Farsi |
|---|---|---|---|
| 0 | Raw annotations | 3,109 | 1,949 |
| 1 | Remove rows with missing or invalid (N/A) labels | -665 | -243 |
| 2a | Manual blacklist (ads, JS-disabled error pages, wrong-language articles caught by human review) | -178 | -106 |
| 2b | Regex filter for structural body-text errors not caught by the blacklist | -2 | -7 |
| 3 | Remove Problematic items missing spans, span labels, or rationales | -297 | -10 |
| 4 | Remove None items that have spans or rationales attached (shouldn't happen, but it does) | -52 | -86 |
| 5 | Remove Problematic items whose rationale was written in English instead of the target language, detected with `langdetect` | -12 | -56 |
| 6 | Exclude articles where one annotator said None and another said Problematic | -384 | -575 |
| 7 | Deduplicate to one annotation per article, keeping the longest rationale when there's a choice | -473 | -309 |
| 8 | Final integrity check: language detection on the article body itself, plus a duplicate-text check, to catch wrong-language articles and repeated texts that slipped through everything else | -55 (48 wrong-language + 7 duplicates) | n/a |
| | **Clean pool** | **991** | **557** |
| | **Final test set** | **100** | **100** |
| | **Final exemplar bank** | **891** | **457** |

Step 6 is the big one for Farsi, reflecting how much annotators disagreed on whether certain articles were problematic at all (see the inter-annotator agreement discussion in Chapter 3). Step 8 only applies to Italian; the Farsi subset didn't have this problem.

After cleaning, the script does a stratified train/test split: 100 items held out per language as the pilot test set, with the rest going into the Exemplar Bank, stratified by severity label so the class distribution is preserved on both sides of the split. It also asserts there's no overlap in `text_id` between the test set and the bank, so there's no chance of testing on something the retrieval system could also retrieve.

**Output, per language:**
- `{lang}_master_clean.csv` — the full clean pool after all 8 steps
- `{lang}_pilot_test.csv` — the 100-item held-out test set
- `{lang}_exemplar_bank.csv` — everything else, used to build the retrieval index in `retrieval/`

**To run it:** put `indor_it.jsonl` and `indor_fa.jsonl` in a `data/` folder, then:
```bash
pip install pandas numpy langdetect scikit-learn
python cleaning_indor.py
```

## Notes

- The manual blacklist IDs (Step 2a) and the integrity-check removals (Step 8) were both identified through human review in the original run; they're hardcoded here as fixed ID lists so the pipeline is exactly reproducible rather than re-running a manual review each time.
- `DetectorFactory.seed = 42` is set before any language detection runs, since `langdetect` is non-deterministic by default and this keeps Step 5 and Step 8 reproducible across runs.
- See Chapter 6, Table 6.1 for the full step-by-step row counts this script produces.
