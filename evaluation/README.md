# evaluation/

This is where the numbers in Chapter 6 actually come from. Two scripts and one log file: `compute_metrics.py` gives you the headline scores (severity F1, span overlap F1, BERTScore), `win_analysis.py` digs deeper into rationale quality on an item-by-item basis, and `evaluation_exclusions.txt` is the audit trail of everything that got dropped before scoring and why.

---

## `compute_metrics.py`

Runs the full automated evaluation across all 16 conditions, that's 4 prompting setups (B0, B1, M1, A1) times 2 models (LLaMA 4 Maverick, Mixtral-8x22B) times 2 languages (Farsi, Italian). Produces a results table and a 3-panel comparison chart.

**What it computes, per condition:**
- **Severity classification.** Macro F1 across the four severity labels (none/slightly/moderately/highly). Macro averaging matters here because the label distribution is skewed toward "none," and macro F1 stops the majority class from hiding how the model does on the rarer, more interesting classes.
- **Span detection.** Token-overlap F1, computed with best-match pairing between predicted and human-reference spans rather than strict positional matching. This is intentionally more forgiving than exact match; the model just needs to land on roughly the right region of text, not character-perfect boundaries.
- **Rationale quality.** BERTScore F1 against the human-written reference rationale, computed only on items that actually have a human-reference rationale (i.e. items annotators flagged as problematic).

Before any of this happens, every row gets checked against a list of pipeline error tokens (FORMAT_ERROR, API_ERROR, SKIPPED_DUE_TO_SPANS_ERROR, and a few legacy variants from earlier pipeline runs). If a row has an error in any of the three output columns, the whole row is dropped from evaluation, not just from the metric where it failed. This keeps all three scores comparable on the exact same item set.

**Output:** a results table printed to console plus `results_all_metrics.png`, grouped bar charts comparing all four conditions side by side for each metric.

**To run it:** point `BASE_DIR` at the folder holding your 16 pipeline result CSVs (the ones the `pipelines/` scripts produce), then:
```bash
python compute_metrics.py
```

---

## `win_analysis.py`

This is the script behind the item-level "win/loss" tables in Chapter 6, the ones comparing B1 against M1 and A1 head-to-head rather than just looking at averaged scores. The idea is to answer a sharper question than the aggregate metrics can: not just whether the average is better, but on how many individual articles retrieval actually helps, and on how many it hurts.

For each language/model combo, it merges the B1, M1, and A1 result files on `text_id`, restricts to Problematic items (slightly, moderately, or highly; None items are excluded since there's no rationale to compare), and computes BERTScore for each condition's rationale against the same human reference.

From there it reports three things:
1. **Miss rates.** How often each condition produced literally nothing (a zero-similarity rationale, usually meaning a None misclassification or a content refusal).
2. **Detection advantage.** Items one condition catches that another completely misses.
3. **Quality advantage.** On the "fair" subset where both conditions produced something, which one scores higher against the human reference.

It also has a small helper, `print_example_items()`, that pulls out the full text of specific items for side-by-side qualitative inspection. This is what generated the worked examples discussed in the Farsi qualitative analysis (Section 6.5).

**To run it:** same `BASE_DIR` setup as above, then:
```bash
python win_analysis.py
```

---

## `evaluation_exclusions.txt`

A full log of every row dropped from evaluation, broken down by language, model, and condition, with the specific error reason per item. This exists mainly for transparency and reproducibility. If someone wants to check exactly which 55 Italian Maverick B0 items got excluded and why, this is where they look.

One thing worth flagging explicitly here since it matters for interpreting Chapter 6: the Italian Maverick B0 and A1 exclusions are not the same kind of error as everything else in this file. Diagnostic work (documented in Chapter 6, Section 6.4 and Chapter 7, Section 7.2.2) confirmed these were genuine model content refusals at the rationale generation step, not infrastructure failures. The model received the prompt, processed it, and returned an empty body for specific sensitive topics. Earlier pipeline runs logged this inconsistently as either API_ERROR or FORMAT_ERROR depending on the run; both are unified here as FORMAT_ERROR for clarity, but they refer to the same underlying refusal behavior. Every other exclusion in this file, the scattered Farsi Mixtral ones especially, is a normal transient pipeline issue (API timeout, malformed JSON, etc.) and was confirmed as such through item-by-item review.

This pattern is treated in the thesis as a methodological finding in its own right rather than a bug to fix: it points to a broader question about how safety alignment in commercial LLMs intersects with legitimate analytical tasks on sensitive topics, an effect that likely extends beyond this specific pipeline and could be worth studying more systematically across other information disorder or content-moderation contexts. See Chapter 6 and Chapter 7 for the full diagnostic and discussion.

---

## Notes

- Both scripts expect `text_id` to be a consistent join key across all condition files for a given language/model pair. If you're using your own data, make sure that column is present and unique.
- `compute_metrics.py` needs the `evaluate` library's BERTScore implementation, which downloads a BERT checkpoint on first run, so expect a short pause the first time you run it.
- See Chapter 5 (methodology) and Chapter 6 (results) of the thesis for the full reasoning behind these specific metric choices.
