# retrieval/

## `build_bge_m3_index.py`

Builds the Exemplar Bank that the M1 and A1 pipelines query at inference time. Takes the per-language exemplar bank CSVs produced by `preprocessing/cleaning_indor.py` and encodes each article's news text into a BGE-M3 embedding, then saves the whole thing as a pickle file.

**What gets encoded:** only the `text` column, meaning the news article body itself. The label, spans, and rationale for each item stay in the dataframe as plain columns, sitting right next to the embedding in the same row. They aren't encoded into the vector; they just travel alongside it. This matters because at retrieval time, similarity is computed purely on article content, and once the closest matches are found, the script looks up their full annotation directly from the same row rather than trying to reconstruct anything from the vector itself. See Chapter 5, Section 5.3 for the full explanation of why retrieval works this way.

Embeddings are computed with `normalize_embeddings=True`, so they come out as unit vectors. This is what makes cosine similarity meaningful at retrieval time in the pipeline scripts.

**Output:** one pickle file per language (`{lang}_bge_m3_bank.pkl`), each containing the original exemplar bank columns plus a `bge_m3_embedding` column.

**To run it:**
```bash
pip install pandas torch sentence-transformers
python build_bge_m3_index.py
```

Update the four path variables at the top of the script (`FA_INPUT_CSV`, `FA_OUTPUT_PKL`, `IT_INPUT_CSV`, `IT_OUTPUT_PKL`) to point at your own exemplar bank CSVs and desired output locations. Runs on GPU automatically if one's available, falls back to CPU otherwise.

This needs to run before any of the M1 or A1 scripts in `pipelines/`, since they load these pickle files directly to do their retrieval.

## Notes

- Model used: `BAAI/bge-m3`, a multilingual dense embedding model covering 100+ languages, including both Farsi and Italian in a shared embedding space.
- At inference time, the unseen article is encoded with this same model and matched against the bank using cosine similarity, retrieving the top k=4 most relevant exemplars. That retrieval logic lives in the pipeline scripts themselves (`pipelines/m1.py`, `pipelines/a1_farsi.py`, `pipelines/a1_italian.py`), not here; this script only builds the index they query.
- See Chapter 5, Section 5.3 for the retrieval mechanism and Figure 5.1 for the full architecture diagram.
