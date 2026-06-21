# pipelines/

The seven scripts here implement the four experimental conditions evaluated in the thesis: B0, B1, M1, and A1. Each one runs the same three-step task structure (severity classification, span detection, rationale generation) against an LLM through the OpenRouter API, but they differ in what context the model is given before it sees the unseen article. See Chapter 5 for the full design rationale and Chapter 6 for the results.

| File | Condition | Language(s) | What's different |
|---|---|---|---|
| `b0_farsi.py` | B0, zero-shot | Farsi | Target-language instructions only, no examples |
| `b0_italian.py` | B0, zero-shot | Italian | Target-language instructions only, no examples |
| `b1_farsi.py` | B1, static few-shot | Farsi | Same as B0, plus a fixed set of target-language examples |
| `b1_italian.py` | B1, static few-shot | Italian | Same as B0, plus a fixed set of target-language examples |
| `m1.py` | M1, dynamic retrieval | Both | English instructions, examples retrieved per article from the Exemplar Bank |
| `a1_farsi.py` | A1, dynamic retrieval | Farsi | Target-language instructions, examples retrieved per article |
| `a1_italian.py` | A1, dynamic retrieval | Italian | Target-language instructions, examples retrieved per article |

`m1.py` covers both languages in one file since the only language-dependent part is the retrieval bank it points to; everything else, including the instructions, stays in English regardless of which language is being processed. The A1 and B1 conditions are split per language because the instructions themselves are written in the target language.

## What stays constant across all seven

- Generation parameters: temperature 0.0, top_p 1.0, seed 42
- Three-step task structure: severity classification first, then span detection and rationale generation only if the article isn't classified as "none"
- Output parsing relies on `<PREDICTED_LABEL>`, `<SPANS>`, and `<RATIONALES>` tags, with one automatic retry if the model returns a malformed response
- Resume logic: if you rerun a script and the output CSV already has results for a row, that row is skipped

## What changes between conditions

- **Examples in the prompt:** none in B0, a fixed set in B1, dynamically retrieved top-k matches in M1 and A1
- **Instruction language:** target language in B0, B1, and A1; English in M1
- **Retrieval:** only M1 and A1 query the Exemplar Bank (see `retrieval/`) using BGE-M3 similarity search

## Running these

You'll need the Exemplar Bank already built (`retrieval/build_bge_m3_index.py`) before running M1 or A1. Set `OPENROUTER_API_KEY` as an environment variable, point the input/output paths at the top of each script to your data, and run the one you need:

```bash
export OPENROUTER_API_KEY=your_key_here
python b0_farsi.py
```

Switch `MODEL_NAME` at the top of each script to run a different model; the thesis uses LLaMA 4 Maverick and Mixtral-8x22B-Instruct.

For the actual reasoning behind why these four conditions were chosen and what each one is meant to isolate, see Chapter 5, Section 5.2.
