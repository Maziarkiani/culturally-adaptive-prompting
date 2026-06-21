# resources/

## `dataset_landscape.csv`

A curated review of 108 publicly available or documented fake news and information disorder datasets, compiled as part of the literature review in Chapter 2. This isn't a list pulled from a single survey paper; it started from the foundational lists in existing landscape surveys (Hussain et al. 2025) and was extended through additional searches across Hugging Face, GitHub, and Kaggle to check accessibility and document key characteristics.

**Columns:**
- `Dataset` — name of the resource
- `Modality` — text, image, video, multimodal, or combinations with social context/graph data
- `Language` — language(s) covered
- `Labeled?` — whether the dataset has veracity labels
- `Has Spans?` — whether it includes span-level annotations marking specific problematic text
- `Has Rationales?` — whether it includes any form of natural-language explanation or rationale
- `Includes True News?` — whether the dataset includes genuine, non-manipulative content alongside the problematic examples, or only the latter
- `Year` — publication year, where available
- `Accesibility` — public or private
- `Link` — source URL

**Why this exists:** the main finding driving Chapter 2's argument is that almost none of the existing datasets in this space combine multilingual coverage with explanation-oriented annotations. The numbers back this up directly: only 7 of the 108 datasets include span annotations, and only 10 include any form of rationale. 63 are English-only. This sheet is the evidence behind that claim, and it's released as a standalone resource so other researchers don't have to redo this search from scratch.

**A caveat worth keeping in mind:** dataset characteristics here were compiled from published papers, repository documentation, and available web resources. Where direct access to a dataset wasn't possible, details were inferred from published descriptions, so minor inconsistencies may exist. Some datasets are listed as private or restricted, but their characteristics (modality, language, spans, rationales, etc.) are still documented here based on what the originating paper or repository describes publicly, even if the data itself wasn't directly inspected. A number of datasets are only accessible by contacting the original authors or data provider directly (commonly by email) rather than through an open download link, so "Public" or "Private" here doesn't always mean instantly downloadable.

Links were verified at the time of compilation, but dataset hosting changes over time: repositories get moved, renamed, taken down, or restructured by their original maintainers. This sheet reflects the state of each resource as found during this review and is not continuously maintained against upstream changes.

See Chapter 2, Section 2.4 of the thesis for the full discussion of what this landscape review found and how it motivated the InDor corpus design.
