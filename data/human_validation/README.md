# Human Validation Instrument

Three expert raters each independently completed an identical 120-pair paired-comparison instrument,
stratified by how far apart the AI judge committee had scored each pair (40 near-tie, 40 moderately
separated, 40 clearly different pairs, in the same order for every rater).

## Files

- `pairwise_review_rater1.html`, `pairwise_review_rater2.html`, `pairwise_review_rater3.html` —
  the self-contained interactive rating instrument, one per rater. Each embeds the same 120 pairs
  (query + two candidate paper titles/abstracts) and lets the rater choose which paper better answers
  the query, or mark the pair "about equal." Progress auto-saves in the browser and the instrument
  resumes where the rater left off; a numbered map allows revisiting and changing any earlier rating.
  **Open these directly in a browser** (the interactive rating UI will not render in a code-viewer
  preview pane).
- `pairwise_review_rater{1,2,3}_completed.csv` — each rater's completed judgments: `pair_id` (encodes
  the query and the two candidate paper record IDs) and `chosen_paper_id` (the record ID chosen, or
  the literal string `equal`).

## Use

Per-rater agreement with the AI committee (by difficulty band) and rater-to-rater concordance (the
human-to-human reference band against which committee-to-human agreement is read) are reported in the
accompanying manuscript.
