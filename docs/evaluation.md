# Evaluation

Phase 1 checks CSV validation, allowed-category matching, deduplication, stable identifiers, price handling, description quality, provenance and reproducibility. Run `python -m unittest discover -s tests -v`. See [observed data-quality results](data_quality.md) and the [handoff](handoffs/phase-01.md).

## Phase 5 evaluation corpus (M5)

17 questions (5 factual, 4 comparison, 4 multipart, 4 unsupported) run against the live pipeline (`src.hybrid.make_retriever` + `src.rag_pipeline.answer_question`, Gemini `gemini-3.1-flash-lite`, hybrid retrieval) on the Phase 4 branch (`feat/phase-04-streamlit-app` at `7506627`). Runner: `scripts/phase5_evaluation.py`. Full per-question results: [`data/evaluation/results.csv`](../data/evaluation/results.csv).

**Retrieval success@5:** 9/9 (100%) on the 9 questions with a known expected product (5 factual + 2 comparison + 1 multipart; the other comparison and multipart questions had no single expected product to check against).

**Unsupported-question handling:** 4/4 correctly refused with no invented information (`capital of France`, delivery policy, return policy, "do you sell smartphones" — the catalogue has 0 phones).

**Observed but not reproduced on demand:** Gemini occasionally stalls past the 15 s timeout even with the SDK's built-in retry (`max_retries=2`), matching the Phase 4 handoff's note on this. Happened once in ~34 live calls across this run and earlier manual testing. Not treated as a failure case below since it is already documented and accepted for the demo (warm the cache beforehand, as the Phase 4 handoff recommends).

### Failure case 1: budget constraint silently dropped on a multi-part question

- **Question:** "I need a home theatre system under 5000 with good power output, what do you suggest?"
- **Expected:** `understand()` extracts `max_price=5000`, so retrieval is filtered to the budget before the LLM ever sees the results.
- **Actual:** `src.query_understanding.understand()` returns `max_price=None, has_price_limit=False` for this phrasing. Retrieval therefore returns all 5 home theatre products unfiltered, including items at INR 18,990 and INR 6,499 — both well over the stated budget. The LLM happened to notice and excluded them from its own answer this time, but that is not a guaranteed, code-enforced behavior; it depends on the model reasoning correctly from unfiltered evidence every time.
- **Cause:** the budget-extraction heuristic in `src/query_understanding.py` does not reliably match "under `<number>`" when a trailing clause ("... with good power output") follows the number.
- **Proposed fix:** broaden the budget regex to not require the number to be the last token before a comma/clause boundary, and add a defense-in-depth price post-filter inside `answer_question()` that drops any source over `constraints.max_price` right before building the prompt, regardless of whether the regex caught the constraint.

### Failure case 2: in-budget, in-category wireless products not retrieved for a wireless-specific query

- **Question:** "Suggest wireless headphones under 2000 and say if any are foldable."
- **Expected:** at least one wireless product should appear in the top 5. The catalogue has 6+ wireless Bluetooth headsets under INR 2,000 with "Wireless" explicitly in the product name (e.g. "LIFE LIKE OTP-200... Wireless Bluetooth Headset" INR 599, "CHKOKKO Earbud Skin Wireless Bluetooth Headset" INR 610).
- **Actual:** all 5 retrieved chunks are wired headphones/earphones (verified directly against `make_retriever().search(...)`); zero wireless products surface. The LLM then correctly and honestly refuses ("The provided data only covers wired headphones and earphones") rather than inventing one — so there is no hallucination, but a real, answerable question goes unanswered.
- **Cause:** the "Headphones & Earphones" category has 51 products, the large majority wired; semantic + BM25 hybrid retrieval for a short query doesn't weight the "wireless"/"Bluetooth" terms highly enough to surface the minority wireless subset ahead of the much larger wired cluster.
- **Proposed fix:** at query time, detect connectivity keywords ("wireless", "bluetooth", "wired") in the question and apply them as an additional metadata filter/boost (would need a `connectivity` field added at chunking/indexing time, since it isn't currently a stored metadata field — only free text in the description); as a smaller interim fix, raise `k` for category-ambiguous queries so more of the long tail gets a chance to surface.

Both causes and fixes above are grounded in direct inspection of `src/query_understanding.py` and the retrieved evidence for these exact questions, not assumed.
