# Phase 4 handoff

## Objective and scope

- Phase and date: Phase 4, Streamlit application and RAG enhancement. Prepared 14 November 2026.
- Outgoing lead M4 (Streamlit & Integration), receiving lead M5 (Evaluation & Release). Team member names are pending.
- Objective: a Streamlit chat app on the Phase 3 `answer_question()` pipeline that shows answers with inspectable sources and historical prices, plus the optional Hybrid RAG enhancement with a baseline comparison.
- Completed: all four PRD tasks. Task 4 (presentation machine) is done on M4's MacBook; repeat it on the presentation laptop if that is a different machine.
- Added beyond the PRD: follow-up question retrieval (the open Phase 3 limitation, owner M3/M4), product cards with image, rating and product link, an answer cache, and a faster default model.
- Decisions:
  - **Model `gemini-3.1-flash-lite`** instead of Phase 3's `gemini-3.5-flash`. On the same prompt 3.5 Flash took about 38 s per answer (it reasons first; about 26 s with `reasoning_effort=none`), Flash-Lite 1.5-7 s with valid citations. Gemini occasionally stalls for about 40 s, so `LLM_TIMEOUT_SECONDS` is now 15 s (was 45) and the stalled request is retried.
  - **Follow-ups:** in a conversation, a question that names no product category ("which is the best?", "is the zebie good?") is retrieved together with the most recent question that did, and keeps its budget and category filters unless it sets its own. A keyword list ("those", "it") missed 7 of 12 natural phrasings, so it was replaced by this rule.
  - **Hybrid RAG adopted** on the evidence below. `RETRIEVAL_MODE=semantic` switches back.
  - **Images and ratings** come from a separate lookup built from the raw CSV, so the accepted Phase 1 CSV and the Phase 2 index and their hashes are unchanged.
  - **Not deployed.** The PRD asks for a local launch on the presentation machine. Hosting would need the data and index on the server, around 1 GB RAM for torch, and a shared Gemini quota; record it as future work if the team wants it.

| PRD task | Where |
| --- | --- |
| 1 Chat with history, clear chat, example prompts, loading and error states | `app.py` |
| 2 Recommendations, specifications and prices with inspectable sources; historical prices labelled | `app.py` product cards and "All retrieved records"; `src/ui_helpers.py` |
| 3 Hybrid RAG compared with the baseline | `src/hybrid.py`, `scripts/hybrid_eval.py`, [evaluation](../hybrid_evaluation.md) |
| 4 Launch and chat flows on the presentation machine | Validation below (M4's MacBook) |

## Snapshot and artifacts

- Repository: https://github.com/bharat0722/flipkart-ai-advisor, base `main` at `75bda24`, branch `feat/phase-04-streamlit-app`.
- Reviewed commit and PR: `d1cc614`, [PR #9](https://github.com/bharat0722/flipkart-ai-advisor/pull/9). Phase 3 review: [PR #8](https://github.com/bharat0722/flipkart-ai-advisor/pull/8). Milestone [Phase 4](https://github.com/bharat0722/flipkart-ai-advisor/milestone/4), no issues filed yet. Tag after acceptance: `phase-4-complete`.
- New files: `src/ui_helpers.py`, `src/hybrid.py`, `scripts/build_display_data.py`, `scripts/hybrid_eval.py`, `tests/test_app.py`, `tests/test_hybrid.py`, `docs/hybrid_evaluation.md`, this handoff.
- Changed files: `app.py` (was a placeholder), `src/rag_pipeline.py` and `src/query_understanding.py` (follow-ups), `src/settings.py` (timeout, retrieval mode, BM25/RRF constants), `tests/test_rag.py`, `requirements.txt` (`streamlit==1.67.0`), `.env.example`, `README.md`, `docs/phase_status.json`, `docs/PRD.md` (generated table).
- Interfaces:

```python
from src.hybrid import make_retriever, HybridRetriever
retriever = make_retriever()          # RETRIEVAL_MODE: "hybrid" (default) or "semantic"
retriever.search(query, k, max_price=..., min_price=..., categories=[...])   # same contract as Retriever.search
from src.query_understanding import is_follow_up   # True when the question names no product category
answer_question(question, history, retriever=retriever)  # unchanged signature; follow-ups now use history for retrieval
```

- Index and data: unchanged from Phase 2 (288 products, 863 chunks, `all-MiniLM-L6-v2`, `flipkart_electronics_v1`). New generated file `data/processed/product_display.json` (288 products, 288 with images, 91 with ratings), ignored by Git.

## Reproduction

- Runtime: macOS arm64, Python 3.12.12 (`uv venv --python 3.12`), `pip check` clean, streamlit 1.67.0.
- Setup and launch: follow the README sections in order: setup, dataset, Phase 2 index, then "Phase 4: run the Streamlit app" (`python scripts/build_display_data.py`, copy `.env.example` to `.env`, `streamlit run app.py`).
- Configuration variable names: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `LLM_MODEL` (`.env.example` now defaults to Gemini and `gemini-3.1-flash-lite`), optional `RETRIEVAL_MODE`.
- Checks: `RUN_MODEL_TESTS=1 python -m unittest discover -s tests`, `python -m compileall -q app.py src tests scripts`, `python scripts/sync_prd.py --check`, `python scripts/hybrid_eval.py`.
- The live LLM test reads the key from the shell, so export `.env` first: `set -a; . ./.env; set +a; RUN_LLM_TESTS=1 python -m unittest tests.test_rag -v`.

## Validation

| Check | Expected | Actual or evidence | PASS / FAIL / NOT RUN | Owner |
| --- | --- | --- | --- | --- |
| Fresh clone follows README only | Data, index, display data, tests and app launch | Fresh clone of `c561ddf`: cleaned SHA-256 `f13d7836…`, 863 chunks, 288 display records, 96 tests OK (1 skipped, live LLM), compile and PRD checks clean, app health endpoint up | PASS | M4 |
| Full suite | Phase 1-3 tests still pass, new tests pass | 96 OK, 1 skipped (live LLM) with `RUN_MODEL_TESTS=1`: 69 earlier + 10 follow-up + 11 display/cache + 8 hybrid (some earlier tests updated) | PASS | M4 |
| App flows (Streamlit `AppTest` driving `app.py`, live Gemini) | Example prompt, chat, follow-ups, comparison, fallback, clear chat | Example button answers; "which is the best?" compares power banks after a power-bank question and routers after a router question; "battery capacity of the zebie?" answers 12000 mAh with one card; comparison shows both named routers; "capital of France" shows the fallback banner; Clear chat empties history; no exceptions | PASS | M4 |
| Product cards | Cited products only, numbered like the answer, image links to product page, historical price, rating | Verified in AppTest output and in Safari; 25/25 sampled image URLs load; product URLs return HTTP 200 | PASS | M4 |
| Answer cache | Repeat answers instantly across sessions; follow-ups keyed on conversation | Second session: same question and follow-up answered in 0.0 s marked "Answered from cache"; unit tests for keys | PASS | M4 |
| Latency, `gemini-3.1-flash-lite` | A few seconds per answer | 8 questions: median 2.2 s, max 2.5 s. A stall in an earlier run took 16.9 s with the 15 s retry (was 45-90 s) | PASS | M4 |
| Hybrid vs semantic ([evidence](../hybrid_evaluation.md)) | Rule fixed before running: better named lookup, at most 1 category hit lost, under 50 ms extra, budgets hold, off-topic scores not higher | Named product rank 1: 22/30 → 25/30; top 5: 25/30 → 28/30; MRR 0.783 → 0.869. Category queries 12/14 both unfiltered (55 → 54 hits), 14/14 both with app filters. Budgets hold. Off-topic scores identical. Median latency +1-2 ms | PASS (adopted) | M4 |
| Presentation laptop launch | App starts and answers there | Run on M4's MacBook only | NOT RUN on a different machine | M4 / M5 |
| CI on Python 3.12 | Green | [Run 43624192175](https://github.com/bharat0722/flipkart-ai-advisor/actions/runs/43624192175) on PR #9 at `d1cc614`: passed in 1m36s | PASS | M4 |

## Issues and transition

- **Off-topic questions mid-conversation** name no category, so they are retrieved with the earlier topic. The LLM still declines ("capital of France" returns `no_support`), but its reply may mention the earlier products. Owner M4; acceptable for the demo.
- **Gate behaviour unchanged:** "best smartphone camera" and "Sony PlayStation 5 games" pass the 0.20 evidence score in both modes and rely on prompt rule 3, as in Phase 3.
- **Hybrid lost one category hit unfiltered:** "laptop for students" returns laptop chargers because BM25 matches "laptop". The app's category filter removes them (2/2 laptops in the live run).
- **Ratings:** only 91 of 288 products have one; cards show "No rating". Prices and product links are 2015-16; listings may have changed or be discontinued.
- **The cache is in memory** and clears on restart. Warm it before a demo by asking the demo questions once.
- **Phase 3 status:** M4's review (branch `review/phase-03-m4`) accepts with tracked issues once Phase 2 is accepted and CI passes. Phase 4 builds on that pipeline.
- Receiving phase (M5): launch the app from the README on the presentation laptop, use the core flows, inspect sources, clear history and verify an honest fallback. Use accessory and networking questions (the catalogue has 2 laptops and no phones). `docs/hybrid_evaluation.md` and `scripts/retrieval_report.py` are starting points for the Phase 5 evaluation.

## Acceptance

- [x] Completed scope meets phase acceptance criteria.
- [ ] Receiving lead independently reproduced the result at the recorded commit.
- [ ] Reviewer recorded evidence, decision and any outstanding items.
- [ ] The reviewed work was merged and the accepted snapshot tagged.

Decision: **Pending** M5 review.

| Role | Name | Date | Commit | Decision or evidence |
| --- | --- | --- | --- | --- |
| Outgoing lead (M4) | M4 | 2026-11-14 | `c561ddf` | Ready for review. Evidence above. |
| Receiving lead (M5) | Pending | | | Pending |
| QA reviewer | Pending | | | Pending |
