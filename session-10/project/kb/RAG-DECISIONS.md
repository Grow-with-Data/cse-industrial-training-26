# RAG decisions — what each retrieval technique bought ShopWise

Every technique here costs model calls on every policy question. A row records a symptom we
actually saw, the technique tried against it, what it cost, and whether it moved **our** number on
**our** corpus. Not a blog's corpus, and not a benchmark's. A technique that moves nothing ships
built and **off**, and its row says so.

Measured with `uv run python evals/run_eval.py --arms` (19 gold questions in
`labs/data/kb_questions.yml`). Re-run that before changing a default in `config.py`, and add a row.

| Session | Symptom | Technique | Cost per policy question | Moved the number? | Shipped |
|---|---|---|---|---|---|
| S09 | the exception clause sometimes missed the top k | hybrid search (dense + BM25) | none (one search) | see session 9, section 14 | `RETRIEVAL_MODE` as measured |
| S09 | right chunk retrieved, ranked below the cut | cross-encoder rerank | ~20 chunk scorings on CPU | see session 9, section 14 | `RERANK` as measured |
| S10 | vocabulary mismatch: "money back for a gift" vs "gift cards" | RAG-Fusion (3 rephrasings + RRF) | +1 call, +3 searches | *fill in from session 10, section 15* | `RAG_FUSION=false` until measured |
| S10 | the same, approached differently | HyDE (search with a fake answer) | +1 call | *fill in from session 10, section 15* | `RAG_HYDE=false` until measured |
| S10 | weak passages go straight to generation | corrective loop (grade → rewrite → retry, max 2) | +1 call, up to +3 | *fill in from session 10, section 15* | `RAG_GRADE=false` until measured |

S11 and S12 add rows (query rewriting from conversation history; routing between sources).

## How to read the table

- **Cost** is the extra model calls compared with session 9's straight line (retrieve → generate:
  one call). The scorecard's `calls/q` column is the measured average.
- **Moved the number** means refusal, citation or retrieval rose on the gold set *and* nothing
  else fell. One question rescued out of nineteen is a signal worth writing down. It is not by
  itself a reason to pay a call on every question.
- **Shipped** is the default in `config.py`. Every switch can be turned on for one run with its
  `SHOPWISE_` variable.
