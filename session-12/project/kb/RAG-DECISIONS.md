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
| S11 | follow-ups retrieve garbage: "and for the watch?" is searched as written | conversational query rewriting (`kb/query.py`, the `rewrite` node) | +1 call on turns with history; none on a first message | *fill in from session 11, section 17* | `REWRITE_FOLLOWUPS=true` |
| S12 | "has this happened to anyone else?" has no answer in the policies | a second source: past tickets, routed by two tools' docstrings (`workflows/specialists/knowledge.py`) | the knowledge agent's own calls (2–3 per question); 27 documents to embed, once | *fill in from session 12, section 13* | shipped: the knowledge specialist's second tool |
| S12 | metadata exists and nothing uses it: "what does the regulator say" ranks the shop's clause first | self-query filters, with a no-filter fallback (`kb/query.py`) | +1 structured call per knowledge question | *fill in from session 12, section 13* | `SELF_QUERY=true`, pending the measurement |
| S12 | a chunk lost its context: it never says which policy it is from | contextual chunks (`policies-contextual__…`) | +1 call per chunk at INGEST (~70, once, cached in `kb/contexts.json`); none per question | *fill in from session 12, section 14* | `KB_SCHEME=plain` until measured |
| S12 | the matched chunk is too small to answer from | parent document, small-to-big (`policies-parent__…`) | none per question; more tokens per answer (whole sections) | *fill in from session 12, section 14* | `KB_SCHEME=plain` until measured |
| S12 | the same policy question asked again and again | [OPT] node cache on `research` (`KB_CACHE_TTL`) | none; saves every repeat | latency only, see session 12, section 15 | `KB_CACHE_TTL=0` (off) |

S12's rows are measured with `uv run python evals/run_eval.py --rag-eval` (sources and self-query,
on `evals/rag_eval.yml`) and `--schemes` (contextual and parent, on the gold set plus the filtered
questions). The S11 row is measured with
`uv run python evals/run_eval.py --followups`: six two-turn conversations in
`evals/conversational.yml`, plus the gold set's answerable questions asked after an unrelated
opening, which the rewrite must not damage.

## How to read the table

- **Cost** is the extra model calls compared with session 9's straight line (retrieve → generate:
  one call). The scorecard's `calls/q` column is the measured average.
- **Moved the number** means refusal, citation or retrieval rose on the gold set *and* nothing
  else fell. One question rescued out of nineteen is a signal worth writing down. It is not by
  itself a reason to pay a call on every question.
- **Shipped** is the default in `config.py`. Every switch can be turned on for one run with its
  `SHOPWISE_` variable.
