# Naive RAG on Code — Phase 1 (Naive RAG, Code Corpus)

Rolling project, Phase 1 of a larger arc: **Naive RAG → Multimodal → Agentic → Agentic variants.**
This phase only: prove/disprove naive (dumb) RAG on code, contrast against AST-aware chunking.

Corpus: 6 files from [`psf/requests`](https://github.com/psf/requests/tree/main/src/requests) core —
`api.py`, `models.py`, `sessions.py`, `exceptions.py`, `utils.py`, `adapters.py`.

MVP target (this phase): full pipeline (ingest → chunk → embed → store → retrieve → generate)
+ minimal scripted eval loop. No frameworks (no LangChain/LlamaIndex) — raw Python throughout.
All pipeline code hand-written; AI used only for trade-off analysis and debugging, not code generation
past the initial piloting/notebook phase.

## Status

- [x] Repo structure initialized
- [x] **Stage 0** — Eval question set (lookup-tier). 12 hand-verified questions across all 6 files,
      saved to `evals/questions.json`. Ground truth confirmed by reading source directly, not
      LLM-generated answers.
- [x] **Stage 1a** — Dumb baseline chunking (line-based, no code-structure awareness).
      40-line windows, 5-line overlap, per-file chunk IDs. Output: `data/chunks/chunks_1a.json`.
- [X] **Stage 1b** — AST-aware chunking (function/class-level, Python `ast` module).
- [X] **Stage 2** — Embedding + vector storage (both chunk sets indexed separately).
- [ ] **Stage 3** — Retrieval (Stage 0 queries run against both indices).
- [ ] **Stage 4** — Thin/naive generation layer.
- [ ] **Stage 5** — Scripted eval pass, manual scoring, comparison writeup.
- [ ] **Stage 6** — Retro: what naive RAG on code cannot do.

## Stage 1a notes

Sliding window: `stride = chunk_size - overlap`. Chunking stops once a chunk's `end_line` reaches
EOF, preventing a trailing duplicate/subset chunk. Metadata captured per chunk: `chunk_id`,
`source_file`, `start_line`, `end_line`, `text`.

Bugs hit and fixed during piloting (full postmortems in Notion):
1. Hardcoded chunk-ID prefix — worked on single-file test, silently wrong once tested against a
   second file.
2. Missing EOF guard — last chunk in a file could produce a redundant trailing chunk that's a pure
   subset of the previous one.
3. `chunk_id` carried the `.py` extension (`exceptions.py_0`) — fixed by using `Path.stem` instead
   of the raw filename.

## Stage 1b notes

AST-based structural chunking via Python's `ast` module — parses each file into a syntax tree
instead of counting lines. Granularity: class-level chunk (signature + docstring + non-method
body only) + one chunk per method (children of class, no duplication) + one chunk per top-level
function + one module-level chunk per file (imports, constants, orphan top-level statements).
No size cap on function/method chunks — a function is one chunk regardless of length; token-limit
truncation risk at the embedding stage is a known, documented tradeoff, not solved here.

Metadata per chunk: `chunk_id`, `source_file`, `chunk_type` (module/class/method/function),
`parent_class`, `docstring`, `start_line`, `end_line`, `text`. `start_line` is decorator-aware
(grabs the topmost decorator, not the `def`/`class` line). Output: `data/chunks/chunks_1b.json`.

Bugs hit and fixed during piloting (full postmortems in Notion):
1. Class chunk `end_line` assumed non-method body nodes were contiguous — a method sandwiched
   between attribute definitions got silently duplicated into the class chunk. Fixed by
   reconstructing text node-by-node instead of one min-to-max line slice.
2. Implicit global `source_lines` caused silently empty `text` fields when switching source files
   mid-session — out-of-range list slicing doesn't error in Python, it just returns empty. Fixed
   by passing `source_lines` explicitly everywhere.
3. Typo in an attribute-name check (`deocrator_list`) silently disabled `@overload` detection
   entirely — passed testing on files with no naturally colliding names, only surfaced against a
   file with real `@overload` collisions (`models.py`).
4. `@overload` typing-stub declarations share a function name across multiple AST nodes, breaking
   the `chunk_id` uniqueness assumption. Resolved by filtering stub bodies (empty `...`, zero
   retrieval value) and keeping only the real implementation.
5. Module docstring double-counted — captured both via `ast.get_docstring()` into the `docstring`
   field and swept into the module chunk's reconstructed `text`. Fixed by excluding that node from
   the module-level sweep.

## Stage 2 notes

Embedding via `BAAI/bge-small-en-v1.5` (local, `sentence-transformers`), chosen over API-based
embedding for iteration speed and full inspectability of the embedding step. No vector DB —
corpus is ~200-400 chunks total across both sets, well under any scale that would justify one.
Storage is raw numpy: normalized embedding vectors saved as `.npy`, with a parallel `chunk_id`
JSON list preserving row-alignment. Cosine similarity reduces to a dot product on normalized
vectors; retrieval is a single matrix-vector multiply + `argsort`, no framework or DB layer
between query and result.

Both chunk sets (`chunks_1a.json`, `chunks_1b.json`) canonicalized to a flat list structure at
this stage — 1a was previously a dict grouped by source file; 1b was already flat. Uniform shape
going forward removes branching logic from all downstream code.

Every chunk in both sets carries an explicit `truncated: bool` field (BGE's 512-token limit,
checked via the model's own tokenizer) — 3 chunks flagged in 1a, 19 in 1b. Oversized chunks are
not split or fixed this stage; the tokenizer would otherwise truncate them silently on encode,
so the flag exists to make that truncation visible and queryable at eval/debug time rather than
leaving it invisible. Whether these specific chunks cause retrieval misses is a Stage 3/5 question,
not assumed here.

Query-side embedding uses BGE's documented asymmetric prefix convention
(`"Represent this sentence for searching relevant passages: "`). Empirical A/B testing on 3 eval
queries showed a smaller effect than expected — top-1 result was stable with/without the prefix,
only minor rank-2/3 reshuffling. Prefix retained on trained-convention grounds; full effect (if
any) to be re-evaluated against the complete Stage 0 eval set in Stage 3/5.

Bugs hit and fixed during this stage (full postmortem in Notion):
1. Truncation-marking script read from and wrote to the same source file across two inconsistent
   versions of the marking logic, producing duplicate nested + flat `truncated` keys on every
   flagged chunk. Fixed by restoring pristine chunk files and making the marking operation a full,
   unconditional overwrite every run — idempotent, safe to re-run in place.

## Repo structure

```
src/naive-rag/       # ast_chunking.py, navie_chunking.py, embed.py, store.py, retrieve.py, generate.py
notebooks/         # piloting/verification before code enters src/
data/corpus/       # the 6 target source files
data/chuks/             # chunk_1a.json, chunk_1b.json (comparison artifacts)
evals/             # questions.json + results/
```