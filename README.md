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
- [ ] **Stage 2** — Embedding + vector storage (both chunk sets indexed separately).
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

## Repo structure

```
src/naive-rag/       # chunking.py, embed.py, store.py, retrieve.py, generate.py
notebooks/         # piloting/verification before code enters src/
data/corpus/       # the 6 target source files
data/chuks/             # chunk_1a.json, chunk_1b.json (comparison artifacts)
evals/             # questions.json + results/
```