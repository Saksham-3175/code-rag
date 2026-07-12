from __future__ import annotations 
from pathlib import Path 

from sentence_transformers import SentenceTransformer 

from embed import load_chunks, save_chunks, mark_truncated, embed_corpus, embed_query
from store import save_embeddings, load_embeddings
from retrieve import top_k 


REPO_ROOT = Path(__file__).resolve().parents[2] 
CHUNK_DIR = REPO_ROOT / "data" / "chunks"
EMBED_DIR = REPO_ROOT / "data" / "embeddings"
CHUNK_SETS = ["chunks_1a", "chunks_1b"]

def build_index(model: SentenceTransformer) -> None:
    for name in CHUNK_SETS:
        path = CHUNK_DIR / f"{name}.json"
        chunks = load_chunks(path)

        flagged = mark_truncated(chunks, model)
        save_chunks(chunks, path)
        print(f"{name}: {flagged} chunks marked truncated")

        embeddings = embed_corpus(chunks, model)
        chunk_ids = [c["chunk_id"] for c in chunks]
        save_embeddings(embeddings, chunk_ids, name.replace("chunks_", ""), EMBED_DIR)
        print(f"{name}: embedded {embeddings.shape[0]} chunks -> {EMBED_DIR}")

def demo_query(model: SentenceTransformer, query: str, set_name: str = "1b", k: int = 5) -> None:
    embeddings, chunk_ids = load_embeddings(set_name, EMBED_DIR)
    query_vec = embed_query(query, model)
    for r in top_k(query_vec, embeddings, chunk_ids, k=k):
        print(f"{r.chunk_id:30s} {r.score:.4f}")

def main() -> None:
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    build_index(model)
    demo_query(model, "How does requests handle connection retries?")


if __name__ == "__main__":
    main()  