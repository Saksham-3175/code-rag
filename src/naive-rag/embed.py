from __future__ import annotations
from pathlib import Path 
from pathlib import Path 
from typing import Union
import json 

import numpy as np 
from sentence_transformers import SentenceTransformer

TOKEN_LIMIT = 512 
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

def load_chunks(path: Path) -> list[dict]:
    """Normalize dict-of-lists (1a) or flat list (1b) into a flat list."""
    with open(path, "r", encoding="utf-8") as f:
        content: Union[dict, list] = json.load(f)
    if isinstance(content, dict):
        return [chunk for chunk_list in content.values() for chunk in chunk_list]
    return content 

def save_chunks(chunks: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

def mark_truncated(chunks: list[dict], model: SentenceTransformer, limit: int = TOKEN_LIMIT) -> int:
    """Sets an explicit True/False on every chunk, never absent.
    Idempotent - safe to re-run on the same file, always a full overwrite."""
    tokenizer = model.tokenizer
    flagged = 0
    for chunk in chunks:
        token_len = len(tokenizer.encode(chunk["text"], add_special_tokens=False))
        is_over = token_len > limit 
        chunk["truncated"] = is_over 
        flagged += is_over
    return flagged

def embed_corpus(chunks: list[dict], model: SentenceTransformer) -> np.ndarray:
    texts = [chunk["text"] for chunk in chunks]
    return model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)

def embed_query(query: str, model: SentenceTransformer, use_prefix: bool = True) -> np.ndarray:
    text = QUERY_PREFIX + query if use_prefix else query
    return model.encode([text], normalize_embeddings=True)[0]