from __future__ import annotations
from dataclasses import dataclass

import numpy as np

@dataclass 
class RetrievalResult:
    chunk_id: str 
    score: float 

def top_k(query_vector: np.ndarray, embeddings: np.ndarray, chunk_ids: list[str], k: int = 5) -> list[RetrievalResult]:
    scores = embeddings @ query_vector
    top_indices = np.argsort(scores)[::-1][:k]
    return [RetrievalResult(chunk_id=chunk_ids[i], score=float(scores[i])) for i in top_indices]