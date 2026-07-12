from __future__ import annotations
from pathlib import Path 
import json 

import numpy as np

def save_embeddings(embeddings: np.ndarray, chunk_ids: list[str], name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{name}_embeddings.npy", embeddings)
    with open(out_dir / f"{name}_ids.json", "w", encoding="utf-8") as f:
        json.dump(chunk_ids, f, indent=2)

def load_embeddings(name: str, in_dir: Path) -> tuple[np.ndarray, list[str]]: 
    embeddings = np.load(in_dir / f"{name}_embeddings.npy")
    with open(in_dir / f"{name}_ids.json", "r", encoding="utf-8") as f:
        chunk_ids: list[str] = json.load(f)
    return embeddings, chunk_ids
