"""
src/naive-rag/chunking.py

Stage 1a - Dumb baseline chunking (line-based, no code awareness).
"""

from __future__ import annotations
from pathlib import Path 
from typing import TypedDict, Iterable, List, Any
import json 
import os

class Chunk(TypedDict):
    chunk_id: str
    source_file: str
    start_line: int 
    end_line: int
    text: str

def chunk_file(file_path: Path, chunk_size: int = 40, overlap: int = 5) -> List[Chunk]:
    """Split files into overlapping line based chunks.
        
        Sliding windows: 
        - window size = chunk_size
        - stride = chunk_size - overlap
        - stops once a chunk reaches EOF (no trailing duplicate/subsest chunk)
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be smaller than chunk_size")
    
    stride = chunk_size - overlap 
    file_path = Path(file_path)
    file_stem = file_path.stem
    
    with file_path.open("r", encoding="utf-8") as f:
        lines = f.read().splitlines()
        
    n = len(lines)
    chunks: list[Chunk] = []
    i = 0
    chunk_idx = 0

    while i < n: 
        window = lines[i: i + chunk_size]
        end_line = min(i + chunk_size, n) -1

        chunks.append(Chunk(
            chunk_id=f"{file_stem}_{chunk_idx}",
            source_file=file_path.name,
            start_line=i,
            end_line=end_line,
            text="\n".join(window)
            )
        )
        # EOF guard for last line of the chunk
        if end_line == n - 1:
                break 

        i += stride 
        chunk_idx += 1
    
    return chunks

def chunk_corpus(corpus_dir: Path, filenames: list[str], chunk_size: int = 40, overlap: int = 5) -> dict[str, list[Chunk]]:
    """
     Chunk a fixed, explicit list of files.
    """
    corpus_dir = Path(corpus_dir)
    results: dict[str,list[Chunk]] = {}

    for name in filenames:
        file_path = corpus_dir / name 
        if not file_path.exists():
            raise FileNotFoundError(f"Expected corpus file missing: {file_path}")
        results[name] = chunk_file(file_path, chunk_size=chunk_size, overlap=overlap)

    return results

def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    corpus_dir = project_root / "data" / "corpus"
    target_files = ["api.py", "models.py", "sessions.py", "exceptions.py", "utils.py", "adapters.py"]

    all_chunks = chunk_corpus(corpus_dir, target_files)

    for filename, chunks in all_chunks.items():
        print(f"{filename}: {len(chunks)} chunks")
    
    out_path = project_root / "data" / "chunks" / "chunks_1a.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent = 2)
    
    print(f"\nSaved {sum(len(c) for c in all_chunks.values())} total chunks to {out_path}")


if __name__ == "__main__":
    main()