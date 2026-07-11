"""Stage 1b - AST-aware chunking for Naive RAG on Code.

Chunks 6 requests-library source files at class/method/function/module granularity using Python's ast module. No size caps - a function is one chunk regardless of length (token-limit truncation risk pushed downstream to the embedding stage, documented, not solved here)
"""

from __future__ import annotations

import ast 
import json 
from dataclasses import dataclass, asdict
from enum import Enum 
from pathlib import Path 

# Config

CORPUS_FILES = ["api.py", "models.py", "sessions.py", "exceptions.py", "utils.py", "adapters.py"]
REPO_ROOT = Path(__file__).resolve().parents[2] 
CORPUS_DIR = REPO_ROOT / "data" / "corpus" 
OUTPUT_PATH = REPO_ROOT / "data" / "chunks_1b.json"

# Types 

class ChunkType(str, Enum):
    MODULE = "module"
    CLASS = "class"
    METHOD = "method"
    FUNCTION = "function"

@dataclass
class Chunk:
    chunk_id: str
    source_file: str
    chunk_type: ChunkType 
    parent_class: str | None 
    docstring: str | None 
    start_line: int 
    end_line: int 
    text: str 

    def to_dict(self) -> dict: 
        d = asdict(self)
        d["chunk_type"] = self.chunk_type.value 
        return d 

FuncNode = ast.FunctionDef | ast.AsyncFunctionDef

# ast helpers 

def get_start_line(node: ast.AST) -> int:
    """Decorator-aware start line - grabs the topmost decorator if present.
    since node.lineno alone points at the 'def'/'class' line and silently
    drops deocrators from the chunk boundry."""

    decorators = getattr(node, "decorator_list", None)
    if decorators: 
        return decorators[0].lineno 
    return node.lineno # type: ignore[attr-defined]

def get_text(start_line: int, end_line: int, source_lines: list[str]) -> str:
    """Slice source lines for a 1-indexed inclusive line range
    source_lines is passed explicitly - never a module-level global to avoid stale-state
    bugs when looping across files."""
    return "\n.".join(source_lines[start_line - 1 : end_line])

def has_overload_decorator(node: FuncNode) -> bool:
    """True if node is deocrated with @overload or @typing.overload"""
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name) and decorator.id == "overload":
            return True 
        if isinstance(decorator, ast.Attribute) and decorator.attr == "overload":
            return True 
    return False

def filter_overload_stubs(nodes: list[FuncNode]) -> list[FuncNode]:
    """Given functions/methods sharing a scope, drop @overload typing stubs, 
    keep real implementations. If a name has only stubs(no concrete impl found), keep the last
    stub rather than silently dropping the name."""
    by_name: dict[str, list[FuncNode]] = {}
    for node in nodes:
        by_name.setdefault(node.name, []).append(node)

    result: list[FuncNode] = []
    for name, group in by_name.items():
        real_imps = [n for n in group if not has_overload_decorator(n)]
        if real_imps:
            result.extend(real_imps)
        else:
            result.append(group[-1]) # stubs only - keep last as fallback
    return result

# chunk builders

def build_function_chunk(node: FuncNode, file_stem: str, source_lines: list[str]) -> Chunk:
    start = get_start_line(node)
    end = node.end_lineno
    assert end is not None 
    return Chunk(
        chunk_id=f"{file_stem}_{node.name}",
        source_file=file_stem,
        chunk_type=ChunkType.FUNCTION,
        parent_class=None,
        docstring=ast.get_docstring(node),
        start_line=start, 
        end_line=end,
        text=get_text(start, end, source_lines)
    )

def build_method_chunk(node: FuncNode, file_stem: str, parent_class: str, source_lines: list[str]) -> Chunk:
    start = get_start_line(node)
    end = node.end_lineno
    assert end is not None 
    return Chunk(
        chunk_id=f"{file_stem}_{parent_class}_{node.name}",
        source_file=file_stem,
        chunk_type=ChunkType.METHOD,
        parent_class=parent_class,
        docstring=ast.get_docstring(node),
        start_line=start,
        end_line=end,
        text=get_text(start, end, source_lines)
    )

def build_class_chunk(
    node: ast.ClassDef, file_stem: str, source_lines: list[str]
) -> Chunk:
    """Class-level chunk = signature line + every non-method body node's
    OWN text, reconstructed piece by piece — NOT one min-to-max line slice.
    A line-range slice would swallow any method sandwiched between
    non-method statements (e.g. attrs defined after a method), silently
    duplicating that method's source into this chunk too."""
    start = get_start_line(node)
    non_method_nodes = [
        n for n in node.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    signature_line = source_lines[start - 1]
    body_pieces = [
        get_text(n.lineno, n.end_lineno, source_lines)  # type: ignore[arg-type]
        for n in non_method_nodes
    ]
    text = "\n".join([signature_line, *body_pieces])
    
    # Ensure end is always int
    if non_method_nodes:
        end = non_method_nodes[-1].end_lineno or start  # Fallback to start if None
    else:
        end = start

    return Chunk(
        chunk_id=f"{file_stem}_{node.name}",
        source_file=file_stem,
        chunk_type=ChunkType.CLASS,
        parent_class=None,
        docstring=ast.get_docstring(node),
        start_line=start,
        end_line=end,
        text=text,
    )


def build_module_chunk(tree: ast.Module, file_stem: str, source_lines: list[str]) -> Chunk | None:
    """One chunk for top-level statements that aren't class/function defs
    (imports, constants, module docstring). Same node-by-node reconstruction
    as build_class_chunk, for the same reason.

    The module docstring is deliberately EXCLUDED from this sweep — it's
    already captured via ast.get_docstring(tree) into the docstring field.
    Including it in both would duplicate that content into `text` too."""
    module_docstring_node = None
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        module_docstring_node = tree.body[0]

    non_code_nodes = [
        n
        for n in tree.body
        if not isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and n is not module_docstring_node
    ]

    if not non_code_nodes:
        return None

    start = get_start_line(non_code_nodes[0])
    end = non_code_nodes[-1].end_lineno
    assert end is not None
    body_pieces = [
        get_text(get_start_line(n), n.end_lineno, source_lines)  # type: ignore[arg-type]
        for n in non_code_nodes
    ]

    return Chunk(
        chunk_id=f"{file_stem}_module",
        source_file=file_stem,
        chunk_type=ChunkType.MODULE,
        parent_class=None,
        docstring=ast.get_docstring(tree),
        start_line=start,
        end_line=end,
        text="\n".join(body_pieces),
    )

# per file orchestration

def chunk_file(file_path: Path) -> list[Chunk]:
    """Parse one source file and return all chunks (module/class/method/function)."""
    source = file_path.read_text()
    source_lines = source.splitlines()
    tree = ast.parse(source)
    file_stem = file_path.stem

    chunks: list[Chunk] = []

    top_level_funcs: list[FuncNode] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            chunks.append(build_class_chunk(node, file_stem, source_lines))

            class_methods = [
                n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            for method_node in filter_overload_stubs(class_methods):
                chunks.append(
                    build_method_chunk(method_node, file_stem, node.name, source_lines)
                )

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            top_level_funcs.append(node)

    for func_node in filter_overload_stubs(top_level_funcs):
        chunks.append(build_function_chunk(func_node, file_stem, source_lines))

    module_chunk = build_module_chunk(tree, file_stem, source_lines)
    if module_chunk is not None:
        chunks.append(module_chunk)

    return chunks

def chunk_corpus(file_names: list[str], corpus_dir: Path) -> list[Chunk]:
    """Chunk every named corpus file, return one flat list across all files."""
    all_chunks: list[Chunk] = []
    for name in file_names:
        file_path = corpus_dir / name
        all_chunks.extend(chunk_file(file_path))
    return all_chunks

def main() -> None:
    chunks = chunk_corpus(CORPUS_FILES, CORPUS_DIR)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        json.dump([c.to_dict() for c in chunks], f, indent=2)

    print(f"Chunked {len(CORPUS_FILES)} files → {len(chunks)} chunks")
    print(f"Written to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()