"""
rag_indexer.py — Build a local RAG index from workspace files.

Usage:
    python rag_indexer.py            # index all text files in workspace
    python rag_indexer.py --glob "*.md"   # index only markdown files
    python rag_indexer.py --path docs/    # index a subdirectory

Output:
    rag_index.json  — list of {file, chunk_id, text, vector}

The index is read by agent_v4.py at startup and queried via rag_search().
Re-run this script whenever workspace files change significantly.
"""

import argparse
import json
import pathlib
import sys
import textwrap

import requests

# ─────────────────────────── Config ──────────────────────────────────────────

OLLAMA_URL   = "http://localhost:11434/api/embed"
EMBED_MODEL  = "nomic-embed-text"
WORK_DIR     = pathlib.Path(__file__).parent.resolve()
INDEX_FILE   = WORK_DIR / "rag_index.json"

# Files to skip — binary, generated, or not useful for retrieval
SKIP_SUFFIXES = {".json", ".pyc", ".png", ".jpg", ".jpeg", ".gif", ".pdf"}
SKIP_NAMES    = {"rag_index.json", "memory.json"}

CHUNK_LINES   = 30    # lines per chunk — tune this for your content
CHUNK_OVERLAP = 5     # lines of overlap between consecutive chunks

# ─────────────────────────── Chunking ────────────────────────────────────────

def chunk_text(text: str, chunk_lines: int, overlap: int) -> list[str]:
    """
    Split text into overlapping line-based chunks.

    Why overlap? A fact that straddles a chunk boundary would otherwise be
    split across two chunks and retrieved poorly. Overlap ensures it appears
    fully in at least one chunk.

    Example with chunk_lines=4, overlap=1:
      lines 0-3  → chunk 0
      lines 3-6  → chunk 1  (line 3 repeated)
      lines 6-9  → chunk 2
    """
    lines  = text.splitlines()
    chunks = []
    step   = chunk_lines - overlap
    for i in range(0, max(1, len(lines)), step):
        chunk = "\n".join(lines[i : i + chunk_lines]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks

# ─────────────────────────── Embedding ───────────────────────────────────────

def embed(text: str) -> list[float]:
    """
    Call nomic-embed-text via Ollama and return the embedding vector.

    Ollama /api/embed accepts a single string or a list.
    Returns a flat list of floats (768 dimensions for nomic-embed-text).
    """
    resp = requests.post(
        OLLAMA_URL,
        json={"model": EMBED_MODEL, "input": text},
        timeout=60,
    )
    resp.raise_for_status()
    # Response: {"embeddings": [[float, ...]]}
    return resp.json()["embeddings"][0]

# ─────────────────────────── File Discovery ──────────────────────────────────

def find_files(base: pathlib.Path, glob: str) -> list[pathlib.Path]:
    files = []
    for f in base.rglob(glob):
        if not f.is_file():
            continue
        if f.suffix.lower() in SKIP_SUFFIXES:
            continue
        if f.name in SKIP_NAMES:
            continue
        if any(part.startswith(".") for part in f.parts):
            continue   # skip hidden dirs like .git
        files.append(f)
    return sorted(files)

# ─────────────────────────── Indexer ─────────────────────────────────────────

def build_index(base: pathlib.Path, glob: str) -> list[dict]:
    """
    Chunk every matching file, embed each chunk, return index entries.

    Each entry:
      {
        "file":     "relative/path.md",
        "chunk_id": 0,          # chunk number within the file
        "text":     "...",      # raw chunk text (returned to the agent)
        "vector":   [float...]  # 768-dim embedding
      }
    """
    files = find_files(base, glob)
    if not files:
        print(f"No files found matching '{glob}' in {base}")
        return []

    print(f"Found {len(files)} file(s) to index.")
    index = []

    for file in files:
        rel = file.relative_to(WORK_DIR)
        try:
            text = file.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            print(f"  SKIP {rel}: {e}")
            continue

        chunks = chunk_text(text, CHUNK_LINES, CHUNK_OVERLAP)
        print(f"  {rel}  ({len(chunks)} chunk(s))", end="", flush=True)

        for i, chunk in enumerate(chunks):
            vector = embed(chunk)
            index.append({
                "file":     str(rel),
                "chunk_id": i,
                "text":     chunk,
                "vector":   vector,
            })
            print(".", end="", flush=True)   # progress dot per chunk

        print()  # newline after each file

    return index

# ─────────────────────────── Main ────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build RAG index for the workspace.")
    parser.add_argument("--glob", default="*.*", help="File glob pattern (default: *.*)")
    parser.add_argument("--path", default=".", help="Subdirectory to index (default: workspace root)")
    args = parser.parse_args()

    base = (WORK_DIR / args.path).resolve()
    if not base.is_relative_to(WORK_DIR):
        print("Error: --path must be inside the workspace.")
        sys.exit(1)

    print(f"Embedding model : {EMBED_MODEL}")
    print(f"Workspace       : {WORK_DIR}")
    print(f"Indexing        : {base} (glob: {args.glob})")
    print(f"Chunk size      : {CHUNK_LINES} lines  overlap: {CHUNK_OVERLAP} lines")
    print()

    # Check Ollama is reachable
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5).raise_for_status()
    except Exception:
        print("Error: cannot reach Ollama. Run: ollama serve")
        sys.exit(1)

    index = build_index(base, args.glob)

    if not index:
        print("Nothing indexed.")
        sys.exit(0)

    INDEX_FILE.write_text(
        json.dumps(index, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    total_chunks = len(index)
    total_files  = len({e["file"] for e in index})
    dim          = len(index[0]["vector"])
    print()
    print(f"Index saved → {INDEX_FILE.name}")
    print(f"  {total_files} file(s)  {total_chunks} chunk(s)  {dim}-dim vectors")
    print()
    print("Run agent_v4.py — the rag_search tool will use this index.")


if __name__ == "__main__":
    main()
