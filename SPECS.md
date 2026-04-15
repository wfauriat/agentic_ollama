# Project Specification — Educational Ollama Agent

> A blueprint for rebuilding a small agentic-AI codebase. Designed to be fed
> to an LLM agent (or a human learner) as a single brief. Captures the full
> feature set in one target — no roadmap, no version split.

---

## 1. Purpose

Build a local, end-to-end agentic-AI loop on top of the Ollama API. The
artefact is a single-file Python REPL that reasons, calls tools, streams
thinking tokens, persists memory, and performs Retrieval-Augmented
Generation over a workspace.

### Educational goals
Someone reading the code — or an agent rebuilding it — should be able to
answer, by reading alone:

- What is the minimal structure of an agentic loop?
- How does native tool calling work with a local LLM?
- How does streaming differ from blocking?
- How does RAG integrate with a tool-using agent?
- Which practical failures (runaway verification loops, path traversal,
  shell injection) are guarded against, and how?

The goal is **understanding**, not shipping a product.

---

## 2. Guiding Principles (non-negotiable)

Apply these to every decision. When in doubt, prefer what a beginner reads
and understands fastest.

1. **Readability first.** Efficiency, performance, and cross-platform
   portability are explicitly *not* priorities. A slower, clearer
   implementation beats a faster opaque one.
2. **Single-file layout.** The agent lives in one Python file. No packages,
   no `src/`, no `__init__.py`. The RAG indexer is the only other script.
3. **Minimal abstraction.** No classes unless unavoidable. No dependency
   injection, plugin architectures, or config frameworks. Module-level
   functions and module-level constants are the default.
4. **Comment the *why*, label the *what*.** A module docstring states what
   the file does. Section banners (`# ── Section ──`) split the file into
   navigable regions. Comments explain reasoning, invariants, and failure
   modes — not what a well-named variable already says.
5. **Errors as strings at the tool boundary.** Tool failures become
   `"ERROR: ..."` strings returned to the model, not raised exceptions.
   The model reads and adapts.
6. **Straight-line code over helpers.** Three similar lines is fine. Do not
   extract a helper until repetition is genuinely painful.
7. **No feature flags or back-compat shims.** Keep one code path per
   concept.
8. **Document what the code teaches.** A `README.md` and a `CHEATSHEET.md`
   accompany the code and explain the concepts pedagogically.

---

## 3. Prerequisites

- Python 3.10+ (uses `match` statements and `str | None` union syntax).
- Ollama running locally at `http://localhost:11434`.
- Models pulled:
  - `qwen3:8b` — reasoning + tool calling
  - `nomic-embed-text` — embeddings for RAG
- Python dependencies: `requests`, `rich`, `numpy`. Exactly those three,
  one per line in `requirements.txt`.

---

## 4. File Layout

```
ollama_agent/
├── agent.py             # main entry point — the REPL
├── rag_indexer.py       # standalone script to build rag_index.json
├── requirements.txt     # requests, rich, numpy
├── README.md            # user-facing tour
├── CHEATSHEET.md        # concept reference
├── SPECS.md             # this file
├── logs/                # per-task JSON logs, auto-created
├── memory.json          # persistent KV facts, auto-created
└── rag_index.json       # chunk + vector index, built by rag_indexer.py
```

---

## 5. Core Architecture

### The agent loop (pseudocode)

```
messages = [system_prompt, user_task]
for step in range(MAX_ITER):
    msg = call_model(messages)          # streaming or blocking
    if msg.tool_calls:
        # intercept `done` before any other tool
        for tc in msg.tool_calls:
            if tc.name == "done":
                show_final_answer(tc.args.summary)
                save_session_log(...)
                return

        messages.append(msg)             # assistant turn BEFORE tool results
        for tc in msg.tool_calls:
            result = execute_tool(tc)
            messages.append({"role": "tool", "content": result})
        # loop again — model sees the results on the next call
    elif msg.content.strip():
        messages.append(msg)
        show_final_answer(msg.content)
        save_session_log(...)
        return
    else:
        # empty response — stop to avoid spinning
        return
```

`messages` is the agent's entire working memory — the model is stateless.

### Message roles

| Role | When added | Key fields |
|---|---|---|
| `system` | Once at start | `content` |
| `user` | Each user task | `content` |
| `assistant` | Model reply | `content`, `thinking`, `tool_calls` |
| `tool` | After executing a tool call | `content` (result string) |

### Tool definition schema

```python
{
    "type": "function",
    "function": {
        "name": "tool_name",
        "description": "Reads this to decide when to use it.",
        "parameters": {
            "type": "object",
            "properties": {"arg": {"type": "string", "description": "..."}},
            "required": ["arg"],
        },
    },
}
```

Descriptions are the tool's contract with the model. Write them carefully;
vague descriptions cause wrong tool choices. Where two tools overlap (e.g.
`search` vs `rag_search`), the descriptions should steer the model
explicitly ("for conceptual queries, prefer rag_search").

---

## 6. Ollama Integration

Two endpoints are used:

- `POST http://localhost:11434/api/chat` — chat + tool calls
- `POST http://localhost:11434/api/embed` — embeddings for RAG

### Blocking call

```python
payload = {
    "model":    MODEL,
    "stream":   False,
    "think":    THINK,
    "tools":    TOOLS,
    "messages": messages,
}
resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
data = resp.json()
# data["message"]: {"role", "content", "thinking", "tool_calls"}
# data["prompt_eval_count"], data["eval_count"]  — token counts
```

### Streaming call

`"stream": True` returns newline-delimited JSON. One chunk per line.

- `chunk["message"]["thinking"]` — reasoning token(s), stream live (dim).
- `chunk["message"]["content"]` — answer token(s), accumulate and render
  in a panel at the end.
- `chunk["message"]["tool_calls"]` — can appear in *any* chunk, not only
  the final one. Accumulate across the whole stream.
- `chunk["done"] == True` — final chunk, read `prompt_eval_count` and
  `eval_count` for token display.
- Beware: `[]` is falsy. Use `len(tool_calls) > 0`, not `if tool_calls`.

Both paths must coexist and be toggleable at runtime via the `stream` REPL
command.

### Embeddings

```python
resp = requests.post(EMBED_URL,
                     json={"model": "nomic-embed-text", "input": text},
                     timeout=60)
vector = resp.json()["embeddings"][0]   # list[float], 768 dims
```

---

## 7. Tool Set

All tools operate inside `WORK_DIR` (the directory containing `agent.py`).
Every filesystem tool goes through `safe_path()`.

| Name | Purpose | Required args |
|---|---|---|
| `ls` | List directory entries, dirs first then files alphabetically | — (path optional) |
| `cat` | Read a file as UTF-8. Description steers toward `rag_search` for large files or open-ended questions | `path` |
| `write_file` | Write text, creating parent dirs as needed | `path`, `content` |
| `mkdir` | Create directory (and parents) | `path` |
| `mv` | Move/rename within workspace | `src`, `dest` |
| `cp` | Copy file or directory (`copytree` for dirs) | `src`, `dest` |
| `rm` | Delete file or empty directory (no recursive delete) | `path` |
| `search` | Case-insensitive substring scan across files; returns `path:line:content` for up to ~100 matches | `pattern` (path and glob optional) |
| `run_python` | Run a Python snippet via `subprocess.run([sys.executable, "-c", code], ...)` with a 15 s timeout; return stdout + stderr + exit code | `code` |
| `rag_search` | Semantic retrieval over the indexed workspace; embed query, cosine-similarity rank, return top-k chunks as `"[i] file (chunk_id)\n<text>"` joined by separators | `query` (top_k optional) |
| `remember` | Persist a fact to `memory.json` | `key`, `value` |
| `forget` | Drop a key from `memory.json` | `key` |
| `recall` | Return all persisted facts as text | — |
| `done` | Explicit completion signal. The loop intercepts this *before* running other tools, prints the summary as the final answer, and returns | `summary` |

### The `done` pattern — why it exists

Small models often keep calling tools after a task is complete ("let me
verify once more"). Instead of relying on a text-only turn to signal end,
give the model an explicit `done` tool and instruct it in the system prompt:

> "As soon as the task is fully accomplished, call the 'done' tool with a
> short summary — do not do extra verification steps unless specifically
> asked."

The loop checks for `done` *before* dispatching other tool calls from the
same turn, so a batch of `[done, something_else]` still exits cleanly.

---

## 8. Persistent Memory

Flat key-value JSON at `WORK_DIR / "memory.json"`. Loaded on each call to
`build_system_prompt()` so facts are injected into every system prompt:

```
Facts you remember from past sessions:
  - key1: value1
  - key2: value2
```

Tools `remember`, `forget`, `recall` read/write this file.

---

## 9. Retrieval-Augmented Generation (RAG)

### Indexer (`rag_indexer.py`, run separately)

- CLI: `argparse` with `--glob` (default `*.*`) and `--path` (default `.`).
- Skip lists:
  - suffixes: `.json`, `.pyc`, `.png`, `.jpg`, `.jpeg`, `.gif`, `.pdf`
  - names: `rag_index.json`, `memory.json`
  - any path where a part starts with `.` (hidden dirs like `.git`)
- Chunk by lines with overlap. Defaults: `CHUNK_LINES = 30`,
  `CHUNK_OVERLAP = 5`. The docstring explains *why* overlap matters: a fact
  split across chunks would otherwise retrieve poorly.
- Embed every chunk via `POST /api/embed`.
- Output `rag_index.json` as a list of
  `{"file", "chunk_id", "text", "vector"}` entries (vector is a raw list of
  768 floats).
- Progress: one dot per chunk, newline per file. Total summary at the end.
- Fail fast with a readable error if Ollama is unreachable.

### Retrieval (inside the agent)

- Load `rag_index.json` at startup. Convert each vector to
  `numpy.array(..., dtype=np.float32)` once so similarity is fast.
- `cosine_similarity(a, b) = dot(a, b) / (||a|| * ||b||)` with a
  zero-denominator guard.
- `retrieve(query, index, top_k)`: embed the query, score all entries,
  return top-k.
- `rag_search` tool wraps `retrieve`; returns a readable block for the model.
- REPL `index` command shells out to `python rag_indexer.py` and reloads
  the in-memory index on success.

RAG only helps when source material exceeds what fits comfortably in
context. For small workspaces, `cat` + `search` are often enough. The system
prompt and tool descriptions reflect this: the model is told to prefer
`rag_search` for conceptual queries or when the user explicitly asks for
indexed-document search.

---

## 10. Session Logging

After every completed (or aborted) task, write
`logs/YYYY-MM-DD_HH-MM-SS.json` containing `{"task": str, "messages": list}`.
Create `logs/` on demand. These files are the primary debugging artefact —
they let you replay or diff any task after the fact.

---

## 11. Safety

Two idioms are required; anything more belongs to a production extension.

1. **Path-traversal guard on every filesystem operation:**

   ```python
   def safe_path(rel: str) -> pathlib.Path:
       resolved = (WORK_DIR / rel).resolve()
       if not resolved.is_relative_to(WORK_DIR):
           raise ValueError(f"Path traversal blocked: '{rel}'")
       return resolved
   ```

2. **No `shell=True`** when running LLM-provided code:

   ```python
   subprocess.run([sys.executable, "-c", code],
                  capture_output=True, text=True, timeout=15, cwd=WORK_DIR)
   ```

All other tool errors are caught and returned as `"ERROR: <message>"` strings
so the model can react.

---

## 12. REPL — Commands & Toggles

The REPL is a simple `while True: input()` loop. Empty input is ignored.
`quit`, `exit`, `q` exit. `Ctrl-C` / EOF also exit gracefully. `readline` is
imported (side-effect only) for arrow-key history.

| Command | Effect |
|---|---|
| `help` | List all commands |
| `history` | Tasks attempted this session |
| `memory` | Show persisted facts (`memory.json`) |
| `index` | Rebuild RAG index (invokes `rag_indexer.py`) |
| `debug` | Toggle debug dump of `messages` each step |
| `think` | Toggle thinking tokens |
| `stream` | Toggle streaming vs blocking |
| `convo` | Toggle cross-task conversation memory |
| `clear` | Wipe conversation memory |
| `persona` | Multi-line editor for the system prompt; blank line confirms, `reset` restores the base prompt |
| `quit` / `exit` / `q` | Leave |

In **conversation mode** (`convo = True`) the same `messages` list is reused
across tasks so the model remembers prior turns. Default is off — each task
starts fresh.

---

## 13. System Prompt

A single base prompt. Keep it short and specific:

> "You are a helpful file-system agent. You operate exclusively inside a
> sandboxed workspace directory. You have tools to list, read, write, search,
> execute Python, move, copy, and delete files within that workspace. You
> have a rag_search tool to semantically retrieve relevant passages from
> indexed documents — use it when asked to search indexed documents or for
> conceptual queries. You also have tools to remember and forget facts
> across sessions. Complete the user's task step by step. As soon as the
> task is fully accomplished, call the 'done' tool with a short summary —
> do not do extra verification steps unless specifically asked."

When memory is non-empty, append the facts list.

The `persona` command lets the user replace this prompt at runtime.

---

## 14. Display (Rich)

- Use `rich.console.Console`, `rich.panel.Panel`, `rich.syntax.Syntax`.
- Four distinct panel styles:
  - **Thinking** — dim/italic border
  - **Tool call** — yellow border, JSON syntax-highlighted args
  - **Tool result** — green border
  - **Final answer** — blue border
- Step separators: `console.rule(f"Step {n}/{MAX_ITER}")`.
- Streaming thinking tokens print inline using dim ANSI (`\033[2m…\033[0m`),
  prefixed by `── Thinking`. Streaming content is buffered and rendered in a
  panel at the end — never mix raw `print` with Rich panels.
- Token counts are printed per step and cumulatively for the task.

---

## 15. Configuration

Module-level constants at the top of `agent.py`. No config files, no env
vars beyond what Ollama itself reads. REPL toggles mutate these globals at
runtime.

```python
OLLAMA_URL        = "http://localhost:11434/api/chat"
EMBED_URL         = "http://localhost:11434/api/embed"
MODEL             = "qwen3:8b"
EMBED_MODEL       = "nomic-embed-text"
WORK_DIR          = pathlib.Path(__file__).parent.resolve()
LOGS_DIR          = WORK_DIR / "logs"
MEMORY_FILE       = WORK_DIR / "memory.json"
INDEX_FILE        = WORK_DIR / "rag_index.json"
MAX_ITER          = 10
THINK             = True
DEBUG             = False
STREAM            = True
CONVERSATION_MODE = False
RAG_TOP_K         = 5
```

---

## 16. Documentation Deliverables

Produce two reader-facing docs alongside the code:

- **`README.md`** — what it is, prerequisites, quick start, feature list,
  an ASCII diagram of the agent loop, a feature-by-feature walkthrough,
  suggested demo tasks, and a feature-parity comparison to a production
  agent (e.g. Claude Code) to put the project in context.
- **`CHEATSHEET.md`** — concept reference: mental model, Ollama API snippets
  (blocking + streaming), tool schema template, loop pseudocode, message-
  role table, idioms (path guard, subprocess safety, error-as-string), and a
  "possible extensions" roadmap.

Both documents must explain *why* a pattern exists, not just *how* it is
coded.

---

## 17. Acceptance Tasks

The finished REPL must handle these end-to-end (they double as the README's
"suggested tasks"):

```
list the files in the workspace
create a directory called notes
create notes/hello.txt with the content 'hello world'
read README.md and summarise it
create a backup dir and copy a file into it
create a CSV with the capitals of all European countries and their population
```

After `python rag_indexer.py --glob "*.md"`:
```
search the indexed documents for anything about the done tool pattern
search the indexed documents for everything related to streaming
```

All tasks should complete without exceeding `MAX_ITER` and should end via
the `done` tool.

---

## 18. Non-Goals (explicit)

To keep the scope honest, these are **out of scope**:

- Async/await, asyncio, concurrency
- Plugin systems, dynamic tool registration
- Cross-LLM abstraction (no OpenAI fallback, no LangChain wrappers)
- Web UI, HTTP server, IDE integration
- Packaging, `pyproject.toml`, entry points, wheels
- `mypy --strict` — annotations welcome, strictness not required
- Unit tests — the REPL is exercised interactively; formal tests would
  balloon code size without pedagogical payoff
- Context-window auto-summarisation, multi-agent orchestration,
  self-critique, plan/act split — flag these in the README as *possible
  extensions*, do not implement them here.

---

## 19. Instructions to the rebuilding agent

If you are an LLM agent given this spec as input:

1. Produce the full feature set in one pass. No phased rollout.
2. Keep each file's opening docstring short and accurate — it should name
   what the file does, not how it evolved.
3. Prefer one-pass implementations. Avoid refactoring for abstraction's
   sake.
4. When naming is ambiguous, pick the name a reader would guess first.
5. The project's value is measured by how quickly a new reader can trace a
   task through the code and understand *every* step. Optimise for that,
   not for lines saved or clever abstractions.
