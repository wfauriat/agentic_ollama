# Project Specification — Educational Ollama Agent (Synthetic)

> A condensed rebuild brief. Same scope as the full spec, stripped of
> exact constants, snippets, and prose an attentive reader can derive
> from the principles.

---

## 1. Purpose

Local end-to-end agentic loop over the Ollama API: a single-file Python
REPL that reasons, calls tools, streams, persists memory, and performs
Retrieval-Augmented Generation over a workspace. The goal is
*understanding* — the code should teach an attentive reader how a minimal
tool-using agent works, how streaming differs from blocking, how RAG
integrates with tool calling, and which practical failures (runaway
verification loops, path traversal, shell injection) are guarded against.

---

## 2. Guiding Principles (non-negotiable)

- **Readability over performance.** A slower, clearer path beats a faster opaque one.
- **Single file.** One Python file for the agent; the indexer is the only other script. No packages, no `src/`.
- **Minimal abstraction.** Module-level functions and constants are the default. No classes, DI, or config frameworks unless unavoidable.
- **Comment the *why*.** Section banners segment the file; comments explain invariants and failure modes, not what a named variable already says.
- **Errors as strings at the tool boundary.** Tool failures return `"ERROR: ..."` to the model, not raised exceptions. The model reads and adapts.
- **One path per concept.** No feature flags, no back-compat shims, no premature helpers. Three similar lines is fine.
- **Teach, don't ship.** Accompanying docs explain *why* each pattern exists.

---

## 3. Prerequisites

- Python 3.10+ (uses `match` and `str | None` syntax).
- Ollama running locally at `http://localhost:11434`.
- Models pulled:
  - `qwen3:8b` — reasoning + tool calling
  - `nomic-embed-text` — embeddings for RAG
- Three Python deps: `requests`, `rich`, `numpy`.

---

## 4. Core Architecture

The agent is a stateless loop over a growing `messages` list — the model
keeps no memory beyond what you resend. Each iteration:

1. Call the model with the current messages.
2. If the reply carries tool calls: intercept `done` first; otherwise append the assistant turn, execute each tool, append each result, and continue.
3. If the reply is plain content: append, render, return.
4. If the reply is empty: stop — never spin.

Message roles: `system` (once), `user` (each task), `assistant`
(`content`, `thinking`, `tool_calls`), `tool` (result string per executed
call).

Tool definitions follow the standard OpenAI-style schema. Descriptions
are the tool's *contract with the model* — write them so overlapping
tools (e.g. substring vs semantic search) steer the choice clearly.

Two Ollama endpoints: chat (streaming or blocking) and embed. Streaming
yields newline-JSON chunks; `tool_calls` can appear in any chunk, not
just the last — accumulate across the whole stream, and test on length,
not truthiness (`[]` is falsy). Both modes must coexist and toggle at
runtime.

---

## 5. Tool Set

All filesystem tools route through a single path-traversal guard and
operate inside one workspace directory. LLM-provided Python runs via
subprocess with a short timeout; never `shell=True`.

- **Filesystem:** list, read, write, mkdir, move, copy, delete (non-recursive).
- **Search:** a case-insensitive substring scan and a semantic `rag_search`. Descriptions steer the model toward the semantic tool for conceptual or large-file queries.
- **Execution:** `run_python` — subprocess with timeout, returns stdout/stderr/exit code.
- **Memory:** `remember`, `forget`, `recall` over a flat KV JSON file that is re-read and injected into the system prompt on each call.
- **Completion:** `done` — explicit end signal. The loop intercepts it *before* dispatching other calls from the same turn, so a small model's urge to "verify once more" can't stall the task.

Every completed task writes a timestamped JSON log of `{task, messages}`
— this is the primary debugging artefact.

---

## 6. REPL

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

## 7. RAG

A standalone indexer script walks the workspace, skipping hidden paths
and binary-ish extensions, chunks text by lines with overlap (so facts
that straddle a boundary still retrieve), embeds each chunk, and writes
an index as a flat list of `{file, chunk_id, text, vector}` records.

The agent loads the index at startup, converts vectors to numpy once,
and ranks by cosine similarity (with a zero-denominator guard) for top-k
retrieval. The `rag_search` tool wraps this and returns readable,
source-attributed blocks.

RAG pays off only when material exceeds the context window. For small
workspaces `cat` + substring search usually suffice — the system prompt
and tool descriptions say so, so the model doesn't over-reach for
embeddings.

---

## 8. System Prompt

A single short base prompt: names the agent, scopes it to the sandboxed
workspace, lists tool *categories* (not every name), steers `rag_search`
toward conceptual or indexed-document queries, and tells the model to
call `done` as soon as the task is complete — no extra verification
unless asked. When persisted facts are non-empty, they are appended as a
bullet list. The `persona` command replaces the prompt at runtime.

---

## 9. Display (Rich)

Four panel styles distinguish the loop's phases at a glance:

- **Thinking** — dim/italic border.
- **Tool call** — yellow border, JSON-highlighted args.
- **Tool result** — green border.
- **Final answer** — blue border.

Steps are separated with `console.rule`. Streaming thinking tokens print
inline in dim ANSI; streaming content is buffered and rendered in a panel
at the end — never mix raw `print` with Rich panels. Per-step and
cumulative token counts are shown.

---

## 10. Acceptance Tasks

The finished REPL must handle these end-to-end without exceeding the
iteration cap, each ending via the `done` tool:

- List files in the workspace.
- Create a directory and a file with given content inside it.
- Read a document and summarise it.
- Create a backup directory and copy a file into it.
- Generate a small structured dataset (e.g. a CSV of European capitals and populations).

After indexing the markdown docs:

- Retrieve passages about the `done` tool pattern.
- Retrieve passages about streaming.
