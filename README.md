# ollama_agent

An educational implementation of an agentic AI loop using the local Ollama API.
Covers tool calling, reasoning tokens, streaming, persistent memory, and RAG
(Retrieval-Augmented Generation) — all running locally with no cloud dependency.

Built as a hands-on learning project: each version adds one layer of complexity
so you can understand each concept in isolation before they are combined.

---

## What it demonstrates

- **Agentic loop** — the core pattern: reason → call tool → observe result → repeat → stop
- **Native tool calling** — structured JSON tool dispatch via Ollama's `/api/chat`
- **Thinking tokens** — qwen3's extended reasoning chain, streamed live
- **Streaming** — token-by-token output via Ollama's SSE stream
- **Persistent memory** — key-value facts that survive session restarts (`memory.json`)
- **RAG** — semantic retrieval over workspace files using `nomic-embed-text` embeddings
- **Session logging** — full message history saved to `logs/` after every task
- **Token counting** — per-step and cumulative input/output token display

---

## Prerequisites

- [Ollama](https://ollama.com) installed and running (`ollama serve`)
- Models pulled:
  ```bash
  ollama pull qwen3:8b          # reasoning + tool calling
  ollama pull nomic-embed-text  # embeddings for RAG (agent_v4 only)
  ```
- Python 3.10+ (uses `match` statements and `str | None` type hints)
- Dependencies:
  ```bash
  pip install -r requirements.txt
  ```

---

## Quick start

```bash
# Run the full-featured agent (v4, with RAG)
python agent_v4.py

# Build the RAG index first (for rag_search to work)
python rag_indexer.py --glob "*.md"

# Or start with the minimal agent (v1)
python agent.py
```

---

## Version progression

Each script is self-contained and runnable. They exist to show the evolution
of the design — read them in order to understand how each concept is added.

### `agent.py` — core loop
The minimal educational implementation. Everything else builds on this.

- 7 filesystem tools: `ls`, `cat`, `write_file`, `mkdir`, `mv`, `cp`, `rm`
- Rich display: thinking (dim), tool calls (yellow), results (green), answer (blue)
- `THINK = True/False` toggle for reasoning tokens
- Safety: `safe_path()` blocks path traversal on every tool

### `agent_v2.py` — observability
- `DEBUG` flag: dumps the full `messages` list before each model call
- Session logging to `logs/YYYY-MM-DD_HH-MM-SS.json`
- `search` tool: grep-like text search across workspace files
- `history` REPL command
- `convo` mode: messages persist across tasks within a session

### `agent_v3.py` — medium features
- Streaming: thinking tokens live, final answer in Rich panel
- Persistent memory: `memory.json` + `remember`/`forget`/`recall` tools
- `run_python` tool: execute Python snippets, capture stdout/stderr
- `persona` command: rewrite system prompt at runtime
- Token counting: `in`/`out` per step + cumulative
- `done` tool: explicit stop signal (fixes runaway verification loops)

### `agent_v4.py` — RAG *(current)*
- `rag_search(query, top_k)`: embeds query, retrieves top-k relevant chunks
- `index` REPL command: rebuild the RAG index on demand
- All v3 features included

---

## REPL commands (v4)

| Command   | Effect |
|-----------|--------|
| `help`    | List all commands |
| `history` | Tasks run this session |
| `memory`  | Show persisted facts |
| `index`   | Rebuild RAG index |
| `debug`   | Toggle messages dump |
| `think`   | Toggle reasoning tokens |
| `stream`  | Toggle streaming mode |
| `convo`   | Toggle cross-task memory |
| `clear`   | Wipe conversation memory |
| `persona` | Edit system prompt live |
| `quit`    | Exit |

---

## Architecture

### The agent loop

```
User task
  └─► messages = [system, user_task]
        └─► for step in range(MAX_ITER):
              msg = call_model(messages)        # POST /api/chat
              if msg.tool_calls:
                  messages.append(msg)          # assistant turn first
                  for tc in tool_calls:
                      result = execute(tc)      # filesystem / python / RAG
                      messages.append(tool_result)
                  # loop — model sees results
              elif msg.tool_calls["done"]:
                  show_final_answer             # explicit stop signal
                  return
              elif msg.content:
                  show_final_answer             # fallback stop
                  return
```

The `messages` list is the agent's entire working memory. The model is
stateless — it only knows what is in that list on each call.

### Tool calling

Tools are described as JSON schemas. The model reads them and decides which
to invoke. Description quality directly determines tool selection:

```python
{
    "type": "function",
    "function": {
        "name": "my_tool",
        "description": "When and why to use this — the model reads this.",
        "parameters": {
            "type": "object",
            "properties": {
                "arg": {"type": "string", "description": "..."}
            },
            "required": ["arg"],
        },
    },
}
```

**Key rule:** the assistant turn with `tool_calls` must be appended to
`messages` *before* appending tool results.

### Streaming

```python
# stream=True → newline-delimited JSON chunks
for raw_line in resp.iter_lines():
    chunk = json.loads(raw_line)
    thinking_token = chunk["message"].get("thinking", "")  # reasoning
    content_token  = chunk["message"].get("content",  "")  # answer
    tool_calls     = chunk["message"].get("tool_calls", []) # actions
    # tool_calls can appear in ANY chunk — accumulate across all
    if chunk["done"]:
        in_tokens  = chunk["prompt_eval_count"]
        out_tokens = chunk["eval_count"]
        break
```

### RAG

```
Index time:  file → chunks → embed (nomic-embed-text) → rag_index.json
Query time:  query → embed → cosine similarity → top-k chunks → inject into prompt
```

```python
# Embed
resp   = requests.post("http://localhost:11434/api/embed",
                       json={"model": "nomic-embed-text", "input": text})
vector = resp.json()["embeddings"][0]   # 768 floats

# Retrieve
q_vec  = embed(query)
scored = [(cosine_similarity(q_vec, e["vector"]), e) for e in index]
chunks = [e for _, e in sorted(scored, reverse=True)[:top_k]]
```

RAG only helps when source material exceeds what fits comfortably in context.
For small workspaces, `cat` + `search` are often sufficient.

### Memory layers

| Type | Scope | Stored in |
|---|---|---|
| `messages` list | Current task | RAM |
| `convo` mode | Current session | RAM |
| `memory.json` | Across sessions | Disk |
| `rag_index.json` | Across sessions | Disk |

### The `done` tool pattern

Small models often keep calling tools after a task is complete ("verification
loops"). The fix: give the model an explicit stop signal.

```python
# Tool definition
{"name": "done", "description": "Call when task is fully complete.",
 "parameters": {"properties": {"summary": {"type": "string"}}}}

# Loop intercepts it before executing other tools
for tc in msg["tool_calls"]:
    if tc["function"]["name"] == "done":
        show_final_answer(tc["function"]["arguments"]["summary"])
        return messages

# System prompt must instruct the model to use it
"As soon as the task is accomplished, call the 'done' tool."
```

### Safety

```python
# Path traversal guard — called by every filesystem tool
resolved = (WORK_DIR / user_input).resolve()
if not resolved.is_relative_to(WORK_DIR):
    raise ValueError("Path traversal blocked")

# No shell=True — prevents command injection
subprocess.run([sys.executable, "-c", code], timeout=15)

# Errors returned as strings — model observes and adapts
except Exception as e:
    return f"ERROR: {e}"
```

---

## Suggested tasks to try

```
list the files in the workspace
create a directory called notes
create notes/hello.txt with the content 'hello world'
read CLAUDE.md and summarise it
create a backup dir and copy a file into it
create a CSV with the capitals of all European countries and their population
```

For RAG (after running `rag_indexer.py --glob "*.md"`):
```
what does the documentation say about the done tool pattern?
find everything related to streaming in the indexed documents
summarise the project based on the documentation
```

---

## Possible extensions

### Plan → Act
Split each task into two explicit phases: the model first produces a written
plan (no tools), then executes it step by step. The plan is injected into the
message history so the model stays on track throughout execution. The user can
inspect and edit the plan between phases.

```
Prompt 1: "Write a step-by-step plan for: {task}. Do not execute yet."
          → model returns plan as text
Prompt 2: "Here is the plan:\n{plan}\n\nNow execute it."
          → normal agentic loop
```

### Self-critique
After generating an answer, pass it to a critic prompt that scores it and
identifies issues. If the score is below a threshold, regenerate with the
critique as context. Loop until satisfied or max retries reached.

```
generate answer → critic scores it → if good: return
                                   → if not: regenerate with critique → repeat
```

The critic can be the same model with a different prompt, a separate smaller
model, or a deterministic check (run the code, do tests pass?).

### Multi-agent orchestration
An orchestrator agent whose tools include spawning specialist sub-agents.
The orchestrator decomposes the task, delegates to sub-agents in parallel or
series, and aggregates results. Each sub-agent is just a normal agent loop
called as a function.

```
orchestrator
  ├─► sub-agent A (research)
  ├─► sub-agent B (write)
  └─► sub-agent C (review)
       └─► aggregated result
```

### Context management
As `messages` grows, the model starts ignoring early context. Mitigations:
- Summarise old turns periodically (replace N messages with one summary)
- Sliding window (drop oldest non-system messages)
- Selective retention (keep tool results, drop thinking turns)
- RAG over your own conversation history

---

## Comparison with Claude Code

The core loop is identical in principle. The differences are engineering
depth, not architectural novelty.

| Dimension | This project | Claude Code |
|---|---|---|
| Model | qwen3:8b, local Ollama | Claude Opus/Sonnet, Anthropic API |
| Tool set | FS + run_python + RAG + memory | Read, Write, Edit, Bash, Glob, Grep, WebFetch, Agent, + MCP |
| Stop signal | `done` tool | Similar |
| Streaming | thinking inline + buffered answer | Full streaming |
| Context management | token display only | Auto-summarises near limit |
| Memory | `memory.json` (flat KV) | `CLAUDE.md` (hierarchical, version-controlled) |
| Permissions | path traversal guard | Full model: allowlists, dangerous-op detection |
| Multi-agent | not implemented | Native — `Agent` tool spawns subagents |
| Hooks | none | Pre/post tool-call hooks |
| MCP | none | Extensible via Model Context Protocol |
| Open source | yes | No (closed-source npm package) |

Building up plan→act, self-critique, multi-agent, and context management
from what is here would produce something structurally very close to Claude Code.
There is no new primitive missing — only composition of what already exists.

---

## File reference

| File | Purpose |
|---|---|
| `agent.py` | v1 — minimal core loop |
| `agent_v2.py` | v2 — observability + search |
| `agent_v3.py` | v3 — streaming, memory, run_python |
| `agent_v4.py` | v4 — RAG (current) |
| `rag_indexer.py` | Standalone indexer for RAG |
| `requirements.txt` | Python dependencies |
| `logs/` | Per-session message history (auto-created) |
| `memory.json` | Persistent facts (auto-created on first `remember`) |
| `rag_index.json` | Chunk + vector index (created by rag_indexer.py) |
