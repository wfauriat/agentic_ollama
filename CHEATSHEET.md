# Agentic AI Cheatsheet

## The Mental Model

```
Task (natural language)
  → LLM reasons → picks tool → you execute it → feed result back → repeat
  → LLM produces text (no tool call) → DONE
```

The agent has no special powers. It's a loop + a growing list + an LLM that
can read JSON schemas.

---

## Version Changelog

| Feature                        | v1 `agent.py` | v2 `agent_v2.py` | v3 `agent_v3.py` |
|-------------------------------|:---:|:---:|:---:|
| Core agentic loop             | ✓   | ✓   | ✓   |
| Tools: ls/cat/write/mkdir/mv/cp/rm | ✓ | ✓ | ✓ |
| Rich display (panels)         | ✓   | ✓   | ✓   |
| Thinking tokens (THINK flag)  | ✓   | ✓   | ✓   |
| DEBUG flag (messages dump)    | ✓*  | ✓   | ✓   |
| Session logging to logs/      |     | ✓   | ✓   |
| `search` tool (grep-like)     |     | ✓   | ✓   |
| `history` REPL command        |     | ✓   | ✓   |
| Conversation mode (convo)     |     | ✓   | ✓   |
| **Streaming responses**       |     |     | ✓   |
| **Persistent memory (JSON)**  |     |     | ✓   |
| **`run_python` tool**         |     |     | ✓   |
| **`persona` system prompt editor** |  |    | ✓   |
| `remember` / `forget` / `recall` tools | | | ✓ |
| `memory` REPL command         |     |     | ✓   |
| `stream` REPL toggle          |     |     | ✓   |
| **Token counting (per-step + cumulative)** | | | ✓ |
| **`done` tool (explicit stop signal)** |  |  | ✓ |

*v1: uncomment a line manually

---

## Ollama Quick Reference

```bash
ollama serve                        # start the server (if not a service)
ollama list                         # show pulled models
ollama pull qwen3:8b                # pull a model
ollama ps                           # show running models
curl localhost:11434/api/tags       # check server is up (JSON response)
```

### Minimal API call (curl)
```bash
curl http://localhost:11434/api/chat -d '{
  "model": "qwen3:8b",
  "stream": false,
  "messages": [{"role":"user","content":"say hi"}]
}'
```

### Blocking call — tools + thinking (Python)
```python
payload = {
    "model":    "qwen3:8b",
    "stream":   False,
    "think":    True,           # reasoning tokens → msg["thinking"]
    "tools":    TOOLS,          # list of JSON schema tool defs
    "messages": messages,       # full history, sent every call
}
resp = requests.post("http://localhost:11434/api/chat", json=payload, timeout=120)
msg = resp.json()["message"]
# msg keys: "role", "content", "thinking", "tool_calls"
```

### Streaming call (Python)
```python
payload = {**payload, "stream": True}
resp = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=120)

full_content = ""
for raw_line in resp.iter_lines():       # one JSON object per line
    chunk = json.loads(raw_line)
    token = chunk["message"].get("content", "")
    print(token, end="", flush=True)     # live output
    full_content += token
    if chunk["done"]:
        tool_calls = chunk["message"].get("tool_calls", [])
        break                            # tool_calls only in final chunk!
```

**Key streaming rules:**
- `"stream": True` → response body is newline-delimited JSON (one chunk per line)
- Thinking tokens stream in `chunk["message"]["thinking"]`
- Content tokens stream in `chunk["message"]["content"]`
- Tool calls can appear in **any** chunk — accumulate across all, not just `done: true`
- `[]` (empty list) is falsy in Python — use `len(tool_calls) > 0`, not `if tool_calls`
- Must accumulate content+thinking while streaming; reconstruct the message dict at the end
- Final answer: buffer content, display in panel — don't mix `print()` and Rich panels

---

## Message Roles

| Role        | When added                          | Key fields                     |
|-------------|-------------------------------------|-------------------------------|
| `system`    | Once at start                       | `content`                     |
| `user`      | Each user turn                      | `content`                     |
| `assistant` | Model reply                         | `content` or `tool_calls`     |
| `tool`      | After you execute a tool            | `content` (the result string) |

**Rule:** assistant turn with `tool_calls` must be appended BEFORE appending
tool results.

---

## Tool Definition Schema

```python
{
    "type": "function",
    "function": {
        "name": "my_tool",
        "description": "What it does — the model reads this to decide when to use it.",
        "parameters": {
            "type": "object",
            "properties": {
                "arg1": {"type": "string", "description": "..."},
                "arg2": {"type": "integer", "description": "..."},
            },
            "required": ["arg1"],   # omit optional args from this list
        },
    },
}
```

**Tips:**
- Description quality matters — vague descriptions = wrong tool choices
- Mark args as `required` only when truly mandatory
- Return errors as strings, not exceptions — the model can read and adapt

---

## The Agent Loop (pseudocode)

```python
messages = [system, user_task]
for step in range(MAX_ITER):
    msg = call_model(messages)          # → thinking + tool_calls or content
    if msg.tool_calls:
        messages.append(msg)            # IMPORTANT: append before tool results
        for tc in msg.tool_calls:
            result = execute(tc)
            messages.append({"role": "tool", "content": result})
        # loop again — model sees results
    elif msg.content:
        show(msg.content)
        break                           # natural stop condition
```

---

## Key Concepts

### Memory = the messages list
The model is stateless. Everything it "knows" is what you put in `messages`.
Add something → it knows it. Remove something → it forgets it.

### Thinking tokens
`"think": true` makes qwen3 emit a reasoning chain before acting.
Slower but more reliable on multi-step tasks.
`msg["thinking"]` contains the raw reasoning text.
Toggle: set `THINK = False` for speed, `True` for quality.

### Streaming vs blocking
Blocking: one POST → wait → full JSON response. Simple but silent.
Streaming: one POST → iterate lines → tokens appear live. Better UX.
The agent loop logic is identical — only `call_model` differs.

### Persistent memory vs conversation memory
| Type | Scope | Stored in |
|---|---|---|
| `messages` list | Current task only | RAM |
| Conversation mode | Current session | RAM |
| `memory.json` | Across sessions | Disk (`remember`/`forget` tools) |

### System prompt = agent's personality
Everything the agent does is shaped by the system prompt.
Same task, different prompt → completely different behaviour.
Experiment with: strict vs. loose, verbose vs. terse, domain-specific personas.

### Tool call batching
The model can return multiple `tool_calls` in one response.
Always loop: `for tc in msg["tool_calls"]`

### Stop condition — the `done` tool pattern
Relying on the model to produce a text-only response is fragile: small models
often keep calling tools even when the task is done.

The robust pattern: add a `done` tool and instruct the model to call it when finished.
The loop intercepts it and exits immediately.

```python
# Tool definition
{"name": "done", "description": "Call when task is fully complete. Do NOT call other tools after this.",
 "parameters": {"properties": {"summary": {"type": "string"}}, "required": ["summary"]}}

# Loop intercept (check BEFORE executing other tools)
for tc in msg["tool_calls"]:
    if tc["function"]["name"] == "done":
        show_final_answer(tc["function"]["arguments"]["summary"])
        return messages   # ← clean exit

# System prompt instruction (critical — model must know to use it)
"As soon as the task is accomplished, call the 'done' tool with a short summary."
```

Why this works better:
- Explicit contract: model knows exactly when to stop
- No ambiguity between "tool-calling turn" and "final text turn"
- Prevents verification loops (model re-reading its own work unnecessarily)

### Token counting
Ollama returns counts at the top level of every response (blocking and streaming final chunk):
```python
data["prompt_eval_count"]   # tokens in the full prompt sent to the model
data["eval_count"]          # tokens the model generated
```
The `prompt_eval_count` grows with every step because the full `messages` list
(history + tool results + tool schemas) is sent on every call. Watching this
number is the best way to understand context cost in practice.

### Context window
qwen3:8b: 40,960 tokens. Long sessions accumulate messages.
Watch for: model starts ignoring early context → time to summarise or truncate.

### run_python safety
- `subprocess.run([sys.executable, "-c", code], timeout=15, cwd=WORK_DIR)`
- No `shell=True` — avoids shell injection
- Timeout prevents infinite loops
- Runs as current user — already low-permission on this machine

---

## Safety Idioms

```python
# Path traversal guard (use for every filesystem tool)
resolved = (WORK_DIR / user_input).resolve()
if not resolved.is_relative_to(WORK_DIR):
    raise ValueError("Path traversal blocked")

# Safe subprocess (no shell injection)
subprocess.run([sys.executable, "-c", code], timeout=15)  # safe
subprocess.run(f"python -c {code}", shell=True)            # UNSAFE

# Return errors as strings (not raise)
try:
    ...
except Exception as e:
    return f"ERROR: {e}"            # model reads this and adapts
```

---

## REPL Commands (v3)

| Command   | Effect |
|-----------|--------|
| `help`    | List all commands |
| `history` | Show tasks run this session |
| `memory`  | Show persisted memory facts |
| `debug`   | Toggle DEBUG (dump messages list each step) |
| `think`   | Toggle THINK (reasoning tokens) |
| `stream`  | Toggle STREAM (live token output) |
| `convo`   | Toggle CONVERSATION_MODE (memory within session) |
| `clear`   | Wipe conversation memory |
| `persona` | Edit system prompt live |
| `quit`    | Exit |

---

## Exploration Roadmap

### Easy ✓ (implemented in v1 + v2)
- `DEBUG` flag — clean toggle for messages inspection
- Session log — save messages to `logs/` after each task
- `search` tool — grep-like text search within workspace files
- `history` REPL command — show tasks attempted this session
- Conversation mode — memory within a session

### Medium ✓ (implemented in v3)
- **Streaming** — tokens appear live as model generates them
- **Persistent memory** — `memory.json` survives restarts; `remember`/`forget` tools
- **`run_python` tool** — agent writes and runs Python, captures output
- **`persona` command** — rewrite system prompt at runtime
- **Token counting** — per-step and cumulative in/out token display
- **`done` tool** — explicit stop signal, fixes runaway verification loops

### Harder (next steps)
- **ReAct from scratch** — tool calling via text parsing, no native API support
  Teaches: how tool use worked before structured outputs; regex fragility
- **Planning step** — agent writes a plan first, then executes step by step
  Pattern: two-phase prompt (plan → act)
- **Self-critique loop** — agent scores its own output and optionally retries
  Pattern: generator + critic + conditional retry
- **RAG / embeddings** — embed workspace files, retrieve relevant chunks for context
  Tools: `ollama embeddings` endpoint, cosine similarity, `numpy`
- **Multi-agent** — orchestrator + specialist sub-agents
  Pattern: one agent plans, others execute, orchestrator aggregates

### Advanced
- **Long-context management** — sliding window, summarisation, importance scoring
- **Tool-use fine-tuning** — generate synthetic tool-call traces, fine-tune a model
- **Autonomous loop** — runs without human in the loop until goal achieved
- **Evaluation harness** — define tasks with expected outcomes, score automatically

---

## Comparison: Our Agent vs Claude Code

The core loop is identical in principle. The differences are engineering depth,
not architectural novelty.

| Dimension | Our agent (v3) | Claude Code |
|---|---|---|
| **Model** | qwen3:8b, local via Ollama | Claude Opus/Sonnet, Anthropic API |
| **Tool set** | 7 FS tools + run_python + memory | Read, Write, Edit, Bash, Glob, Grep, WebFetch, WebSearch, Agent, + MCP |
| **Stop signal** | `done` tool | Similar (inferred from behaviour) |
| **Streaming** | thinking inline + buffered answer | Full streaming |
| **Token/context mgmt** | display only | Auto-summarises old context near limit |
| **Memory** | `memory.json` (flat KV) | `CLAUDE.md` files (hierarchical, per-project, version-controlled) |
| **Permissions** | path traversal guard only | Full model: allowlists, per-tool confirmation, dangerous-op detection |
| **Multi-agent** | not yet | Native — `Agent` tool spawns subagents in parallel or series |
| **Hooks** | none | Pre/post tool-call hooks (shell commands, user-configurable) |
| **MCP** | none | Extends tool set via Model Context Protocol servers |
| **Plan mode** | none | Dedicated planning phase before execution |
| **IDE integration** | none | VS Code, JetBrains, web UI |
| **Open source** | yes (you wrote it) | No — closed-source npm package |

### Key architectural gaps worth understanding

**CLAUDE.md as memory**
Rather than a flat `memory.json`, Claude Code reads `CLAUDE.md` files at the
project root (and parent directories). Hierarchical, human-readable,
version-controlled. This session's `CLAUDE.md` is exactly that pattern.

**Subagents**
The most significant gap. Claude Code can spawn parallel subagents — each with
their own full tool loop — and aggregate results. One agent plans, several run
in parallel, one aggregates. This is the multi-agent item on our roadmap.

**Context summarisation**
As `messages` grows, Claude Code compresses older turns into summaries
automatically. We watch the token count grow and stop. This is our
"long-context management" advanced item.

**Permission taxonomy**
Claude Code distinguishes reversible (read, search) from dangerous (delete,
push, modify CI) actions and asks before the latter. Our `safe_path()` is the
seed of this idea — a full implementation would tier all tools by blast radius.

---

## Constants Quick Reference

```python
# ── All versions ──────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL      = "qwen3:8b"
WORK_DIR   = pathlib.Path(__file__).parent.resolve()
MAX_ITER   = 10     # increase for complex multi-step tasks
THINK      = True   # False = faster, less reliable on hard tasks
DEBUG      = False  # True = print full messages list each step

# ── v3 additions ──────────────────────────────────────────────
STREAM            = True   # False = blocking (v1/v2 behaviour)
CONVERSATION_MODE = False  # True = messages persist across tasks
MEMORY_FILE       = WORK_DIR / "memory.json"
```
