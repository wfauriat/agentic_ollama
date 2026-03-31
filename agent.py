"""
agent_v4.py — v3 + RAG (Retrieval-Augmented Generation).

New in v4:
  - rag_search(query, top_k) tool: embeds query, retrieves top-k relevant chunks
  - index REPL command: rebuild rag_index.json on demand
  - RAG index loaded at startup from rag_index.json (built by rag_indexer.py)

All v3 features preserved.
"""

import datetime
import json
import pathlib
import readline  # noqa: F401
import shutil
import subprocess
import sys

import numpy as np
import requests
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

# ─────────────────────────── Constants ───────────────────────────────────────

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
CONVERSATION_MODE = False
STREAM            = True
RAG_TOP_K         = 5     # number of chunks returned by rag_search

console = Console()

# ─────────────────────────── RAG: index + retrieval ──────────────────────────

def load_index() -> list[dict]:
    """Load rag_index.json. Returns [] if not found."""
    if not INDEX_FILE.exists():
        return []
    try:
        entries = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        # Pre-convert vectors to numpy arrays for fast similarity computation
        for e in entries:
            e["vector"] = np.array(e["vector"], dtype=np.float32)
        return entries
    except Exception as e:
        console.print(f"[yellow]Warning: could not load RAG index: {e}[/yellow]")
        return []


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Range: -1 (opposite) to 1 (identical)."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def embed_query(text: str) -> np.ndarray:
    """Embed a query string using nomic-embed-text."""
    resp = requests.post(
        EMBED_URL,
        json={"model": EMBED_MODEL, "input": text},
        timeout=60,
    )
    resp.raise_for_status()
    return np.array(resp.json()["embeddings"][0], dtype=np.float32)


def retrieve(query: str, index: list[dict], top_k: int) -> list[dict]:
    """Return the top_k most similar chunks to the query."""
    if not index:
        return []
    q_vec  = embed_query(query)
    scored = [(cosine_similarity(q_vec, e["vector"]), e) for e in index]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:top_k]]

# ─────────────────────────── Persistent Memory ───────────────────────────────

def load_memory() -> dict:
    if MEMORY_FILE.exists():
        try:
            return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_memory(mem: dict) -> None:
    MEMORY_FILE.write_text(json.dumps(mem, indent=2, ensure_ascii=False), encoding="utf-8")

# ─────────────────────────── System Prompt ───────────────────────────────────

BASE_SYSTEM_PROMPT = (
    "You are a helpful file-system agent. You operate exclusively inside a "
    "sandboxed workspace directory. You have tools to list, read, write, search, "
    "execute Python, move, copy, and delete files within that workspace. "
    "You have a rag_search tool to semantically retrieve relevant passages from "
    "indexed documents — use it when asked to search indexed documents or for conceptual queries. "
    "You also have tools to remember and forget facts across sessions. "
    "Complete the user's task step by step. "
    "As soon as the task is fully accomplished, call the 'done' tool with a short "
    "summary — do not do extra verification steps unless specifically asked."
)

def build_system_prompt(custom: str | None = None) -> str:
    base = custom if custom else BASE_SYSTEM_PROMPT
    mem  = load_memory()
    if mem:
        facts = "\n".join(f"  - {k}: {v}" for k, v in mem.items())
        return base + f"\n\nFacts you remember from past sessions:\n{facts}"
    return base

# ─────────────────────────── Tool Definitions ────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ls",
            "description": "List files and directories at a path inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path. Defaults to '.'."}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cat",
            "description": "Read the full text content of a file. Use for short files. For large files or open-ended questions, prefer rag_search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the file."}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text content to a file, creating it (and parent dirs) if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path":    {"type": "string", "description": "Relative file path."},
                    "content": {"type": "string", "description": "Text content to write."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mkdir",
            "description": "Create a directory (and parents) inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path of directory."}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mv",
            "description": "Move or rename a file or directory inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "src":  {"type": "string", "description": "Source relative path."},
                    "dest": {"type": "string", "description": "Destination relative path."},
                },
                "required": ["src", "dest"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cp",
            "description": "Copy a file or directory inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "src":  {"type": "string", "description": "Source relative path."},
                    "dest": {"type": "string", "description": "Destination relative path."},
                },
                "required": ["src", "dest"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rm",
            "description": "Delete a file or empty directory inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to delete."}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for a text pattern across workspace files (case-insensitive substring match). Use for finding specific known strings or keywords.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Text to search."},
                    "path":    {"type": "string", "description": "Directory to search in. Defaults to '.'."},
                    "glob":    {"type": "string", "description": "File glob filter. Defaults to '*'."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Execute a Python code snippet and return stdout/stderr. 15s timeout.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute."},
                },
                "required": ["code"],
            },
        },
    },
    # ── NEW in v4 ─────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": (
                "Semantically search indexed workspace documents using vector similarity. "
                "Use when explicitly asked to search the indexed documents, or when a query "
                "is conceptual and keyword search would be insufficient. "
                "Returns the most relevant text chunks with their source file. "
                "The index must have been built with rag_indexer.py first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query to search for.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": f"Number of chunks to return. Defaults to {RAG_TOP_K}.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Save a fact to persistent memory for future sessions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key":   {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget",
            "description": "Remove a fact from persistent memory.",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Read all facts currently in persistent memory.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Call when task is fully complete. Do NOT call other tools after this.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Short summary of what was done."}
                },
                "required": ["summary"],
            },
        },
    },
]

# ─────────────────────────── Safety ──────────────────────────────────────────

def safe_path(rel: str) -> pathlib.Path:
    resolved = (WORK_DIR / rel).resolve()
    if not resolved.is_relative_to(WORK_DIR):
        raise ValueError(f"Path traversal blocked: '{rel}' resolves outside workspace.")
    return resolved

# ─────────────────────────── Tool Executor ───────────────────────────────────

def execute_tool(name: str, arguments: dict, rag_index: list[dict]) -> str:
    try:
        match name:
            case "ls":
                p = safe_path(arguments.get("path", "."))
                entries = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name))
                lines = [f"{'DIR ' if e.is_dir() else 'FILE'} {e.name}" for e in entries]
                return "\n".join(lines) if lines else "(empty directory)"

            case "cat":
                p = safe_path(arguments["path"])
                return p.read_text(encoding="utf-8", errors="replace")

            case "write_file":
                p = safe_path(arguments["path"])
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(arguments["content"], encoding="utf-8")
                return f"Written {len(arguments['content'])} chars to {p.relative_to(WORK_DIR)}"

            case "mkdir":
                p = safe_path(arguments["path"])
                p.mkdir(parents=True, exist_ok=True)
                return f"Directory created: {p.relative_to(WORK_DIR)}"

            case "mv":
                src  = safe_path(arguments["src"])
                dest = safe_path(arguments["dest"])
                shutil.move(str(src), str(dest))
                return f"Moved {src.relative_to(WORK_DIR)} → {dest.relative_to(WORK_DIR)}"

            case "cp":
                src  = safe_path(arguments["src"])
                dest = safe_path(arguments["dest"])
                if src.is_dir():
                    shutil.copytree(str(src), str(dest))
                else:
                    shutil.copy2(str(src), str(dest))
                return f"Copied {src.relative_to(WORK_DIR)} → {dest.relative_to(WORK_DIR)}"

            case "rm":
                p = safe_path(arguments["path"])
                if p.is_dir():
                    p.rmdir()
                else:
                    p.unlink()
                return f"Deleted: {p.relative_to(WORK_DIR)}"

            case "search":
                base    = safe_path(arguments.get("path", "."))
                pattern = arguments["pattern"].lower()
                glob    = arguments.get("glob", "*")
                matches = []
                for file in base.rglob(glob):
                    if not file.is_file():
                        continue
                    try:
                        for i, line in enumerate(
                            file.read_text(encoding="utf-8", errors="replace").splitlines(),
                            start=1,
                        ):
                            if pattern in line.lower():
                                matches.append(f"{file.relative_to(WORK_DIR)}:{i}: {line.rstrip()}")
                    except OSError:
                        pass
                return "\n".join(matches[:100]) if matches else f"No matches for '{arguments['pattern']}'"

            case "run_python":
                result = subprocess.run(
                    [sys.executable, "-c", arguments["code"]],
                    capture_output=True, text=True, timeout=15, cwd=str(WORK_DIR),
                )
                parts = []
                if result.stdout.strip(): parts.append(f"stdout:\n{result.stdout.strip()}")
                if result.stderr.strip(): parts.append(f"stderr:\n{result.stderr.strip()}")
                if result.returncode != 0: parts.append(f"exit code: {result.returncode}")
                return "\n".join(parts) if parts else "(no output)"

            # ── NEW: rag_search ───────────────────────────────────────────────
            case "rag_search":
                if not rag_index:
                    return (
                        "RAG index is empty. Run: python rag_indexer.py\n"
                        "Or use the 'index' command in the REPL."
                    )
                top_k   = int(arguments.get("top_k", RAG_TOP_K))
                chunks  = retrieve(arguments["query"], rag_index, top_k)
                if not chunks:
                    return "No relevant chunks found."
                parts = []
                for i, c in enumerate(chunks, 1):
                    parts.append(
                        f"[{i}] {c['file']} (chunk {c['chunk_id']})\n{c['text']}"
                    )
                return "\n\n---\n\n".join(parts)

            case "remember":
                mem = load_memory()
                mem[arguments["key"]] = arguments["value"]
                save_memory(mem)
                return f"Remembered: {arguments['key']} = {arguments['value']}"

            case "forget":
                mem = load_memory()
                if arguments["key"] in mem:
                    del mem[arguments["key"]]
                    save_memory(mem)
                    return f"Forgot: {arguments['key']}"
                return f"Key not found: {arguments['key']}"

            case "recall":
                mem = load_memory()
                return "\n".join(f"{k}: {v}" for k, v in mem.items()) if mem else "(memory is empty)"

            case "done":
                return arguments.get("summary", "Task complete.")

            case _:
                return f"ERROR: Unknown tool '{name}'"

    except subprocess.TimeoutExpired:
        return "ERROR: Python execution timed out (15s limit)"
    except Exception as e:
        return f"ERROR: {e}"

# ─────────────────────────── Session Logging ─────────────────────────────────

def save_session_log(task: str, messages: list) -> None:
    LOGS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path  = LOGS_DIR / f"{timestamp}.json"
    log_path.write_text(
        json.dumps({"task": task, "messages": messages}, indent=2, default=str),
        encoding="utf-8",
    )
    console.print(f"[dim]Session log → logs/{log_path.name}[/dim]")

# ─────────────────────────── Ollama API ──────────────────────────────────────

def call_model(messages: list) -> tuple[dict, int, int]:
    payload = {
        "model": MODEL, "stream": False, "think": THINK,
        "tools": TOOLS, "messages": messages,
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["message"], data.get("prompt_eval_count", 0), data.get("eval_count", 0)


def call_model_stream(messages: list) -> tuple[dict, int, int]:
    payload = {
        "model": MODEL, "stream": True, "think": THINK,
        "tools": TOOLS, "messages": messages,
    }
    resp = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=120)
    resp.raise_for_status()

    full_thinking = ""
    full_content  = ""
    tool_calls    = []
    in_thinking   = False
    in_tokens     = 0
    out_tokens    = 0

    for raw_line in resp.iter_lines():
        if not raw_line:
            continue
        chunk = json.loads(raw_line)
        msg   = chunk.get("message", {})

        if msg.get("tool_calls"):
            tool_calls.extend(msg["tool_calls"])

        thinking_token = msg.get("thinking", "")
        if thinking_token:
            if not in_thinking:
                print("\n\033[2m── Thinking ", end="", flush=True)
                in_thinking = True
            print(f"\033[2m{thinking_token}\033[0m", end="", flush=True)
            full_thinking += thinking_token

        content_token = msg.get("content", "")
        if content_token:
            if in_thinking:
                print("\033[0m")
                in_thinking = False
            full_content += content_token

        if chunk.get("done"):
            if DEBUG:
                print(f"\n[DEBUG final chunk] {json.dumps(chunk, default=str)}", flush=True)
            in_tokens  = chunk.get("prompt_eval_count", 0)
            out_tokens = chunk.get("eval_count", 0)
            break

    if in_thinking:
        print()

    return (
        {"role": "assistant", "content": full_content,
         "thinking": full_thinking, "tool_calls": tool_calls},
        in_tokens,
        out_tokens,
    )

# ─────────────────────────── Display ─────────────────────────────────────────

def show_thinking(text: str) -> None:
    if text and text.strip():
        console.print(Panel(text.strip(), title="[dim]Thinking[/dim]",
                            border_style="dim", style="dim italic"))

def show_tool_call(name: str, args: dict) -> None:
    console.print(Panel(
        Syntax(json.dumps(args, indent=2), "json", theme="monokai"),
        title=f"[bold yellow]Tool Call → {name}[/bold yellow]", border_style="yellow",
    ))

def show_tool_result(name: str, result: str) -> None:
    console.print(Panel(result, title=f"[bold green]Result ← {name}[/bold green]",
                        border_style="green"))

def show_final_answer(text: str) -> None:
    console.print(Panel(text.strip(), title="[bold blue]Agent[/bold blue]",
                        border_style="blue"))

# ─────────────────────────── Agent Loop ──────────────────────────────────────

def run_agent(task: str, messages: list, rag_index: list[dict]) -> list:
    messages.append({"role": "user", "content": task})
    console.rule(f"[bold]Task[/bold]: {task}")

    total_in  = 0
    total_out = 0

    for step in range(1, MAX_ITER + 1):
        console.rule(f"Step {step}/{MAX_ITER}", style="dim")

        if DEBUG:
            console.print_json(json.dumps(messages, default=str))

        msg, in_tok, out_tok = (
            call_model_stream(messages) if STREAM else call_model(messages)
        )
        total_in  += in_tok
        total_out += out_tok
        console.print(
            f"[dim]tokens → in: {in_tok:,}  out: {out_tok:,}  "
            f"(task total in: {total_in:,}  out: {total_out:,})[/dim]"
        )

        if not STREAM:
            show_thinking(msg.get("thinking", ""))

        if msg.get("tool_calls") is not None and len(msg["tool_calls"]) > 0:
            for tc in msg["tool_calls"]:
                if tc["function"]["name"] == "done":
                    summary = tc["function"]["arguments"].get("summary", "Task complete.")
                    show_final_answer(summary)
                    messages.append(msg)
                    save_session_log(task, messages)
                    return messages

            messages.append(msg)
            for tc in msg["tool_calls"]:
                name   = tc["function"]["name"]
                args   = tc["function"]["arguments"]
                show_tool_call(name, args)
                result = execute_tool(name, args, rag_index)
                show_tool_result(name, result)
                messages.append({"role": "tool", "content": result})

        elif msg.get("content", "").strip():
            messages.append(msg)
            show_final_answer(msg["content"])
            save_session_log(task, messages)
            return messages

        else:
            console.print("[red]Empty response — stopping.[/red]")
            return messages

    console.print(f"[red]Reached max iterations ({MAX_ITER}).[/red]")
    save_session_log(task, messages)
    return messages

# ─────────────────────────── REPL ────────────────────────────────────────────

REPL_COMMANDS = {
    "help":    "Show this help",
    "history": "List tasks this session",
    "memory":  "Show persisted memory facts",
    "index":   "Rebuild RAG index (runs rag_indexer.py)",
    "debug":   "Toggle DEBUG",
    "think":   "Toggle THINK",
    "stream":  "Toggle STREAM",
    "convo":   "Toggle CONVERSATION_MODE",
    "clear":   "Clear conversation memory",
    "persona": "Edit system prompt",
    "quit":    "Exit",
}

def main() -> None:
    global DEBUG, THINK, STREAM, CONVERSATION_MODE

    # Load RAG index at startup
    rag_index = load_index()
    index_status = (
        f"{len(rag_index)} chunks from "
        f"{len({e['file'] for e in rag_index})} file(s)"
        if rag_index else "not found — run: python rag_indexer.py"
    )

    custom_system_prompt: str | None = None

    console.print(Panel(
        f"[bold]Agentic AI REPL v4[/bold]\n"
        f"Model        : {MODEL}\n"
        f"Embed model  : {EMBED_MODEL}\n"
        f"RAG index    : {index_status}\n"
        f"Thinking     : {THINK}   Streaming: {STREAM}   Debug: {DEBUG}\n"
        f"Workspace    : {WORK_DIR}\n\n"
        "Commands: " + ", ".join(f"[bold]{c}[/bold]" for c in REPL_COMMANDS),
        title="Welcome", border_style="cyan",
    ))

    task_history: list[str] = []

    def fresh_messages() -> list:
        return [{"role": "system", "content": build_system_prompt(custom_system_prompt)}]

    conversation: list[dict] = fresh_messages()

    while True:
        try:
            user_input = input("\n[You] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye.[/dim]")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in ("quit", "exit", "q"):
            console.print("[dim]Bye.[/dim]")
            break
        elif cmd == "help":
            for c, desc in REPL_COMMANDS.items():
                console.print(f"  [bold]{c:10}[/bold] {desc}")
        elif cmd == "history":
            [console.print(f"  {i}. {t}") for i, t in enumerate(task_history, 1)] or console.print("[dim]No tasks yet.[/dim]")
        elif cmd == "memory":
            mem = load_memory()
            [console.print(f"  [bold]{k}[/bold]: {v}") for k, v in mem.items()] or console.print("[dim]Memory is empty.[/dim]")
        elif cmd == "index":
            console.print("[dim]Rebuilding RAG index...[/dim]")
            result = subprocess.run(
                [sys.executable, str(WORK_DIR / "rag_indexer.py")],
                cwd=str(WORK_DIR),
            )
            if result.returncode == 0:
                rag_index = load_index()
                console.print(f"[green]Index reloaded: {len(rag_index)} chunks.[/green]")
        elif cmd == "debug":
            DEBUG = not DEBUG
            console.print(f"[yellow]DEBUG = {DEBUG}[/yellow]")
        elif cmd == "think":
            THINK = not THINK
            console.print(f"[yellow]THINK = {THINK}[/yellow]")
        elif cmd == "stream":
            STREAM = not STREAM
            console.print(f"[yellow]STREAM = {STREAM}[/yellow]")
        elif cmd == "convo":
            CONVERSATION_MODE = not CONVERSATION_MODE
            console.print(f"[yellow]CONVERSATION_MODE = {CONVERSATION_MODE}[/yellow]")
        elif cmd == "clear":
            conversation = fresh_messages()
            console.print("[yellow]Conversation memory cleared.[/yellow]")
        elif cmd == "persona":
            console.print("[dim]Enter new system prompt (blank line to finish, 'reset' to restore):[/dim]")
            lines = []
            while True:
                try:
                    line = input()
                except (EOFError, KeyboardInterrupt):
                    break
                if line.strip().lower() == "reset":
                    custom_system_prompt = None
                    console.print("[yellow]System prompt reset.[/yellow]")
                    break
                if line == "" and lines:
                    custom_system_prompt = "\n".join(lines)
                    console.print("[yellow]System prompt updated.[/yellow]")
                    break
                lines.append(line)
            if conversation:
                conversation[0] = {"role": "system", "content": build_system_prompt(custom_system_prompt)}
        else:
            task_history.append(user_input)
            try:
                if CONVERSATION_MODE:
                    conversation = run_agent(user_input, conversation, rag_index)
                else:
                    run_agent(user_input, fresh_messages(), rag_index)
            except requests.exceptions.ConnectionError:
                console.print("[red]Cannot reach Ollama. Try: ollama serve[/red]")
            except Exception as e:
                console.print(f"[red]Unexpected error: {e}[/red]")


if __name__ == "__main__":
    main()
