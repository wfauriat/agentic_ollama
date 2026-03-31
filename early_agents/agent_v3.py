"""
agent_v3.py — Medium-tier agentic features on top of v2.

New in v3:
  - Streaming responses  : tokens appear live as the model generates them
  - Persistent memory    : remember/forget facts that survive session restarts
  - run_python tool      : agent can write and execute Python code in the workspace
  - persona command      : edit the system prompt at runtime without touching the file

All v2 features are preserved: DEBUG, session logging, search tool,
history command, conversation mode, think/debug/convo toggles.
"""

import datetime
import json
import pathlib
import readline  # noqa: F401
import shutil
import subprocess
import sys

import requests
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

# ─────────────────────────── Constants ───────────────────────────────────────

OLLAMA_URL        = "http://localhost:11434/api/chat"
MODEL             = "qwen3:8b"
WORK_DIR          = pathlib.Path(__file__).parent.resolve()
LOGS_DIR          = WORK_DIR / "logs"
MEMORY_FILE       = WORK_DIR / "memory.json"
MAX_ITER          = 10
THINK             = True
DEBUG             = False
CONVERSATION_MODE = False
STREAM            = True   # False → wait for full response (v1/v2 behaviour)

console = Console()

# ─────────────────────────── Persistent Memory ───────────────────────────────
#
# A simple key-value store in memory.json.
# Loaded once at startup and injected into the system prompt.
# The agent can update it via remember/forget tools.

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
    "You also have tools to remember and forget facts across sessions. "
    "Complete the user's task step by step. "
    "As soon as the task is fully accomplished, call the 'done' tool with a short "
    "summary — do not do extra verification steps unless specifically asked."
)

def build_system_prompt(custom: str | None = None) -> str:
    """Inject persisted memory facts into the system prompt."""
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
            "description": "Read the text content of a file inside the workspace.",
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
            "description": (
                "Search for a text pattern inside files in the workspace. "
                "Returns matching lines with file path and line number. "
                "Optionally filter by file glob (e.g. '*.py')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Text to search (case-insensitive)."},
                    "path":    {"type": "string", "description": "Relative directory to search in. Defaults to '.'."},
                    "glob":    {"type": "string", "description": "File glob filter, e.g. '*.py'. Defaults to '*'."},
                },
                "required": ["pattern"],
            },
        },
    },
    # ── NEW in v3 ─────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute a Python code snippet in the workspace and return its stdout/stderr. "
                "Use this to compute, transform data, or run scripts. "
                "Execution is limited to 15 seconds. The working directory is the workspace root."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute."},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": (
                "Save a fact to persistent memory so it is available in future sessions. "
                "Use this for important context: user preferences, project state, key facts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key":   {"type": "string", "description": "Short label for the fact."},
                    "value": {"type": "string", "description": "The fact to remember."},
                },
                "required": ["key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget",
            "description": "Remove a previously remembered fact from persistent memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Label of the fact to forget."},
                },
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
    # ── Explicit stop signal ──────────────────────────────────────────────────
    # The model calls this when the task is complete. This is the canonical way
    # to stop the loop — more reliable than waiting for a text-only response,
    # because small models sometimes keep calling tools even when done.
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": (
                "Call this when the task is fully complete. "
                "Provide a short summary of what was accomplished. "
                "Do NOT call any other tools after this."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "One or two sentences summarising what was done.",
                    }
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

def execute_tool(name: str, arguments: dict) -> str:
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
                                rel = file.relative_to(WORK_DIR)
                                matches.append(f"{rel}:{i}: {line.rstrip()}")
                    except OSError:
                        pass
                if not matches:
                    return f"No matches for '{arguments['pattern']}'"
                return "\n".join(matches[:100])

            # ── NEW: run_python ───────────────────────────────────────────────
            case "run_python":
                code = arguments["code"]
                result = subprocess.run(
                    [sys.executable, "-c", code],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    cwd=str(WORK_DIR),
                )
                out = result.stdout.strip()
                err = result.stderr.strip()
                parts = []
                if out:
                    parts.append(f"stdout:\n{out}")
                if err:
                    parts.append(f"stderr:\n{err}")
                if result.returncode != 0:
                    parts.append(f"exit code: {result.returncode}")
                return "\n".join(parts) if parts else "(no output)"

            # ── NEW: memory tools ─────────────────────────────────────────────
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
                if not mem:
                    return "(memory is empty)"
                return "\n".join(f"{k}: {v}" for k, v in mem.items())

            case "done":
                # Sentinel — the loop checks for this, doesn't feed it back
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
    """
    Non-streaming call.
    Returns (message_dict, prompt_token_count, generated_token_count).
    Token counts come from the top-level response fields:
      prompt_eval_count  — tokens in the full prompt sent to the model
      eval_count         — tokens the model generated
    """
    payload = {
        "model":    MODEL,
        "stream":   False,
        "think":    THINK,
        "tools":    TOOLS,
        "messages": messages,
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return (
        data["message"],
        data.get("prompt_eval_count", 0),
        data.get("eval_count", 0),
    )


def call_model_stream(messages: list) -> dict:
    """
    Streaming call — prints tokens live, returns the assembled message dict.

    How Ollama streaming works:
      - Each line is a JSON chunk: {"message": {"content": "token"}, "done": false}
      - Thinking tokens arrive first (when think=True), in chunk["message"]["thinking"]
      - Content tokens follow in chunk["message"]["content"]
      - Tool calls can appear in ANY chunk (not guaranteed to be in the final one)
        → we accumulate tool_calls across ALL chunks
      - We print tokens live while accumulating them for the return value

    The returned dict has the same shape as the non-streaming response so the
    agent loop doesn't need to know which mode is active.
    """
    payload = {
        "model":    MODEL,
        "stream":   True,
        "think":    THINK,
        "tools":    TOOLS,
        "messages": messages,
    }
    resp = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=120)
    resp.raise_for_status()

    full_thinking = ""
    full_content  = ""
    tool_calls    = []   # accumulate from ALL chunks — not just the final one
    in_thinking   = False
    in_tokens     = 0
    out_tokens    = 0

    for raw_line in resp.iter_lines():
        if not raw_line:
            continue
        chunk = json.loads(raw_line)
        msg   = chunk.get("message", {})

        # ── Accumulate tool_calls from every chunk ─────────────────────────
        if msg.get("tool_calls"):
            tool_calls.extend(msg["tool_calls"])

        # ── Thinking tokens ────────────────────────────────────────────────
        thinking_token = msg.get("thinking", "")
        if thinking_token:
            if not in_thinking:
                print("\n\033[2m── Thinking ", end="", flush=True)
                in_thinking = True
            print(f"\033[2m{thinking_token}\033[0m", end="", flush=True)
            full_thinking += thinking_token

        # ── Content tokens — buffered (panel shown after loop) ────────────
        content_token = msg.get("content", "")
        if content_token:
            if in_thinking:
                print("\033[0m")  # close dim, newline after thinking
                in_thinking = False
            full_content += content_token

        if chunk.get("done"):
            if DEBUG:
                print(f"\n[DEBUG final chunk] {json.dumps(chunk, default=str)}", flush=True)
            in_tokens  = chunk.get("prompt_eval_count", 0)
            out_tokens = chunk.get("eval_count", 0)
            break

    # Close thinking stream if it was still open
    if in_thinking:
        print()  # final newline

    return (
        {
            "role":       "assistant",
            "content":    full_content,
            "thinking":   full_thinking,
            "tool_calls": tool_calls,
        },
        in_tokens,
        out_tokens,
    )

# ─────────────────────────── Display Helpers ─────────────────────────────────

def show_thinking(text: str) -> None:
    """Used in non-streaming mode only; streaming mode prints inline."""
    if text and text.strip():
        console.print(
            Panel(text.strip(), title="[dim]Thinking[/dim]",
                  border_style="dim", style="dim italic")
        )

def show_tool_call(name: str, args: dict) -> None:
    console.print(
        Panel(
            Syntax(json.dumps(args, indent=2), "json", theme="monokai"),
            title=f"[bold yellow]Tool Call → {name}[/bold yellow]",
            border_style="yellow",
        )
    )

def show_tool_result(name: str, result: str) -> None:
    console.print(
        Panel(result, title=f"[bold green]Result ← {name}[/bold green]",
              border_style="green")
    )

def show_final_answer(text: str) -> None:
    """Always shown in the blue panel — streaming mode buffers content for this."""
    console.print(
        Panel(text.strip(), title="[bold blue]Agent[/bold blue]",
              border_style="blue")
    )

# ─────────────────────────── Agent Loop ──────────────────────────────────────

def run_agent(task: str, messages: list) -> list:
    """
    Core agentic loop — unchanged in structure from v1/v2.
    The only difference: call_model_stream vs call_model based on STREAM flag.
    """
    messages.append({"role": "user", "content": task})
    console.rule(f"[bold]Task[/bold]: {task}")

    total_in  = 0
    total_out = 0

    for step in range(1, MAX_ITER + 1):
        console.rule(f"Step {step}/{MAX_ITER}", style="dim")

        if DEBUG:
            console.print_json(json.dumps(messages, default=str))

        # ── Model call (streaming or blocking) ────────────────────────────
        msg, in_tok, out_tok = (
            call_model_stream(messages) if STREAM else call_model(messages)
        )
        total_in  += in_tok
        total_out += out_tok
        console.print(
            f"[dim]tokens → in: {in_tok:,}  out: {out_tok:,}  "
            f"(task total in: {total_in:,}  out: {total_out:,})[/dim]"
        )

        # Non-streaming: show panels. Streaming: already printed inline.
        if not STREAM:
            show_thinking(msg.get("thinking", ""))

        # ── Tool calls ────────────────────────────────────────────────────
        # Use explicit len() check: msg["tool_calls"] can be [] which is falsy
        if msg.get("tool_calls") is not None and len(msg["tool_calls"]) > 0:

            # Check for done sentinel before appending anything
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
                result = execute_tool(name, args)
                show_tool_result(name, result)
                messages.append({"role": "tool", "content": result})

        # ── Final answer ──────────────────────────────────────────────────
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
    "memory":  "Show all persisted memory facts",
    "debug":   "Toggle DEBUG (print messages list)",
    "think":   "Toggle THINK (reasoning tokens)",
    "stream":  "Toggle STREAM (live token output)",
    "convo":   "Toggle CONVERSATION_MODE (persistent within session)",
    "clear":   "Clear conversation memory",
    "persona": "Edit the system prompt",
    "quit":    "Exit",
}

def main() -> None:
    global DEBUG, THINK, STREAM, CONVERSATION_MODE

    custom_system_prompt: str | None = None

    console.print(
        Panel(
            f"[bold]Agentic AI REPL v3[/bold]\n"
            f"Model        : {MODEL}\n"
            f"Thinking     : {THINK}\n"
            f"Streaming    : {STREAM}\n"
            f"Debug        : {DEBUG}\n"
            f"Convo mode   : {CONVERSATION_MODE}\n"
            f"Workspace    : {WORK_DIR}\n"
            f"Memory file  : {MEMORY_FILE.name} "
            f"({'exists' if MEMORY_FILE.exists() else 'empty'})\n\n"
            "Commands: " + ", ".join(f"[bold]{c}[/bold]" for c in REPL_COMMANDS),
            title="Welcome",
            border_style="cyan",
        )
    )

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
            if not task_history:
                console.print("[dim]No tasks yet.[/dim]")
            else:
                for i, t in enumerate(task_history, 1):
                    console.print(f"  {i}. {t}")

        elif cmd == "memory":
            mem = load_memory()
            if not mem:
                console.print("[dim]Memory is empty.[/dim]")
            else:
                for k, v in mem.items():
                    console.print(f"  [bold]{k}[/bold]: {v}")

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
            # Let the user type a new system prompt inline
            console.print("[dim]Current system prompt:[/dim]")
            console.print(build_system_prompt(custom_system_prompt))
            console.print("\n[dim]Enter new system prompt (blank line to finish, 'reset' to restore default):[/dim]")
            lines = []
            while True:
                try:
                    line = input()
                except (EOFError, KeyboardInterrupt):
                    break
                if line.strip().lower() == "reset":
                    custom_system_prompt = None
                    console.print("[yellow]System prompt reset to default.[/yellow]")
                    break
                if line == "":
                    if lines:
                        custom_system_prompt = "\n".join(lines)
                        console.print("[yellow]System prompt updated.[/yellow]")
                    break
                lines.append(line)
            # Refresh system message in current conversation
            if conversation:
                conversation[0] = {"role": "system", "content": build_system_prompt(custom_system_prompt)}

        else:
            task_history.append(user_input)
            try:
                if CONVERSATION_MODE:
                    conversation = run_agent(user_input, conversation)
                else:
                    run_agent(user_input, fresh_messages())
            except requests.exceptions.ConnectionError:
                console.print("[red]Cannot reach Ollama. Try: ollama serve[/red]")
            except Exception as e:
                console.print(f"[red]Unexpected error: {e}[/red]")


if __name__ == "__main__":
    main()
