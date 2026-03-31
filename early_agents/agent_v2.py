"""
agent_v2.py — Extended agentic loop with new features on top of agent.py concepts.

New in v2:
  - DEBUG flag        : clean toggle for messages inspection
  - Session logging   : saves full messages list to logs/ after each task
  - search tool       : grep-like text search within workspace files
  - history command   : REPL command to review past tasks this session
  - conversation mode : keep messages across tasks (persistent memory within session)
"""

import datetime
import fnmatch
import json
import pathlib
import readline  # noqa: F401
import shutil

import requests
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

# ─────────────────────────── Constants ───────────────────────────────────────

OLLAMA_URL        = "http://localhost:11434/api/chat"
MODEL             = "qwen3:8b"
WORK_DIR          = pathlib.Path(__file__).parent.resolve()
LOGS_DIR          = WORK_DIR / "logs"
MAX_ITER          = 10
THINK             = True
DEBUG             = False   # True → print full messages list before each model call
CONVERSATION_MODE = False   # True → messages persist across tasks (agent remembers)

console = Console()

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
    # ── NEW in v2 ─────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search for a text pattern inside files in the workspace. "
                "Returns matching lines with their file path and line number. "
                "Optionally filter by file glob pattern (e.g. '*.py', '*.txt')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern":  {"type": "string", "description": "Text to search for (case-insensitive substring)."},
                    "path":     {"type": "string", "description": "Relative directory to search in. Defaults to '.'."},
                    "glob":     {"type": "string", "description": "File glob filter, e.g. '*.py'. Defaults to '*'."},
                },
                "required": ["pattern"],
            },
        },
    },
]

# ─────────────────────────── Safety ──────────────────────────────────────────

def safe_path(rel: str) -> pathlib.Path:
    resolved = (WORK_DIR / rel).resolve()
    if not resolved.is_relative_to(WORK_DIR):
        raise ValueError(
            f"Path traversal blocked: '{rel}' resolves outside workspace."
        )
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

            # ── NEW: search ───────────────────────────────────────────────────
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
                return "\n".join(matches[:100])  # cap at 100 lines

            case _:
                return f"ERROR: Unknown tool '{name}'"

    except Exception as e:
        return f"ERROR: {e}"

# ─────────────────────────── Session Logging ─────────────────────────────────

def save_session_log(task: str, messages: list) -> None:
    """
    Save the full messages list to logs/YYYY-MM-DD_HH-MM-SS.json.
    Great for reviewing what the agent actually sent/received.
    """
    LOGS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path  = LOGS_DIR / f"{timestamp}.json"
    payload   = {"task": task, "messages": messages}
    log_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    console.print(f"[dim]Session log saved → logs/{log_path.name}[/dim]")

# ─────────────────────────── Ollama API ──────────────────────────────────────

def call_model(messages: list) -> dict:
    payload = {
        "model":    MODEL,
        "stream":   False,
        "think":    THINK,
        "tools":    TOOLS,
        "messages": messages,
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["message"]

# ─────────────────────────── Display Helpers ─────────────────────────────────

def show_thinking(text: str) -> None:
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
    console.print(
        Panel(text.strip(), title="[bold blue]Agent[/bold blue]",
              border_style="blue")
    )

# ─────────────────────────── System Prompt ───────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful file-system agent. You operate exclusively inside a "
    "sandboxed workspace directory. You have tools to list, read, write, search, "
    "move, copy, and delete files within that workspace. Complete the user's task "
    "step by step. When you are done, summarise what you did in plain language."
)

# ─────────────────────────── Agent Loop ──────────────────────────────────────

def run_agent(task: str, messages: list) -> list:
    """
    Run one task through the agent loop.

    `messages` is passed in (and returned) so the caller controls
    whether memory persists across tasks (CONVERSATION_MODE).
    """
    messages.append({"role": "user", "content": task})
    console.rule(f"[bold]Task[/bold]: {task}")

    for step in range(1, MAX_ITER + 1):
        console.rule(f"Step {step}/{MAX_ITER}", style="dim")

        # DEBUG: print full messages list before each model call
        if DEBUG:
            console.print_json(json.dumps(messages, default=str))

        msg = call_model(messages)
        show_thinking(msg.get("thinking", ""))

        if msg.get("tool_calls"):
            messages.append(msg)
            for tc in msg["tool_calls"]:
                name   = tc["function"]["name"]
                args   = tc["function"]["arguments"]
                show_tool_call(name, args)
                result = execute_tool(name, args)
                show_tool_result(name, result)
                messages.append({"role": "tool", "content": result})

        elif msg.get("content", "").strip():
            messages.append(msg)
            show_final_answer(msg["content"])
            save_session_log(task, messages)
            return messages

        else:
            console.print("[red]Empty response from model — stopping.[/red]")
            return messages

    console.print(f"[red]Reached max iterations ({MAX_ITER}). Stopping.[/red]")
    save_session_log(task, messages)
    return messages

# ─────────────────────────── REPL ────────────────────────────────────────────

REPL_COMMANDS = {
    "help":    "Show this help",
    "history": "List tasks attempted this session",
    "debug":   "Toggle DEBUG flag (print messages list each step)",
    "think":   "Toggle THINK flag (reasoning tokens on/off)",
    "convo":   "Toggle CONVERSATION_MODE (persistent memory across tasks)",
    "clear":   "Clear conversation memory (only relevant in convo mode)",
    "quit":    "Exit",
}

def main() -> None:
    global DEBUG, THINK, CONVERSATION_MODE

    console.print(
        Panel(
            f"[bold]Agentic AI REPL v2[/bold]\n"
            f"Model        : {MODEL}\n"
            f"Thinking     : {THINK}\n"
            f"Debug        : {DEBUG}\n"
            f"Convo mode   : {CONVERSATION_MODE}\n"
            f"Workspace    : {WORK_DIR}\n"
            f"Max steps    : {MAX_ITER}\n\n"
            "Type a task, or a command: "
            + ", ".join(f"[bold]{c}[/bold]" for c in REPL_COMMANDS),
            title="Welcome",
            border_style="cyan",
        )
    )

    task_history: list[str] = []
    # Persistent memory: seeded once with system prompt
    conversation: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input("\n[You] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye.[/dim]")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        # ── REPL commands ──────────────────────────────────────────────────
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

        elif cmd == "debug":
            DEBUG = not DEBUG
            console.print(f"[yellow]DEBUG = {DEBUG}[/yellow]")

        elif cmd == "think":
            THINK = not THINK
            console.print(f"[yellow]THINK = {THINK}[/yellow]")

        elif cmd == "convo":
            CONVERSATION_MODE = not CONVERSATION_MODE
            console.print(f"[yellow]CONVERSATION_MODE = {CONVERSATION_MODE}[/yellow]")

        elif cmd == "clear":
            conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
            console.print("[yellow]Conversation memory cleared.[/yellow]")

        # ── Task ───────────────────────────────────────────────────────────
        else:
            task_history.append(user_input)
            try:
                if CONVERSATION_MODE:
                    # Messages persist: agent remembers previous tasks
                    conversation = run_agent(user_input, conversation)
                else:
                    # Fresh context each task
                    fresh = [{"role": "system", "content": SYSTEM_PROMPT}]
                    run_agent(user_input, fresh)
            except requests.exceptions.ConnectionError:
                console.print(
                    "[red]Cannot reach Ollama. Is it running? Try: ollama serve[/red]"
                )
            except Exception as e:
                console.print(f"[red]Unexpected error: {e}[/red]")


if __name__ == "__main__":
    main()
