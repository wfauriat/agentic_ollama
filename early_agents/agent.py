"""
agent.py — A minimal educational agentic AI loop.

Architecture:
  User task → Ollama (qwen3:8b) → tool calls → filesystem ops → observe → repeat → answer

The agent uses Ollama's native tool-calling API with thinking tokens enabled.
Every key concept is labelled with a comment so you can follow along.
"""

import json
import pathlib
import readline  # noqa: F401 — enables arrow-key history in the REPL input
import shutil

import requests
from rich import print as rprint  # noqa: F401
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

# ─────────────────────────── Constants ───────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL      = "qwen3:8b"
WORK_DIR   = pathlib.Path(__file__).parent.resolve()  # absolute sandbox root
MAX_ITER   = 10    # hard cap on agent steps per task (safety net)
THINK      = True  # set False to skip reasoning chain and get faster responses

console = Console()

# ─────────────────────────── Tool Definitions ────────────────────────────────
#
# This list is sent to the model on every API call. The model reads the
# descriptions and schemas to decide which tool to invoke and with what args.

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ls",
            "description": "List files and directories at a path inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to list. Defaults to '.' (workspace root).",
                    }
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
                    "path": {"type": "string", "description": "Relative path of the directory to create."}
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
]

# ─────────────────────────── Safety ──────────────────────────────────────────

def safe_path(rel: str) -> pathlib.Path:
    """
    Resolve a relative path against WORK_DIR and raise if it escapes.

    This is the ONLY place path resolution happens — every tool calls this.
    Defends against: '../' traversal, absolute paths, symlink escapes.
    """
    resolved = (WORK_DIR / rel).resolve()
    if not resolved.is_relative_to(WORK_DIR):
        raise ValueError(
            f"Path traversal blocked: '{rel}' resolves outside workspace ({WORK_DIR})."
        )
    return resolved


# ─────────────────────────── Tool Executor ───────────────────────────────────

def execute_tool(name: str, arguments: dict) -> str:
    """
    Dispatch a tool call to the matching filesystem operation.

    Always returns a string — including error messages. This is deliberate:
    the model can read the error and decide what to do next (retry, report, etc.).
    """
    try:
        match name:
            case "ls":
                p = safe_path(arguments.get("path", "."))
                entries = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name))
                lines = [
                    f"{'DIR ' if e.is_dir() else 'FILE'} {e.name}"
                    for e in entries
                ]
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
                    p.rmdir()  # only empty dirs — no recursive delete by design
                else:
                    p.unlink()
                return f"Deleted: {p.relative_to(WORK_DIR)}"

            case _:
                return f"ERROR: Unknown tool '{name}'"

    except Exception as e:
        return f"ERROR: {e}"


# ─────────────────────────── Ollama API ──────────────────────────────────────

def call_model(messages: list) -> dict:
    """
    Send the full conversation history to Ollama and return the model's message.

    The response message may contain:
      - message["thinking"]   — the model's reasoning chain (when THINK=True)
      - message["tool_calls"] — list of tools the model wants to invoke
      - message["content"]    — final text reply (signals task completion)
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
    return resp.json()["message"]


# ─────────────────────────── Display Helpers ─────────────────────────────────

def show_thinking(text: str) -> None:
    """Model's internal reasoning — shown dim so it doesn't dominate the output."""
    if text and text.strip():
        console.print(
            Panel(
                text.strip(),
                title="[dim]Thinking[/dim]",
                border_style="dim",
                style="dim italic",
            )
        )


def show_tool_call(name: str, args: dict) -> None:
    """The action the model decided to take, with syntax-highlighted JSON args."""
    console.print(
        Panel(
            Syntax(json.dumps(args, indent=2), "json", theme="monokai"),
            title=f"[bold yellow]Tool Call → {name}[/bold yellow]",
            border_style="yellow",
        )
    )


def show_tool_result(name: str, result: str) -> None:
    """What the filesystem returned after the tool ran."""
    console.print(
        Panel(
            result,
            title=f"[bold green]Result ← {name}[/bold green]",
            border_style="green",
        )
    )


def show_final_answer(text: str) -> None:
    """The agent's concluding message to the user."""
    console.print(
        Panel(
            text.strip(),
            title="[bold blue]Agent[/bold blue]",
            border_style="blue",
        )
    )


# ─────────────────────────── System Prompt ───────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful file-system agent. You operate exclusively inside a "
    "sandboxed workspace directory. You have tools to list, read, write, move, "
    "copy, and delete files within that workspace. Complete the user's task step "
    "by step. When you are done, summarise what you did in plain language."
)

# ─────────────────────────── Agent Loop ──────────────────────────────────────

def run_agent(task: str) -> None:
    """
    The core agentic loop.

    Concept: messages is the agent's working memory. It grows each step.
    The loop stops naturally when the model replies with text (no tool calls).

    Step anatomy:
      1. Call the model with full conversation history
      2. Show the model's reasoning (thinking tokens)
      3a. If tool_calls → execute each, append results, loop again
      3b. If content text → show answer, done
    """
    # Seed the conversation
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": task},
    ]

    console.rule(f"[bold]Task[/bold]: {task}")

    for step in range(1, MAX_ITER + 1):
        console.rule(f"Step {step}/{MAX_ITER}", style="dim")

        # DEBUG: uncomment to inspect the full messages list before each call
        console.print_json(json.dumps(messages, default=str))

        # ── 1. Ask the model what to do next ──────────────────────────────
        msg = call_model(messages)

        # ── 2. Show reasoning (thinking tokens, only present when THINK=True)
        show_thinking(msg.get("thinking", ""))

        # ── 3a. Model wants to call one or more tools ──────────────────────
        if msg.get("tool_calls"):
            # Append the assistant turn (with its tool_calls) to history
            messages.append(msg)

            for tc in msg["tool_calls"]:
                name   = tc["function"]["name"]
                args   = tc["function"]["arguments"]  # already a dict from Ollama

                show_tool_call(name, args)
                result = execute_tool(name, args)
                show_tool_result(name, result)

                # Feed the result back as a "tool" role message
                messages.append({"role": "tool", "content": result})

            # Continue the loop — model will see the results and decide next step

        # ── 3b. Model produced a text reply → task complete ────────────────
        elif msg.get("content", "").strip():
            messages.append(msg)
            show_final_answer(msg["content"])
            return

        # ── Edge case: empty response (shouldn't happen, but guard anyway) ─
        else:
            console.print("[red]Empty response from model — stopping.[/red]")
            return

    console.print(f"[red]Reached max iterations ({MAX_ITER}). Stopping.[/red]")


# ─────────────────────────── REPL Entry Point ────────────────────────────────

def main() -> None:
    console.print(
        Panel(
            f"[bold]Agentic AI REPL[/bold]\n"
            f"Model     : {MODEL}\n"
            f"Thinking  : {THINK}  (toggle: edit THINK at top of file)\n"
            f"Workspace : {WORK_DIR}\n"
            f"Max steps : {MAX_ITER}\n\n"
            "Type a task and press Enter.\n"
            "Type [bold]quit[/bold] or [bold]exit[/bold] to stop.",
            title="Welcome",
            border_style="cyan",
        )
    )

    while True:
        try:
            task = input("\n[You] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye.[/dim]")
            break

        if not task:
            continue
        if task.lower() in ("quit", "exit", "q"):
            console.print("[dim]Bye.[/dim]")
            break

        try:
            run_agent(task)
        except requests.exceptions.ConnectionError:
            console.print(
                "[red]Cannot reach Ollama. Is it running? Try: ollama serve[/red]"
            )
        except Exception as e:
            console.print(f"[red]Unexpected error: {e}[/red]")


if __name__ == "__main__":
    main()
