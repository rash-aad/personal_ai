"""Thin wrapper around the `claude` CLI. This is the only place the model is called."""
import json
import os
import subprocess
import time


class LLMError(RuntimeError):
    pass


def ask(system: str, prompt: str, timeout: int = 300, retries: int = 2) -> str:
    system, prompt = system.replace("\x00", ""), prompt.replace("\x00", "")  # argv cannot carry NUL bytes
    cmd = [
        "claude", "-p",
        "--tools", "",              # no built-in tools: the AI only gets our skills
        "--setting-sources", "",    # ignore CLAUDE.md / user settings
        "--output-format", "json",
        "--system-prompt", system,
    ]
    model = os.environ.get("PAI_MODEL")
    if model:
        cmd += ["--model", model]
    cmd.append(prompt)
    last = None
    for attempt in range(retries + 1):
        try:
            return _call(cmd, timeout)
        except LLMError as e:
            last = e
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
    raise last


def _call(cmd: list[str], timeout: int) -> str:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        raise LLMError("`claude` CLI not found on PATH. Install Claude Code first.")
    except subprocess.TimeoutExpired:
        raise LLMError("claude CLI timed out")
    if proc.returncode != 0:
        raise LLMError(f"claude CLI failed: {proc.stderr.strip() or proc.stdout.strip()}")
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return proc.stdout.strip()
    if data.get("is_error"):
        raise LLMError(data.get("result", "unknown error"))
    return (data.get("result") or "").strip()
