"""Minimal MCP (Model Context Protocol) over stdio, stdlib only.

Client side: `call(server, tool, args)` starts the server listed in mcp.json on first use and keeps it running.
Server side: `serve(name, tools)` turns a dict of Python functions into an MCP stdio server; the connectors in
personal_ai/connectors/ use it, and the same servers can be registered with any MCP client (e.g. `claude mcp add`).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "mcp.json"
PROTOCOL = "2025-03-26"


class MCPError(RuntimeError):
    pass


# ---------- client ----------

class Client:
    def __init__(self, name: str, command: list[str], env: dict | None = None):
        self.name, self.command, self.env = name, command, env or {}
        self.proc = None
        self._id = 0

    def start(self) -> None:
        env = {**os.environ, **{k: os.path.expandvars(str(v)) for k, v in self.env.items()}}
        self.proc = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, cwd=ROOT, env=env)
        self._request("initialize", {"protocolVersion": PROTOCOL, "capabilities": {},
                                     "clientInfo": {"name": "personal-ai", "version": "0.1"}})
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _send(self, msg: dict) -> None:
        try:
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            raise MCPError(f"MCP server '{self.name}' is not running")

    def _request(self, method: str, params: dict) -> dict:
        self._id += 1
        self._send({"jsonrpc": "2.0", "id": self._id, "method": method, "params": params})
        while True:
            line = self.proc.stdout.readline()
            if not line:
                self.proc = None
                raise MCPError(f"MCP server '{self.name}' exited unexpectedly")
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("id") == self._id:
                if "error" in msg:
                    raise MCPError(msg["error"].get("message", str(msg["error"])))
                return msg.get("result") or {}

    def list_tools(self) -> list[dict]:
        if not self.proc:
            self.start()
        return self._request("tools/list", {}).get("tools", [])

    def call(self, tool: str, args: dict) -> str:
        if not self.proc:
            self.start()
        res = self._request("tools/call", {"name": tool, "arguments": args})
        text = "\n".join(c.get("text", "") for c in res.get("content", []) if c.get("type") == "text")
        if res.get("isError"):
            raise MCPError(text or "tool error")
        return text

    def close(self) -> None:
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
            self.proc = None


_clients: dict[str, Client] = {}


def config() -> dict:
    if not CONFIG.exists():
        return {}
    return json.loads(CONFIG.read_text()).get("mcpServers", {})


def client(server: str) -> Client:
    if server not in _clients:
        cfg = config().get(server)
        if not cfg:
            raise MCPError(f"no MCP server '{server}' in {CONFIG.name}")
        cmd = [sys.executable if c == "python" else c for c in cfg["command"]]
        _clients[server] = Client(server, cmd, cfg.get("env"))
    return _clients[server]


def call(server: str, tool: str, args: dict) -> str:
    return client(server).call(tool, args)


# ---------- server ----------

def serve(name: str, tools: dict) -> None:
    """tools: {tool_name: {"fn": callable, "desc": str, "args": {arg: description}}}. Blocks, serving stdin/stdout."""
    def listing():
        return [{"name": t, "description": s["desc"],
                 "inputSchema": {"type": "object", "properties": {a: {"type": "string", "description": d} for a, d in s["args"].items()},
                                 "required": [a for a in s["args"] if not s.get("optional", {}).get(a)]}}
                for t, s in tools.items()]

    def reply(id_, result=None, error=None):
        msg = {"jsonrpc": "2.0", "id": id_}
        msg["error" if error else "result"] = error if error else result
        sys.stdout.write(json.dumps(msg) + "\n")
        sys.stdout.flush()

    for line in sys.stdin:
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        method, id_, params = msg.get("method"), msg.get("id"), msg.get("params") or {}
        if id_ is None:
            continue  # notification
        if method == "initialize":
            reply(id_, {"protocolVersion": PROTOCOL, "capabilities": {"tools": {}}, "serverInfo": {"name": name, "version": "0.1"}})
        elif method == "tools/list":
            reply(id_, {"tools": listing()})
        elif method == "tools/call":
            t = tools.get(params.get("name"))
            if not t:
                reply(id_, error={"code": -32602, "message": f"unknown tool {params.get('name')!r}"})
                continue
            try:
                text = t["fn"](**(params.get("arguments") or {}))
                reply(id_, {"content": [{"type": "text", "text": str(text)}], "isError": False})
            except Exception as e:
                reply(id_, {"content": [{"type": "text", "text": f"{type(e).__name__}: {e}"}], "isError": True})
        elif method == "ping":
            reply(id_, {})
        else:
            reply(id_, error={"code": -32601, "message": f"method not found: {method}"})
