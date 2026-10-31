"""Skills are plain Python functions an agent may call. Each returns a string.
Add a function here and register it in SKILLS to make it available to agents."""
import gzip
import html
import json
import re
import urllib.parse
import urllib.request
import zlib
from pathlib import Path

MAX_CHARS = 12000
ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT / "workspace"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
}


def _clip(text: str) -> str:
    text = text.strip()
    if len(text) > MAX_CHARS:
        return text[:MAX_CHARS] + f"\n...[truncated, {len(text)} chars total]"
    return text or "(empty)"


def _path(path: str) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (ROOT / p if (ROOT / p).exists() else WORKSPACE / p)


# ---------- reading ----------

def read_txt(path: str) -> str:
    return _clip(_path(path).read_text(errors="replace"))


def read_pdf(path: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(_path(path))
    pages = [f"--- page {i + 1} ---\n{(p.extract_text() or '').strip()}" for i, p in enumerate(reader.pages)]
    return _clip("\n".join(pages))


def list_files(path: str = ".") -> str:
    p = _path(path)
    if not p.is_dir():
        return f"error: {p} is not a directory"
    entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
    return _clip("\n".join(f"{'  ' if e.is_file() else '[dir] '}{e.name}" for e in entries if not e.name.startswith(".")))


# ---------- writing ----------

def write_file(path: str, content: str) -> str:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = WORKSPACE / p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"wrote {len(content)} chars to {p}"


# ---------- web ----------

def _get(url: str) -> str:
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read()
        enc = r.headers.get("Content-Encoding", "")
        charset = r.headers.get_content_charset() or "utf-8"
    if "gzip" in enc:
        raw = gzip.decompress(raw)
    elif "deflate" in enc:
        raw = zlib.decompress(raw, -zlib.MAX_WBITS)
    return raw.decode(charset, errors="replace")


def _strip_html(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    raw = html.unescape(raw)
    return re.sub(r"\s+", " ", raw)


def fetch_url(url: str) -> str:
    return _clip(_strip_html(_get(url)))


def web_search(query: str) -> str:
    raw = _get("https://html.duckduckgo.com/html/?q=" + urllib.parse.quote_plus(query))
    results = []
    for m in re.finditer(r'class="result__a" href="([^"]+)"[^>]*>(.*?)</a>.*?class="result__snippet"[^>]*>(.*?)</a>', raw, re.S):
        href, title, snippet = m.groups()
        q = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
        href = q.get("uddg", [href])[0]
        results.append(f"- {_strip_html(title)}\n  {href}\n  {_strip_html(snippet)}")
        if len(results) >= 8:
            break
    return "\n".join(results) or "(no results)"


# ---------- hybrid document search ----------

def search_docs(query: str, k: int = 5) -> str:
    from . import index
    return _clip(index.format_hits(index.search(query, k=int(k))))


# ---------- MCP connectors (email, business database) ----------

def _mcp(server: str, tool: str, **args) -> str:
    from . import mcp
    try:
        return _clip(mcp.call(server, tool, {k: str(v) for k, v in args.items() if v is not None}))
    except mcp.MCPError as e:
        return f"error: {e}"


def email_search(query: str, limit: int = 10) -> str:
    return _mcp("email", "search", query=query, limit=limit)


def email_read(uid: str) -> str:
    return _mcp("email", "read", uid=uid)


def email_send(to: str, subject: str, body: str) -> str:
    return _mcp("email", "send", to=to, subject=subject, body=body)


def db_schema() -> str:
    return _mcp("db", "schema")


def db_query(sql: str) -> str:
    return _mcp("db", "query", sql=sql)


# ---------- delegation (used by the planner agent) ----------

_DEPTH = {"n": 0}


def delegate(agent: str, task: str) -> str:
    """Run another agent on a sub-task and return its answer."""
    from . import agents, runner
    if _DEPTH["n"] >= 1:
        return "error: nested delegation is not allowed"
    if agent not in agents.load_all():
        return f"error: no agent '{agent}'. Use list_agents to see what exists."
    if agents.get(agent)["skills"] and "delegate" in agents.get(agent)["skills"]:
        return f"error: '{agent}' is itself a planner; delegate to a worker agent instead"
    _DEPTH["n"] += 1
    try:
        return _clip(runner.run(agent, task, verbose=True, learn=True))
    finally:
        _DEPTH["n"] -= 1


# ---------- self-management (used by the trainer agent) ----------

def list_agents() -> str:
    from . import agents
    return "\n".join(f"- {a['name']}: {a['description']} (skills: {', '.join(a['skills']) or 'none'})"
                     for a in agents.load_all().values())


def get_agent(name: str) -> str:
    from . import agents
    return json.dumps(agents.get(name), indent=2, ensure_ascii=False)


def set_agent_instructions(name: str, instructions: str) -> str:
    from . import agents
    path = agents.update(name, instructions=instructions, by="trainer")
    return f"updated instructions of '{name}' (previous version backed up). File: {path}"


def create_agent(name: str, description: str, skills: str, instructions: str) -> str:
    from . import agents
    names = [s.strip() for s in skills.split(",") if s.strip()]
    return f"created {agents.create(name, description, names, instructions)}"


def get_rules() -> str:
    from . import memory
    return memory.rules() or "(no rules yet)"


def set_rules(content: str) -> str:
    from . import memory
    memory.set_rules(content)
    return "rules replaced:\n" + (memory.rules() or "(empty)")


def read_memory() -> str:
    from . import memory
    return _clip(memory.load() or "(memory is empty)")


def remember(fact: str) -> str:
    from . import memory
    added = memory.add([fact], "trainer")
    return "stored" if added else "already known"


def forget(text: str) -> str:
    from . import memory
    return f"removed {memory.forget(text)} fact(s) containing {text!r}"


SKILLS = {
    "read_txt": {"fn": read_txt, "args": {"path": "file path"}, "desc": "Read a plain text / markdown / code file."},
    "read_pdf": {"fn": read_pdf, "args": {"path": "file path"}, "desc": "Extract the text of a PDF file."},
    "list_files": {"fn": list_files, "args": {"path": "directory (default: project root)"}, "desc": "List files in a directory."},
    "write_file": {"fn": write_file, "args": {"path": "file path (relative paths go to workspace/)", "content": "text"}, "desc": "Write text to a file."},
    "fetch_url": {"fn": fetch_url, "args": {"url": "http(s) url"}, "desc": "Download a web page and return its text."},
    "web_search": {"fn": web_search, "args": {"query": "search terms"}, "desc": "Search the web, returns titles, links and snippets."},
    "search_docs": {"fn": search_docs, "args": {"query": "what to look for, in plain words", "k": "how many passages (default 5)"}, "desc": "Search the indexed business documents (hybrid lexical + semantic search, reranked). Returns the best passages with their source files."},
    "email_search": {"fn": email_search, "args": {"query": "search terms (Gmail syntax ok: from:x subject:y newer_than:7d)", "limit": "max results (default 10)"}, "desc": "Search the user's mailbox; returns uid, date, sender, subject per email."},
    "email_read": {"fn": email_read, "args": {"uid": "email uid from email_search"}, "desc": "Read one email in full."},
    "email_send": {"fn": email_send, "args": {"to": "recipient address", "subject": "subject line", "body": "plain-text body"}, "approval": True, "desc": "Send an email from the user's account. It is NOT sent immediately: it goes to the approval queue and the user must approve it."},
    "db_schema": {"fn": db_schema, "args": {}, "desc": "List the tables and columns of the business database."},
    "db_query": {"fn": db_query, "args": {"sql": "one SELECT statement (SQLite syntax)"}, "desc": "Run a read-only SELECT on the business database. Call db_schema first to see the tables."},
    "delegate": {"fn": delegate, "args": {"agent": "agent name", "task": "complete, self-contained sub-task including any data it needs"}, "desc": "Run another agent on a sub-task and get its answer back."},
    "list_agents": {"fn": list_agents, "args": {}, "desc": "List all agents with their descriptions and skills."},
    "get_agent": {"fn": get_agent, "args": {"name": "agent name"}, "desc": "Show an agent's full definition including its current instructions (system prompt)."},
    "set_agent_instructions": {"fn": set_agent_instructions, "args": {"name": "agent name", "instructions": "the complete new instructions"}, "desc": "Replace an agent's instructions (its system prompt). The old version is backed up."},
    "create_agent": {"fn": create_agent, "args": {"name": "short_name", "description": "one line", "skills": "comma separated skill names", "instructions": "full instructions"}, "desc": "Create a brand new agent."},
    "get_rules": {"fn": get_rules, "args": {}, "desc": "Show the standing rules that apply to every agent."},
    "set_rules": {"fn": set_rules, "args": {"content": "the complete new rules, one '- ' bullet per line"}, "desc": "Replace the standing rules that apply to every agent."},
    "read_memory": {"fn": read_memory, "args": {}, "desc": "Show everything in long-term memory."},
    "remember": {"fn": remember, "args": {"fact": "one short sentence"}, "desc": "Store a fact in long-term memory."},
    "forget": {"fn": forget, "args": {"text": "text to match"}, "desc": "Delete every memory fact containing this text."},
}


def run(name: str, args: dict, gated: bool = True) -> str:
    """Run a skill. Skills marked approval=True are refused here when gated; the runner queues them instead."""
    skill = SKILLS.get(name)
    if not skill:
        return f"error: unknown skill '{name}'"
    if not isinstance(args, dict):
        return f"error: args for {name} must be an object with keys {list(skill['args'])}"
    if gated and skill.get("approval"):
        return f"error: {name} needs user approval; it can only run from the approval queue"
    try:
        return skill["fn"](**args)
    except TypeError as e:
        return f"error: bad arguments for {name}: {e}. Expected {list(skill['args'])}"
    except Exception as e:  # skill failures are reported back to the model, not raised
        return f"error: {type(e).__name__}: {e}"
