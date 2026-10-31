"""Long-term memory: two plain markdown files that grow as the AI is used.

memory/facts.md  - things learned from files, pages and the user (one bullet per line)
memory/rules.md  - standing instructions on how to behave, applied to every agent
memory/brand.md  - the brand guide, applied to every agent (edit by hand)
"""
import datetime as _dt
import json
import re
from pathlib import Path

from . import llm

ROOT = Path(__file__).resolve().parent.parent
FACTS = ROOT / "memory" / "facts.md"
RULES = ROOT / "memory" / "rules.md"
BRAND = ROOT / "memory" / "brand.md"
MAX_FACTS_IN_PROMPT = 60   # above this, only the most relevant + most recent facts are shown
_PREFIX = re.compile(r"^- \[[^\]]*\]\s*")

EXTRACT_SYSTEM = """You maintain the long-term memory of a small assistant that starts out knowing nothing and learns only by being used.
Return ONLY a JSON object: {"facts": [...], "rules": [...]} with short, self-contained strings.
"facts" (0-6): what should be remembered from this interaction. Be eager to learn:
- concrete facts found in files or web pages (names, numbers, dates, definitions, decisions), and which file/url they came from
- files, folders and urls the user works with
- facts about the user, their preferences, and any corrections they made
"rules" (0-3): standing instructions the user gave about how the assistant should behave from now on
  (e.g. "always ...", "never ...", "from now on ...", "I prefer ..."). Only when the user clearly said so.
Do not repeat anything already in EXISTING MEMORY or EXISTING RULES, including rephrasings of the same meaning. Do not store the assistant's own opinions,
generic knowledge, or temporary tool failures (a fetch or search that did not work).
Write plain sentences without any date or agent prefix. Return {"facts": [], "rules": []} if there is nothing new."""


# ---------- facts ----------

def _lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [ln.rstrip() for ln in path.read_text().splitlines() if ln.strip()]


def _norm(line: str) -> str:
    return " ".join(_PREFIX.sub("", line).lower().split())


def load() -> str:
    return "\n".join(_lines(FACTS))


def count() -> int:
    return len(_lines(FACTS))


def add(facts: list[str], agent: str) -> list[str]:
    """Append facts, skipping exact duplicates. Returns what was actually added."""
    known = {_norm(ln) for ln in _lines(FACTS)}
    today = _dt.date.today().isoformat()
    added = []
    FACTS.parent.mkdir(exist_ok=True)
    with FACTS.open("a") as f:
        for fact in facts:
            fact = " ".join(str(fact).split())
            if fact and _norm(fact) not in known:
                f.write(f"- [{today}, {agent}] {fact}\n")
                known.add(_norm(fact))
                added.append(fact)
    return added


def forget(text: str) -> int:
    """Remove every fact containing `text` (case-insensitive). Returns how many were removed."""
    lines = _lines(FACTS)
    keep = [ln for ln in lines if text.lower() not in ln.lower()]
    FACTS.write_text("\n".join(keep) + ("\n" if keep else ""))
    return len(lines) - len(keep)


def clear() -> None:
    for p in (FACTS, RULES):
        if p.exists():
            p.unlink()


_WORD = re.compile(r"[a-z0-9]{3,}")


def recall(task: str) -> str:
    """Facts to put in a prompt. All of them while memory is small; otherwise the most relevant + recent."""
    lines = _lines(FACTS)
    if len(lines) <= MAX_FACTS_IN_PROMPT:
        return "\n".join(lines)
    words = set(_WORD.findall(task.lower()))
    scored = sorted(range(len(lines)), key=lambda i: -len(words & set(_WORD.findall(lines[i].lower()))))
    recent = set(range(len(lines) - 10, len(lines)))
    chosen = set(scored[: MAX_FACTS_IN_PROMPT - 10]) | recent
    picked = [lines[i] for i in sorted(chosen)]
    return f"(showing {len(picked)} of {len(lines)} facts, chosen by relevance and recency)\n" + "\n".join(picked)


# ---------- rules ----------

def rules() -> str:
    return "\n".join(_lines(RULES))


def set_rules(text: str) -> None:
    RULES.parent.mkdir(exist_ok=True)
    RULES.write_text(text.strip() + "\n" if text.strip() else "")


def add_rules(new: list[str]) -> list[str]:
    known = {_norm(ln) for ln in _lines(RULES)}
    added = []
    RULES.parent.mkdir(exist_ok=True)
    with RULES.open("a") as f:
        for r in new:
            r = " ".join(str(r).split())
            if r and _norm(r) not in known:
                f.write(f"- {r}\n")
                known.add(_norm(r))
                added.append(r)
    return added


def brand() -> str:
    return BRAND.read_text().strip() if BRAND.exists() else ""


# ---------- learning ----------

def learn(agent: str, task: str, answer: str, tool_log: str = "", allow_rules: bool = True) -> dict:
    """Ask the model what was worth remembering from this exchange; store it. Returns what was added."""
    prompt = (
        f"EXISTING MEMORY:\n{load() or '(empty)'}\n\n"
        f"EXISTING RULES:\n{rules() or '(none)'}\n\n"
        f"USER TASK:\n{task}\n\n"
        f"INFORMATION GATHERED BY SKILLS:\n{tool_log[:6000] or '(none)'}\n\n"
        f"ASSISTANT ANSWER:\n{answer[:4000]}"
    )
    raw = llm.ask(EXTRACT_SYSTEM, prompt)
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {"facts": [], "rules": []}
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"facts": [], "rules": []}
    if isinstance(data, list):  # tolerate the old bare-list shape
        data = {"facts": data}
    return {
        "facts": add(list(data.get("facts") or []), agent),
        "rules": add_rules(list(data.get("rules") or [])) if allow_rules else [],
    }
