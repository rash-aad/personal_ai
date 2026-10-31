"""Agents are JSON files in agents/. Each has a name, description, skills and instructions.
Every change to an agent's instructions is backed up under agents/history/ so it can be reverted."""
import datetime as _dt
import json
import re
from pathlib import Path

from . import llm

DIR = Path(__file__).resolve().parent.parent / "agents"
HISTORY = DIR / "history"
_NAME = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


def _clean_name(name: str) -> str:
    name = name.strip().lower().replace("-", "_").replace(" ", "_")
    if not _NAME.match(name):
        raise ValueError(f"bad agent name '{name}': use letters, digits and underscores")
    return name


def load_all() -> dict[str, dict]:
    agents = {}
    for p in sorted(DIR.glob("*.json")):
        try:
            a = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(f"{p} is not valid JSON: {e}")
        a.setdefault("skills", [])
        a.setdefault("instructions", "")
        a.setdefault("description", "")
        agents[a["name"]] = a
    return agents


def get(name: str) -> dict:
    agents = load_all()
    if name not in agents:
        raise KeyError(f"no agent '{name}'. Available: {', '.join(agents) or 'none'}")
    return agents[name]


def _save(agent: dict) -> Path:
    DIR.mkdir(exist_ok=True)
    path = DIR / f"{agent['name']}.json"
    path.write_text(json.dumps(agent, indent=2, ensure_ascii=False) + "\n")
    return path


def _validate_skills(skill_names: list[str]) -> None:
    from . import skills  # local import: skills imports this module too
    unknown = [s for s in skill_names if s not in skills.SKILLS]
    if unknown:
        raise ValueError(f"unknown skills: {unknown}. Available: {list(skills.SKILLS)}")


def create(name: str, description: str, skill_names: list[str], instructions: str = "") -> Path:
    name = _clean_name(name)
    _validate_skills(skill_names)
    if not instructions:
        instructions = llm.ask(
            "You write concise system instructions (3-6 sentences, plain text) for a small assistant agent. "
            "Describe how it should behave and when to use its skills. Do not mention that you are writing instructions, "
            "and never mention email addresses, directory paths or dates.",
            f"Agent name: {name}\nPurpose: {description}\nSkills it can use: {', '.join(skill_names) or 'none'}",
        )
    return _save({"name": name, "description": description, "skills": skill_names, "instructions": instructions})


def update(name: str, instructions: str | None = None, description: str | None = None,
           skill_names: list[str] | None = None, by: str = "user") -> Path:
    """Change an agent. The previous version is backed up to agents/history/."""
    agent = get(name)
    HISTORY.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    (HISTORY / f"{name}-{stamp}.json").write_text(json.dumps(agent, indent=2, ensure_ascii=False) + "\n")
    if instructions is not None:
        agent["instructions"] = instructions.strip()
    if description is not None:
        agent["description"] = description.strip()
    if skill_names is not None:
        _validate_skills(skill_names)
        agent["skills"] = skill_names
    agent["updated"] = f"{stamp} by {by}"
    return _save(agent)


def history(name: str) -> list[Path]:
    return sorted(HISTORY.glob(f"{name}-*.json")) if HISTORY.exists() else []


def revert(name: str) -> Path:
    """Restore the most recent backup of an agent."""
    backups = history(name)
    if not backups:
        raise ValueError(f"no history for agent '{name}'")
    last = backups[-1]
    agent = json.loads(last.read_text())
    last.unlink()
    return _save(agent)


def delete(name: str) -> None:
    path = DIR / f"{get(name)['name']}.json"
    path.unlink()
