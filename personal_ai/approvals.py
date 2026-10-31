"""Approval queue: skills that change the outside world (sending email) are queued instead of run.
`pai approvals` lists them, `pai approve <id>` runs one, `pai reject <id>` drops it. File: approvals/queue.jsonl."""
import datetime as _dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUEUE = ROOT / "approvals" / "queue.jsonl"


def _load() -> list[dict]:
    if not QUEUE.exists():
        return []
    return [json.loads(ln) for ln in QUEUE.read_text().splitlines() if ln.strip()]


def _save(items: list[dict]) -> None:
    QUEUE.parent.mkdir(exist_ok=True)
    QUEUE.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items))


def enqueue(agent: str, skill: str, args: dict, task: str = "") -> int:
    items = _load()
    item = {"id": (max((i["id"] for i in items), default=0) + 1), "time": _dt.datetime.now().isoformat(timespec="seconds"),
            "agent": agent, "task": task[:300], "skill": skill, "args": args, "status": "pending"}
    items.append(item)
    _save(items)
    return item["id"]


def pending() -> list[dict]:
    return [i for i in _load() if i["status"] == "pending"]


def get(item_id: int) -> dict:
    for i in _load():
        if i["id"] == item_id:
            return i
    raise KeyError(f"no approval #{item_id}")


def _set(item_id: int, **fields) -> dict:
    items = _load()
    for i in items:
        if i["id"] == item_id:
            i.update(fields)
            _save(items)
            return i
    raise KeyError(f"no approval #{item_id}")


def approve(item_id: int) -> str:
    """Run the queued skill for real. Returns the skill's result."""
    from . import skills
    item = get(item_id)
    if item["status"] != "pending":
        raise ValueError(f"approval #{item_id} is already {item['status']}")
    result = skills.run(item["skill"], item["args"], gated=False)
    _set(item_id, status="done" if not result.startswith("error:") else "failed",
         result=result[:2000], decided=_dt.datetime.now().isoformat(timespec="seconds"))
    return result


def reject(item_id: int) -> None:
    item = get(item_id)
    if item["status"] != "pending":
        raise ValueError(f"approval #{item_id} is already {item['status']}")
    _set(item_id, status="rejected", decided=_dt.datetime.now().isoformat(timespec="seconds"))


def describe(item: dict) -> str:
    args = ", ".join(f"{k}={str(v)[:80]!r}" for k, v in item["args"].items())
    line = f"#{item['id']}  {item['status']:<8} {item['time']}  {item['agent']}: {item['skill']}({args})"
    if item.get("result"):
        line += f"\n      -> {item['result'][:200]}"
    return line
