"""Runs one agent on one task: a small think -> skill -> observe loop, then memory update and a log entry."""
import datetime as _dt
import json
import re
from pathlib import Path

from . import agents, approvals, llm, memory, skills

MAX_STEPS = 8
LOG = Path(__file__).resolve().parent.parent / "logs" / "runs.jsonl"

BASE = """You are "{name}", a small personal assistant that starts out knowing almost nothing.
You must NOT rely on general world knowledge. The only things you may treat as true are:
1. what is written in MEMORY below (learned from earlier use),
2. what your skills return during this task,
3. what the user tells you.
If you cannot find something out with your skills, say plainly that you don't know yet.
The runtime may attach metadata such as an email address, a working directory or a date; treat it as background only, never repeat it unless the user asks.
Use file paths exactly as the user wrote them; never turn a relative path into an absolute one.

YOUR INSTRUCTIONS:
{instructions}

STANDING RULES (always follow these):
{rules}

BRAND GUIDE (follow it in anything the user or a customer will read):
{brand}

MEMORY:
{memory}

SKILLS you can call:
{skill_list}

You work in steps. Every reply must be exactly one JSON object, nothing else:
- to use a skill:  {{"thought": "...", "skill": "<name>", "args": {{...}}}}
- when finished:   {{"thought": "...", "answer": "<final answer for the user>"}}
Do not invent skill results. Call at most one skill per reply. Do not repeat a call that already failed with the same arguments."""


def system_prompt(agent: dict, task: str = "") -> str:
    skill_list = "\n".join(
        f"- {s}: {skills.SKILLS[s]['desc']} args={json.dumps(skills.SKILLS[s]['args'])}"
        for s in agent["skills"] if s in skills.SKILLS
    ) or "(none)"
    return BASE.format(
        name=agent["name"],
        instructions=agent.get("instructions") or "(none)",
        rules=memory.rules() or "(none yet)",
        brand=memory.brand() or "(none yet)",
        memory=memory.recall(task) or "(empty - you have not learned anything yet)",
        skill_list=skill_list,
    )


def _parse(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\s*|\s*```$", "", text, flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {"answer": text}


def _log(entry: dict) -> None:
    try:
        LOG.parent.mkdir(exist_ok=True)
        with LOG.open("a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


def run(agent_name: str, task: str, history: list[tuple[str, str]] | None = None,
        verbose: bool = True, learn: bool = True) -> str:
    agent = agents.get(agent_name)
    system = system_prompt(agent, task)
    transcript = ""
    for u, a in history or []:
        transcript += f"EARLIER - user: {u}\nEARLIER - you answered: {a}\n\n"
    transcript += f"TASK: {task}\n"
    tool_log, steps, answer = "", [], ""
    seen_calls: set[str] = set()
    max_steps = int(agent.get("max_steps") or MAX_STEPS)
    for _ in range(max_steps):
        step = _parse(llm.ask(system, transcript))
        if "skill" not in step:
            answer = str(step.get("answer") or step.get("thought") or "")
            break
        name, args = step["skill"], step.get("args") or {}
        key = json.dumps({"skill": name, "args": args}, sort_keys=True)
        if verbose:
            print(f"  {'  ' * skills._DEPTH['n']}[{agent['name']}] {name}({', '.join(f'{k}={str(v)[:60]!r}' for k, v in args.items())})")
        if name not in agent["skills"]:
            result = f"error: skill '{name}' is not available to this agent"
        elif key in seen_calls:
            result = "error: you already made this exact call; use the earlier result or try something else"
        elif skills.SKILLS[name].get("approval"):
            aid = approvals.enqueue(agent["name"], name, args, task)
            result = (f"queued for approval as #{aid}. It has NOT happened yet: the user must run `pai approve {aid}`. "
                      f"Do not retry; tell the user it is waiting for their approval.")
            if verbose:
                print(f"  [approval] #{aid} queued: {name}")
        else:
            result = skills.run(name, args)
        seen_calls.add(key)
        transcript += f"\nYOU CALLED: {key}\nRESULT:\n{result}\n"
        tool_log += f"{name}: {result[:2000]}\n"
        steps.append({"skill": name, "args": args, "ok": not result.startswith("error:")})
    else:
        answer = "I ran out of steps before finishing. Here is what I gathered:\n" + tool_log[:3000]

    learned = {"facts": [], "rules": []}
    if learn:
        try:
            # an agent that edits rules/agents itself already recorded the user's wishes: don't re-extract them
            self_managing = any(s["skill"] in ("set_rules", "set_agent_instructions") for s in steps)
            learned = memory.learn(agent["name"], task, answer, tool_log, allow_rules=not self_managing)
            if verbose and learned["facts"]:
                print("  [memory] learned: " + " | ".join(learned["facts"]))
            if verbose and learned["rules"]:
                print("  [rules] new rule: " + " | ".join(learned["rules"]))
        except llm.LLMError as e:
            if verbose:
                print(f"  [memory] could not update memory: {e}")
    _log({"time": _dt.datetime.now().isoformat(timespec="seconds"), "agent": agent["name"], "task": task,
          "steps": steps, "answer": answer, "learned": learned})
    return answer


def route(task: str) -> str | None:
    """Pick the best agent for a task by name. Returns None when no existing agent fits."""
    all_agents = agents.load_all()
    listing = "\n".join(f"- {a['name']}: {a['description']} (skills: {', '.join(a['skills']) or 'none'})"
                        for a in all_agents.values())
    raw = llm.ask(
        "You are a dispatcher. Reply with only the name of the single best agent for the task. "
        "If no agent has a purpose and skills that fit the task, reply with exactly NEW.",
        f"AGENTS:\n{listing}\n\nAVAILABLE SKILLS (for judging what a new agent could do):\n"
        + "\n".join(f"- {k}: {v['desc']}" for k, v in skills.SKILLS.items())
        + f"\n\nTASK: {task}",
    )
    name = raw.strip().strip("`\"' .").lower()
    if name == "new":
        return None
    return name if name in all_agents else next(iter(all_agents))


def create_agent_for(task: str, verbose: bool = True) -> str | None:
    """Ask the trainer to create an agent suited to a task. Returns the new agent's name."""
    before = set(agents.load_all())
    runner_agent = "trainer" if "trainer" in before else None
    if runner_agent is None:
        return None
    run(runner_agent, f"No existing agent fits this task. Create one suited for it, using the available skills, "
                      f"then stop (do not do the task yourself). TASK: {task}", verbose=verbose, learn=False)
    new = set(agents.load_all()) - before
    return next(iter(new)) if new else None
