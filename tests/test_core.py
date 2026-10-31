"""Fast tests with a fake model - no `claude` CLI needed. Run: .venv/bin/pytest -q"""
import json

import pytest

from personal_ai import agents, memory, runner, skills


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    """Point every file location at a temp dir and give the model scripted answers."""
    monkeypatch.setattr(memory, "FACTS", tmp_path / "memory" / "facts.md")
    monkeypatch.setattr(memory, "RULES", tmp_path / "memory" / "rules.md")
    monkeypatch.setattr(agents, "DIR", tmp_path / "agents")
    monkeypatch.setattr(agents, "HISTORY", tmp_path / "agents" / "history")
    monkeypatch.setattr(runner, "LOG", tmp_path / "logs" / "runs.jsonl")
    monkeypatch.setattr(skills, "WORKSPACE", tmp_path / "workspace")
    monkeypatch.setattr(skills, "ROOT", tmp_path)
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "reader.json").write_text(json.dumps(
        {"name": "reader", "description": "reads", "skills": ["read_txt"], "instructions": "Read files."}))
    (tmp_path / "note.txt").write_text("the code word is TANGERINE")


class FakeLLM:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def __call__(self, system, prompt, **kw):
        self.calls.append((system, prompt))
        return self.replies.pop(0) if self.replies else '{"facts": [], "rules": []}'


def test_runner_calls_skill_then_answers_and_learns(monkeypatch, tmp_path):
    fake = FakeLLM(
        '{"thought": "open it", "skill": "read_txt", "args": {"path": "note.txt"}}',
        '```json\n{"thought": "done", "answer": "It is TANGERINE."}\n```',
        '{"facts": ["note.txt says the code word is TANGERINE"], "rules": ["Always answer in one line"]}',
    )
    monkeypatch.setattr(runner.llm, "ask", fake)
    monkeypatch.setattr(memory.llm, "ask", fake)
    answer = runner.run("reader", "what is the code word in note.txt?", verbose=False)
    assert answer == "It is TANGERINE."
    assert "RESULT:\nthe code word is TANGERINE" in fake.calls[1][1]
    assert "TANGERINE" in memory.load()
    assert "Always answer in one line" in memory.rules()
    log = json.loads(runner.LOG.read_text().splitlines()[-1])
    assert log["steps"][0]["skill"] == "read_txt" and log["steps"][0]["ok"]
    # rules and memory show up in the next prompt
    assert "Always answer in one line" in runner.system_prompt(agents.get("reader"))


def test_runner_rejects_unavailable_skill_and_repeated_call(monkeypatch):
    fake = FakeLLM(
        '{"skill": "web_search", "args": {"query": "x"}}',
        '{"skill": "read_txt", "args": {"path": "missing.txt"}}',
        '{"skill": "read_txt", "args": {"path": "missing.txt"}}',
        '{"answer": "gave up"}',
    )
    monkeypatch.setattr(runner.llm, "ask", fake)
    monkeypatch.setattr(memory.llm, "ask", fake)
    assert runner.run("reader", "t", verbose=False, learn=False) == "gave up"
    assert "not available to this agent" in fake.calls[1][1]
    assert "FileNotFoundError" in fake.calls[2][1]
    assert "already made this exact call" in fake.calls[3][1]


def test_runner_gives_up_after_max_steps(monkeypatch):
    fake = FakeLLM(*['{"skill": "read_txt", "args": {"path": "note%d.txt"}}' % i for i in range(20)])
    monkeypatch.setattr(runner.llm, "ask", fake)
    out = runner.run("reader", "t", verbose=False, learn=False)
    assert out.startswith("I ran out of steps") and len(fake.calls) == runner.MAX_STEPS


def test_parse_handles_plain_text():
    assert runner._parse("just words") == {"answer": "just words"}
    assert runner._parse('text before {"answer": "x"} after') == {"answer": "x"}


def test_memory_dedup_forget_and_recall():
    assert memory.add(["A is B", "A is B", "C is D"], "t") == ["A is B", "C is D"]
    assert memory.add(["a  is b"], "t") == []          # same fact, different spacing/case
    assert memory.count() == 2
    assert memory.forget("c is") == 1 and memory.count() == 1
    memory.add([f"fact number {i} about topic{i}" for i in range(100)], "t")
    shown = memory.recall("tell me about topic42")
    assert "topic42" in shown and "showing" in shown
    assert shown.count("\n- ") < memory.count()
    assert "fact number 99" in shown                     # recent facts always included
    memory.clear()
    assert memory.load() == "" and memory.rules() == ""


def test_agents_update_history_revert(monkeypatch):
    monkeypatch.setattr(agents.llm, "ask", lambda s, p, **k: "Drafted instructions.")
    agents.create("Web Guy", "browses", ["fetch_url"], "")
    assert agents.get("web_guy")["instructions"] == "Drafted instructions."
    with pytest.raises(ValueError):
        agents.create("bad", "x", ["nope"], "x")
    agents.update("reader", instructions="Read files carefully.", by="trainer")
    assert agents.get("reader")["instructions"] == "Read files carefully."
    assert len(agents.history("reader")) == 1
    agents.revert("reader")
    assert agents.get("reader")["instructions"] == "Read files."
    assert agents.history("reader") == []


def test_trainer_skills_roundtrip():
    assert "reader" in skills.run("list_agents", {})
    assert "updated instructions" in skills.run("set_agent_instructions", {"name": "reader", "instructions": "New."})
    assert json.loads(skills.run("get_agent", {"name": "reader"}))["instructions"] == "New."
    assert "created" in skills.run("create_agent", {"name": "x", "description": "d", "skills": "read_txt", "instructions": "i"})
    assert skills.run("remember", {"fact": "sky is blue"}) == "stored"
    assert skills.run("remember", {"fact": "sky is blue"}) == "already known"
    assert "removed 1" in skills.run("forget", {"text": "sky"})
    skills.run("set_rules", {"content": "- be brief"})
    assert skills.run("get_rules", {}) == "- be brief"
    assert skills.run("read_txt", {"wrong": 1}).startswith("error: bad arguments")
    assert skills.run("read_txt", "notadict").startswith("error: args")
    assert skills.run("nope", {}).startswith("error: unknown skill")
    assert "wrote" in skills.run("write_file", {"path": "a/b.txt", "content": "hi"})
    assert (skills.WORKSPACE / "a" / "b.txt").read_text() == "hi"


def test_planner_delegates_and_blocks_nesting(monkeypatch, tmp_path):
    (tmp_path / "agents" / "planner.json").write_text(json.dumps(
        {"name": "planner", "description": "plans", "skills": ["delegate"], "instructions": "Plan."}))
    fake = FakeLLM(
        '{"skill": "delegate", "args": {"agent": "reader", "task": "read note.txt"}}',   # planner step 1
        '{"skill": "read_txt", "args": {"path": "note.txt"}}',                           # reader step 1
        '{"answer": "TANGERINE"}',                                                        # reader answer
        '{"facts": [], "rules": []}',                                                     # reader learn
        '{"skill": "delegate", "args": {"agent": "planner", "task": "loop"}}',           # planner tries itself
        '{"skill": "delegate", "args": {"agent": "ghost", "task": "x"}}',                # unknown agent
        '{"answer": "plan done: TANGERINE"}',
    )
    monkeypatch.setattr(runner.llm, "ask", fake)
    monkeypatch.setattr(memory.llm, "ask", fake)
    out = runner.run("planner", "find the code word", verbose=False, learn=False)
    assert out == "plan done: TANGERINE"
    assert "RESULT:\nTANGERINE" in fake.calls[4][1]
    assert "itself a planner" in fake.calls[5][1]
    assert "no agent 'ghost'" in fake.calls[6][1]
    assert skills._DEPTH["n"] == 0


def test_route_new_triggers_trainer_creation(monkeypatch, tmp_path):
    (tmp_path / "agents" / "trainer.json").write_text(json.dumps(
        {"name": "trainer", "description": "trains", "skills": ["create_agent"], "instructions": "Train."}))
    fake = FakeLLM(
        "NEW",
        '{"skill": "create_agent", "args": {"name": "lister", "description": "lists", "skills": "list_files", "instructions": "List."}}',
        '{"answer": "created lister"}',
        "lister",
    )
    monkeypatch.setattr(runner.llm, "ask", fake)
    assert runner.route("list files") is None
    assert runner.create_agent_for("list files", verbose=False) == "lister"
    assert agents.get("lister")["skills"] == ["list_files"]
    assert runner.route("list files") == "lister"
    assert "NEW" in fake.calls[0][0]  # dispatcher is told it may answer NEW
