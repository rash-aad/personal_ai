"""Hybrid search, MCP connectors and the approval queue - offline, fake model / fake embeddings."""
import json
import sqlite3
import sys
import zipfile

import pytest

from personal_ai import agents, approvals, index, mcp, memory, runner, skills
from tests.test_core import FakeLLM  # noqa: F401  (fixture import side-effect not needed; class reused)


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(index, "DB", tmp_path / "index" / "index.db")
    monkeypatch.setattr(index, "SOURCES", tmp_path / "index" / "sources.txt")
    monkeypatch.setattr(index, "SEARCH_LOG", tmp_path / "logs" / "search.jsonl")
    monkeypatch.setattr(approvals, "QUEUE", tmp_path / "approvals" / "queue.jsonl")
    monkeypatch.setattr(agents, "DIR", tmp_path / "agents")
    monkeypatch.setattr(agents, "HISTORY", tmp_path / "agents" / "history")
    monkeypatch.setattr(memory, "FACTS", tmp_path / "memory" / "facts.md")
    monkeypatch.setattr(memory, "RULES", tmp_path / "memory" / "rules.md")
    monkeypatch.setattr(memory, "BRAND", tmp_path / "memory" / "brand.md")
    monkeypatch.setattr(runner, "LOG", tmp_path / "logs" / "runs.jsonl")
    (tmp_path / "agents").mkdir()


VOCAB = ["otter", "river", "invoice", "granite", "clinic", "whatsapp"]


def fake_embed(texts, kind="search_document"):
    """Bag-of-vocabulary vectors: cosine similarity == shared vocabulary words."""
    out = []
    for t in texts:
        v = [1.0 if w in t.lower() else 0.0 for w in VOCAB]
        n = sum(v) ** 0.5 or 1.0
        out.append([x / n for x in v])
    return out


def test_chunk_respects_size_and_overlap():
    text = "\n".join(f"line {i} " + "x" * 50 for i in range(100))
    parts = index.chunk(text)
    assert len(parts) > 1 and all(len(p) <= index.CHUNK_CHARS for p in parts)
    assert parts[0][-30:] in "".join(parts[:2]) and "".join(parts).count("line 0 ") == 1
    assert index.chunk("   ") == []


def test_office_parsers(tmp_path):
    with zipfile.ZipFile(tmp_path / "a.docx", "w") as z:
        z.writestr("word/document.xml", "<w:document><w:p><w:r><w:t>Hello</w:t><w:t xml:space=\"preserve\"> world</w:t></w:r></w:p><w:p><w:r><w:t>Two &amp; three</w:t></w:r></w:p></w:document>")
    assert index.extract_text(tmp_path / "a.docx") == "Hello world\nTwo & three"
    with zipfile.ZipFile(tmp_path / "a.pptx", "w") as z:
        z.writestr("ppt/slides/slide2.xml", "<p:sld><a:p><a:t>Second</a:t></a:p></p:sld>")
        z.writestr("ppt/slides/slide1.xml", "<p:sld><a:p><a:t>First</a:t><a:t>slide</a:t></a:p></p:sld>")
    assert index.extract_text(tmp_path / "a.pptx") == "--- slide 1 ---\nFirst slide\n--- slide 2 ---\nSecond"
    with zipfile.ZipFile(tmp_path / "a.xlsx", "w") as z:
        z.writestr("xl/sharedStrings.xml", "<sst><si><t>name</t></si><si><t>Rex</t></si></sst>")
        z.writestr("xl/worksheets/sheet1.xml",
                   '<ws><sheetData><row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1"><v>1</v></c></row>'
                   '<row r="2"><c r="A2" t="s"><v>1</v></c><c r="B2" t="inlineStr"><is><t>7</t></is></c></row>'
                   '<row r="3"><c r="A3" s="1"/><c r="B3"><v>9</v></c></row></sheetData></ws>')
    assert index.extract_text(tmp_path / "a.xlsx") == "--- sheet1.xml ---\nname\t1\nRex\t7\n\t9"  # empty cell keeps its column
    assert index.extract_text(tmp_path / "a.bin") == ""


def test_hybrid_search_fuses_lexical_and_dense_then_reranks(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "animals.txt").write_text("The otter lives by the river and eats fish.")
    (docs / "billing.md").write_text("Invoice 42 for the granite slab order is due Friday.")
    (docs / "notes.txt").write_text("Random note about clinics and WhatsApp booking links.")
    (docs / "skip.png").write_bytes(b"\x89PNG")
    monkeypatch.setattr(index, "embed", fake_embed)
    index.add_sources([str(docs)])
    st = index.build(verbose=False)
    assert st["indexed"] == 3 and st["embedded"] and index.stats()["embedded"] == 3
    assert index.build(verbose=False)["skipped"] == 3  # unchanged files are not re-indexed

    hits = index.search("otter", k=2, rerank=False)
    assert hits[0]["title"] == "animals.txt" and "river" in hits[0]["text"]
    # dense-only match: the query word is in the vocabulary but "river" is not in the query text lexically
    hits = index.search("river", k=1, rerank=False)
    assert hits[0]["title"] == "animals.txt"
    # reranker can reorder: the model says passage 2 is best
    fake = FakeLLM("[2, 1]")
    monkeypatch.setattr(index.llm, "ask", fake)
    first_two = index.search("otter invoice", k=2, rerank=False)
    reranked = index.search("otter invoice", k=2)
    assert reranked[0]["id"] == first_two[1]["id"] and "PASSAGES" in fake.calls[0][1]
    # a broken reranker reply falls back to the fused order
    monkeypatch.setattr(index.llm, "ask", FakeLLM("no idea"))
    assert index.search("otter invoice", k=2)[0]["id"] == first_two[0]["id"]
    log = [json.loads(l) for l in index.SEARCH_LOG.read_text().splitlines()]
    assert log[-1]["query"] == "otter invoice" and log[-1]["dense"] and log[-1]["kept"]
    assert "animals.txt" in skills.run("search_docs", {"query": "otter", "k": 1})


def test_search_works_without_ollama(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("granite quotation for Nishan")
    monkeypatch.setattr(index, "embed", lambda texts, kind="x": None)
    st = index.build([tmp_path / "a.txt"], verbose=False)
    assert st["indexed"] == 1 and not st["embedded"] and index.stats()["embedded"] == 0
    hits = index.search("granite", rerank=False)
    assert hits and hits[0]["title"] == "a.txt"
    assert index.search("nothing matches here", rerank=False) == []


def test_mcp_db_server_roundtrip(tmp_path):
    db = tmp_path / "biz.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE leads (name TEXT, status TEXT)")
    con.executemany("INSERT INTO leads VALUES (?,?)", [("A", "new"), ("B", "replied")])
    con.commit()
    con.close()
    c = mcp.Client("db", [sys.executable, "-m", "personal_ai.connectors.db_server"], {"PAI_DB_PATH": str(db)})
    try:
        assert {t["name"] for t in c.list_tools()} == {"schema", "query"}
        assert c.call("schema", {}) == "leads (2 rows): name TEXT, status TEXT"
        assert c.call("query", {"sql": "select name from leads where status='replied'"}) == "name\nB\n(1 rows)"
        with pytest.raises(mcp.MCPError, match="SELECT"):
            c.call("query", {"sql": "delete from leads"})
        with pytest.raises(mcp.MCPError, match="unknown tool"):
            c.call("drop", {})
    finally:
        c.close()


def test_mcp_config_and_skill_wrapper(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp, "CONFIG", tmp_path / "mcp.json")
    monkeypatch.setattr(mcp, "_clients", {})
    assert skills.run("db_schema", {}).startswith("error: no MCP server 'db'")
    db = tmp_path / "biz.db"
    sqlite3.connect(db).execute("CREATE TABLE t (x)").connection.commit()
    (tmp_path / "mcp.json").write_text(json.dumps({"mcpServers": {"db": {
        "command": ["python", "-m", "personal_ai.connectors.db_server"], "env": {"PAI_DB_PATH": str(db)}}}}))
    assert skills.run("db_schema", {}) == "t (0 rows): x"
    assert skills.run("db_query", {"sql": "drop table t"}).startswith("error:")
    mcp._clients["db"].close()


def test_email_send_is_queued_and_runs_only_on_approval(tmp_path, monkeypatch):
    (tmp_path / "agents" / "biz.json").write_text(json.dumps(
        {"name": "biz", "description": "b", "skills": ["email_send"], "instructions": "Mail."}))
    sent = []
    monkeypatch.setattr(skills, "_mcp", lambda server, tool, **a: sent.append((server, tool, a)) or f"sent to {a['to']}")
    fake = FakeLLM(
        '{"skill": "email_send", "args": {"to": "a@b.c", "subject": "Hi", "body": "Hello"}}',
        '{"answer": "Queued, waiting for you to approve."}',
    )
    monkeypatch.setattr(runner.llm, "ask", fake)
    out = runner.run("biz", "email a@b.c", verbose=False, learn=False)
    assert out == "Queued, waiting for you to approve."
    assert "queued for approval as #1" in fake.calls[1][1] and sent == []
    assert skills.run("email_send", {"to": "x", "subject": "y", "body": "z"}).startswith("error: email_send needs user approval")
    item = approvals.pending()[0]
    assert item["skill"] == "email_send" and item["agent"] == "biz" and "#1" in approvals.describe(item)
    assert approvals.approve(1) == "sent to a@b.c"
    assert sent == [("email", "send", {"to": "a@b.c", "subject": "Hi", "body": "Hello"})]
    assert approvals.get(1)["status"] == "done" and approvals.pending() == []
    with pytest.raises(ValueError, match="already done"):
        approvals.approve(1)
    approvals.enqueue("biz", "email_send", {"to": "q"}, "t")
    approvals.reject(2)
    assert approvals.get(2)["status"] == "rejected"
    with pytest.raises(KeyError):
        approvals.get(9)


def test_brand_guide_goes_into_every_prompt(tmp_path):
    (tmp_path / "agents" / "r.json").write_text(json.dumps({"name": "r", "description": "", "skills": [], "instructions": ""}))
    assert "BRAND GUIDE" in runner.system_prompt(agents.get("r")) and "(none yet)" in runner.system_prompt(agents.get("r"))
    memory.BRAND.parent.mkdir()
    memory.BRAND.write_text("Tagline: Your clinic's memory.")
    assert "Your clinic's memory" in runner.system_prompt(agents.get("r"))
