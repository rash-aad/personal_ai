"""MCP server: read-only access to the business database (SQLite).
Run: python -m personal_ai.connectors.db_server   (env PAI_DB_PATH = path to the .db file)"""
import os
import re
import sqlite3
from pathlib import Path

from .. import mcp

ROOT = Path(__file__).resolve().parent.parent.parent
MAX_ROWS = 50


def _con() -> sqlite3.Connection:
    path = Path(os.environ.get("PAI_DB_PATH") or ROOT / "data" / "business.db")
    if not path.exists():
        raise FileNotFoundError(f"database not found: {path} (set PAI_DB_PATH)")
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def schema() -> str:
    con = _con()
    out = []
    for (name,) in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"):
        cols = ", ".join(f"{c[1]} {c[2]}".strip() for c in con.execute(f'PRAGMA table_info("{name}")'))
        n = con.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
        out.append(f"{name} ({n} rows): {cols}")
    con.close()
    return "\n".join(out) or "(no tables)"


def query(sql: str, limit: str = str(MAX_ROWS)) -> str:
    s = sql.strip().rstrip(";")
    if ";" in s or not re.match(r"(?is)^\s*(select|with)\b", s):
        raise ValueError("only a single SELECT statement is allowed (the database is read-only)")
    limit = min(int(limit or MAX_ROWS), MAX_ROWS)
    if not re.search(r"(?is)\blimit\s+\d+\s*$", s):
        s += f" LIMIT {limit}"
    con = _con()
    cur = con.execute(s)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    con.close()
    if not rows:
        return "(no rows)"
    lines = ["\t".join(cols)] + ["\t".join("" if v is None else str(v) for v in r) for r in rows]
    return "\n".join(lines) + (f"\n({len(rows)} rows, capped at {limit})" if len(rows) >= limit else f"\n({len(rows)} rows)")


TOOLS = {
    "schema": {"fn": schema, "args": {}, "desc": "List the tables, their columns and row counts."},
    "query": {"fn": query, "args": {"sql": "a single SELECT statement", "limit": f"max rows (default {MAX_ROWS})"},
              "optional": {"limit": True}, "desc": "Run a read-only SELECT and return the rows."},
}

if __name__ == "__main__":
    mcp.serve("business-db", TOOLS)
