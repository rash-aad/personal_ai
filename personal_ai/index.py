"""Hybrid document search: BM25 (lexical) + dense vectors (Ollama nomic-embed-text), fused with
reciprocal-rank fusion, then reranked by the model. Everything lives in index/index.db (SQLite).

index/sources.txt  - one file or directory per line; `pai index` (re)indexes them incrementally
logs/search.jsonl  - one line per query: what was retrieved and what the reranker kept
"""
import array
import datetime as _dt
import json
import math
import re
import sqlite3
import urllib.request
import zipfile
from pathlib import Path

from . import llm

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "index" / "index.db"
SOURCES = ROOT / "index" / "sources.txt"
SEARCH_LOG = ROOT / "logs" / "search.jsonl"
EMBED_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "nomic-embed-text"
CHUNK_CHARS, OVERLAP = 1200, 200
CANDIDATES = 20          # taken from the fused ranking and handed to the reranker
TEXT_EXT = {".txt", ".md", ".markdown", ".csv", ".json", ".py", ".html", ".rst"}
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".claude", ".cache"}
_TOKEN = re.compile(r"[a-z0-9]{2,}")

RERANK_SYSTEM = """You rerank search results. You get a QUERY and numbered PASSAGES.
Reply with ONLY a JSON array of the passage numbers that actually help answer the query, best first.
Leave out passages that are irrelevant. Example: [3, 1]"""


# ---------- parsing ----------

def _xml_text(xml: str, tag: str) -> list[str]:
    return re.findall(rf"<{tag}(?:\s[^>]*)?>([^<]*)</{tag}>", xml)


def _unescape(s: str) -> str:
    return s.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&apos;", "'")


def _docx(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf8", "replace")
    paras = ["".join(_xml_text(p, "w:t")) for p in xml.split("</w:p>")]
    return _unescape("\n".join(p for p in paras if p.strip()))


def _pptx(path: Path) -> str:
    out = []
    with zipfile.ZipFile(path) as z:
        slides = sorted((n for n in z.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)),
                        key=lambda n: int(re.findall(r"\d+", n)[0]))
        for i, n in enumerate(slides, 1):
            xml = z.read(n).decode("utf8", "replace")
            paras = [" ".join(_xml_text(p, "a:t")) for p in xml.split("</a:p>")]
            out.append(f"--- slide {i} ---\n" + "\n".join(p for p in paras if p.strip()))
    return _unescape("\n".join(out))


def _xlsx(path: Path) -> str:
    out = []
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        shared = [_unescape("".join(_xml_text(si, "t"))) for si in
                  z.read("xl/sharedStrings.xml").decode("utf8", "replace").split("</si>")] if "xl/sharedStrings.xml" in names else []
        for n in sorted(n for n in names if re.match(r"xl/worksheets/sheet\d+\.xml$", n)):
            xml = z.read(n).decode("utf8", "replace")
            out.append(f"--- {n.split('/')[-1]} ---")
            for row in re.findall(r"<row[^>]*>(.*?)</row>", xml, re.S):
                cells = []
                for attrs, body in re.findall(r"<c\b([^>]*?)(?:/>|>(.*?)</c>)", row, re.S):
                    v = re.search(r"<v>([^<]*)</v>", body)
                    t = re.search(r'\bt="(\w+)"', attrs)
                    if t and t.group(1) == "s" and v:
                        cells.append(shared[int(v.group(1))] if int(v.group(1)) < len(shared) else "")
                    elif t and t.group(1) == "inlineStr":
                        cells.append(_unescape("".join(_xml_text(body, "t"))))
                    else:
                        cells.append(v.group(1) if v else "")  # empty cells keep their column
                if any(c.strip() for c in cells):
                    out.append("\t".join(cells))
    return "\n".join(out)


def _pdf(path: Path) -> str:
    from pypdf import PdfReader
    return "\n".join(f"--- page {i + 1} ---\n{(p.extract_text() or '').strip()}"
                     for i, p in enumerate(PdfReader(path).pages))


def extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _pdf(path)
    if ext == ".docx":
        return _docx(path)
    if ext == ".pptx":
        return _pptx(path)
    if ext == ".xlsx":
        return _xlsx(path)
    if ext in TEXT_EXT:
        return path.read_text(errors="replace")
    return ""


def chunk(text: str) -> list[str]:
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + CHUNK_CHARS)
        if end < len(text):
            cut = text.rfind("\n", start + CHUNK_CHARS // 2, end)
            if cut == -1:
                cut = text.rfind(". ", start + CHUNK_CHARS // 2, end)
            if cut != -1:
                end = cut + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - OVERLAP, start + 1)
    return [c for c in chunks if c]


# ---------- embeddings ----------

def embed(texts: list[str], kind: str = "search_document") -> list[list[float]] | None:
    """Vectors from the local Ollama model, unit-normalised. None when Ollama is not running."""
    if not texts:
        return []
    out = []
    for i in range(0, len(texts), 32):
        body = json.dumps({"model": EMBED_MODEL, "input": [f"{kind}: {t}" for t in texts[i:i + 32]]}).encode()
        req = urllib.request.Request(EMBED_URL, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                vecs = json.load(r)["embeddings"]
        except Exception:
            return None
        for v in vecs:
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
    return out


def _pack(vec: list[float] | None) -> bytes | None:
    return array.array("f", vec).tobytes() if vec else None


def _unpack(blob: bytes | None) -> list[float] | None:
    if not blob:
        return None
    a = array.array("f")
    a.frombytes(blob)
    return list(a)


# ---------- storage ----------

def _db() -> sqlite3.Connection:
    DB.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(DB)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS docs (path TEXT PRIMARY KEY, mtime REAL, title TEXT);
        CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY, path TEXT, ord INTEGER, text TEXT, vec BLOB);
        CREATE INDEX IF NOT EXISTS chunks_path ON chunks(path);
    """)
    return con


def sources() -> list[Path]:
    if not SOURCES.exists():
        return []
    return [Path(ln.strip()).expanduser() for ln in SOURCES.read_text().splitlines() if ln.strip() and not ln.startswith("#")]


def add_sources(paths: list[str]) -> list[Path]:
    known = {str(p) for p in sources()}
    SOURCES.parent.mkdir(exist_ok=True)
    added = []
    with SOURCES.open("a") as f:
        for p in paths:
            rp = Path(p).expanduser().resolve()
            if str(rp) not in known:
                f.write(f"{rp}\n")
                known.add(str(rp))
                added.append(rp)
    return added


def _files(paths: list[Path]) -> list[Path]:
    out = []
    for p in paths:
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and not any(part in SKIP_DIRS or part.startswith(".") for part in f.relative_to(p).parts[:-1]) \
                        and not f.name.startswith(".") and f.suffix.lower() in TEXT_EXT | {".pdf", ".docx", ".pptx", ".xlsx"}:
                    out.append(f)
    return out


def build(paths: list[Path] | None = None, rebuild: bool = False, verbose: bool = True) -> dict:
    """Index every file under the given paths (default: index/sources.txt). Unchanged files are skipped."""
    con = _db()
    if rebuild:
        con.execute("DELETE FROM docs")
        con.execute("DELETE FROM chunks")
    files = _files(paths if paths is not None else sources())
    known = dict(con.execute("SELECT path, mtime FROM docs"))
    stats = {"files": len(files), "indexed": 0, "chunks": 0, "skipped": 0, "embedded": True}
    for f in files:
        key = str(f)
        if not rebuild and known.get(key) == f.stat().st_mtime:
            stats["skipped"] += 1
            continue
        try:
            parts = chunk(extract_text(f))
        except Exception as e:
            if verbose:
                print(f"  ! {f}: {type(e).__name__}: {e}")
            continue
        vecs = embed(parts)
        if vecs is None:
            stats["embedded"] = False
            vecs = [None] * len(parts)
        con.execute("DELETE FROM chunks WHERE path=?", (key,))
        con.executemany("INSERT INTO chunks (path, ord, text, vec) VALUES (?,?,?,?)",
                        [(key, i, t, _pack(v)) for i, (t, v) in enumerate(zip(parts, vecs))])
        con.execute("INSERT OR REPLACE INTO docs VALUES (?,?,?)", (key, f.stat().st_mtime, f.name))
        con.commit()
        stats["indexed"] += 1
        stats["chunks"] += len(parts)
        if verbose:
            print(f"  + {f.name}: {len(parts)} chunk(s)")
    # drop documents that no longer exist on disk
    for key in known:
        if not Path(key).exists():
            con.execute("DELETE FROM chunks WHERE path=?", (key,))
            con.execute("DELETE FROM docs WHERE path=?", (key,))
    con.commit()
    con.close()
    return stats


def stats() -> dict:
    if not DB.exists():
        return {"docs": 0, "chunks": 0, "embedded": 0}
    con = _db()
    d = con.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
    c, e = con.execute("SELECT COUNT(*), COUNT(vec) FROM chunks").fetchone()
    con.close()
    return {"docs": d, "chunks": c, "embedded": e}


# ---------- search ----------

def _bm25(query: str, rows: list[tuple], k1: float = 1.5, b: float = 0.75) -> list[float]:
    docs = [_TOKEN.findall(r[2].lower()) for r in rows]
    n = len(docs) or 1
    avg = sum(map(len, docs)) / n
    df: dict[str, int] = {}
    for d in docs:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    q = set(_TOKEN.findall(query.lower()))
    scores = []
    for d in docs:
        tf: dict[str, int] = {}
        for t in d:
            if t in q:
                tf[t] = tf.get(t, 0) + 1
        s = 0.0
        for t, f in tf.items():
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(d) / (avg or 1)))
        scores.append(s)
    return scores


def _rrf(*rankings: list[int], k: int = 60) -> list[int]:
    score: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            score[idx] = score.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(score, key=lambda i: -score[i])


def _rerank(query: str, hits: list[dict], k: int) -> list[dict]:
    listing = "\n\n".join(f"[{i + 1}] ({h['title']}) {h['text'][:700]}" for i, h in enumerate(hits))
    try:
        raw = llm.ask(RERANK_SYSTEM, f"QUERY: {query}\n\nPASSAGES:\n{listing}", timeout=120, retries=0)
        order = [int(x) for x in re.findall(r"\d+", re.search(r"\[.*?\]", raw, re.S).group(0))]
    except Exception:
        return hits[:k]
    picked = [hits[i - 1] for i in order if 1 <= i <= len(hits)]
    return (picked or hits)[:k]


def search(query: str, k: int = 5, rerank: bool = True) -> list[dict]:
    """Hybrid search. Returns up to k dicts: path, title, ord, text, score."""
    if not DB.exists():
        return []
    con = _db()
    rows = con.execute("SELECT id, path, text, vec, ord FROM chunks").fetchall()
    titles = dict(con.execute("SELECT path, title FROM docs"))
    con.close()
    if not rows:
        return []
    lex = _bm25(query, rows)
    lex_rank = [i for i in sorted(range(len(rows)), key=lambda i: -lex[i]) if lex[i] > 0][:CANDIDATES * 2]
    rankings = [lex_rank]
    qv = embed([query], "search_query") if any(r[3] for r in rows) else None
    dense_used = False
    if qv:
        qv = qv[0]
        sims = [sum(a * b for a, b in zip(qv, v)) if (v := _unpack(r[3])) else -1.0 for r in rows]
        rankings.append(sorted(range(len(rows)), key=lambda i: -sims[i])[:CANDIDATES * 2])
        dense_used = True
    fused = _rrf(*rankings)[:CANDIDATES]
    hits = [{"id": rows[i][0], "path": rows[i][1], "title": titles.get(rows[i][1], Path(rows[i][1]).name),
             "ord": rows[i][4], "text": rows[i][2], "score": round(lex[i], 3)} for i in fused]
    kept = _rerank(query, hits, k) if rerank and hits else hits[:k]
    _log({"time": _dt.datetime.now().isoformat(timespec="seconds"), "query": query, "dense": dense_used,
          "candidates": [h["id"] for h in hits], "kept": [h["id"] for h in kept]})
    return kept


def _log(entry: dict) -> None:
    try:
        SEARCH_LOG.parent.mkdir(exist_ok=True)
        with SEARCH_LOG.open("a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


def format_hits(hits: list[dict]) -> str:
    if not hits:
        return "(nothing found - is the index built? run `pai index`)"
    return "\n\n".join(f"[{i + 1}] {h['title']} (chunk {h['ord'] + 1}) - {h['path']}\n{h['text']}" for i, h in enumerate(hits))
