#!/usr/bin/env python3
"""Seed data/business.db from the Praxos Reddit leads workbook (the 'Leads' sheet). Re-run to refresh.
Usage: .venv/bin/python scripts/import_leads.py path/to/praxos-reddit-leads.xlsx"""
import re
import sqlite3
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if len(sys.argv) < 2:
    sys.exit("usage: import_leads.py path/to/praxos-reddit-leads.xlsx")
SRC = Path(sys.argv[1])
DB = ROOT / "data" / "business.db"


def unescape(s):
    return s.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&apos;", "'")


def rows(sheet_xml, shared):
    out = []
    for row in re.findall(r"<row[^>]*>(.*?)</row>", sheet_xml, re.S):
        cells = {}
        for attrs, body in re.findall(r"<c\b([^>]*?)(?:/>|>(.*?)</c>)", row, re.S):
            col = re.search(r'r="([A-Z]+)\d+"', attrs).group(1)
            v = re.search(r"<v>([^<]*)</v>", body or "")
            t = re.search(r'\bt="(\w+)"', attrs)
            if t and t.group(1) == "s" and v:
                cells[col] = shared[int(v.group(1))]
            elif t and t.group(1) == "inlineStr":
                cells[col] = unescape("".join(re.findall(r"<t[^>]*>([^<]*)</t>", body or "")))
            elif v:
                cells[col] = v.group(1)
        if cells:
            out.append(cells)
    return out


z = zipfile.ZipFile(SRC)
shared = [unescape("".join(re.findall(r"<t[^>]*>([^<]*)</t>", si))) for si in z.read("xl/sharedStrings.xml").decode().split("</si>")]
data = rows(z.read("xl/worksheets/sheet2.xml").decode(), shared)
header, body = data[0], data[1:]
cols = sorted(header, key=lambda c: (len(c), c))
names = [re.sub(r"[^a-z0-9]+", "_", header[c].strip().lower()).strip("_") for c in cols]
DB.parent.mkdir(exist_ok=True)
con = sqlite3.connect(DB)
con.execute("DROP TABLE IF EXISTS leads")
con.execute("CREATE TABLE leads (" + ", ".join(f'"{n}" TEXT' for n in names) + ")")
con.executemany("INSERT INTO leads VALUES (" + ",".join("?" * len(names)) + ")", [[r.get(c, "") for c in cols] for r in body])
con.commit()
print(f"{DB}: leads table, {len(body)} rows, columns: {', '.join(names)}")
