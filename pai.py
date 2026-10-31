#!/usr/bin/env python3
"""personal ai - a dumb assistant that learns as you use it. Backend: the `claude` CLI."""
import argparse
import json
import os
import sys

from personal_ai import agents, approvals, index, llm, memory, runner, skills


_USE_COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ


def _paint(text, code):
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else str(text)


def _title(text):
    print(f"\n{_paint(text, '1;36')}\n{_paint('─' * len(text), '90')}")


def _label(text):
    return _paint(text, "1;34")


def cmd_agents(_):
    _title("Your agents")
    for a in agents.load_all().values():
        print(f"{_paint(a['name'], '1;32'):<19} {a['description']}")
        print(f"{'':19}{_paint('Skills', '90')}: {', '.join(a['skills']) or 'No tools'}")


def cmd_show(args):
    a = agents.get(args.agent)
    if args.prompt:
        print(runner.system_prompt(a))
    else:
        _title(f"Agent · {a['name']}")
        print(f"{_label('Purpose')}     {a.get('description') or '—'}")
        print(f"{_label('Skills')}      {', '.join(a.get('skills', [])) or 'No tools'}")
        print(f"{_label('Instructions')}\n{a.get('instructions') or '—'}")
        if a.get("max_steps"):
            print(f"\n{_label('Step limit')}   {a['max_steps']}")
        hist = agents.history(args.agent)
        if hist:
            print(f"\n{_paint(f'{len(hist)} earlier version(s)', '90')} · latest: {hist[-1].name}")
        return


def cmd_revert(args):
    print(f"restored previous version: {agents.revert(args.agent)}")


def cmd_run(args):
    print(runner.run(args.agent, " ".join(args.task)))


def cmd_chat(args):
    names = list(agents.load_all())
    _title("personal ai")
    print("Ask anything, or choose an agent with " + _paint("@name", "1;36") + ".")
    print(f"Available: {', '.join('@' + n for n in names)}  ·  Type {_paint('quit', '1;36')} to leave.")
    history: list[tuple[str, str]] = []
    while True:
        try:
            line = input(f"\n{_paint('you', '1;36')}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line or line in {"quit", "exit"}:
            break
        if line.startswith("@"):
            name, _, task = line[1:].partition(" ")
        else:
            name, task = runner.route(line), line
            if name is None:
                print(f"  {_paint('·', '33')} No agent fits yet; asking the trainer to create one.")
                name = runner.create_agent_for(task)
                if name is None:
                    print("  could not create an agent for this task; try @trainer or 'pai create-agent'")
                    continue
            print(f"  {_paint('↳', '90')} routed to {_paint(name, '1;32')}")
        if not task.strip():
            print("  (empty task)")
            continue
        try:
            answer = runner.run(name, task, history)
        except (KeyError, llm.LLMError) as e:
            print(f"  {_paint('Error', '1;31')}: {e}")
            continue
        print(f"\n{_paint(name, '1;32')}> {answer}")
        history.append((task, answer))
        history = history[-6:]


def cmd_create_agent(args):
    name = args.name or input("agent name: ").strip()
    description = args.description or input("what is it for (one line): ").strip()
    print("available skills:\n" + "\n".join(f"  {k:<24} {v['desc']}" for k, v in skills.SKILLS.items()))
    picked = args.skills if args.skills is not None else input("skills (comma separated, blank for none): ")
    skill_names = [s.strip() for s in picked.split(",") if s.strip()]
    instructions = args.instructions
    if instructions is None:
        instructions = input("instructions (blank = let the AI draft them): ").strip()
    path = agents.create(name, description, skill_names, instructions)
    print(f"created {path}")
    print(path.read_text())


def cmd_delete_agent(args):
    agents.delete(args.agent)
    print(f"deleted agent '{args.agent}'")


def cmd_teach(args):
    text = " ".join(args.text)
    if args.rule:
        added = memory.add_rules([text])
        print("rule added" if added else "rule already present")
    else:
        added = memory.add([text], "user")
        print("stored" if added else "already known")


def cmd_memory(args):
    if args.clear:
        memory.clear()
        print("memory and rules cleared - the AI is dumb again.")
        return
    if args.forget:
        print(f"removed {memory.forget(args.forget)} fact(s)")
        return
    facts = memory.load()
    _title("Memory")
    print(f"{_label(f'Facts · {memory.count()}')}\n{facts or 'Nothing learned yet.'}")
    print(f"\n{_label('Standing rules')}\n{memory.rules() or 'No rules set.'}")


def cmd_rules(args):
    _title("Standing rules")
    print(memory.rules() or "No rules set.")


def cmd_log(args):
    if not runner.LOG.exists():
        print("(no runs yet)")
        return
    lines = runner.LOG.read_text().splitlines()[-args.n:]
    for ln in lines:
        e = json.loads(ln)
        used = ", ".join(s["skill"] for s in e["steps"]) or "-"
        print(f"{e['time']}  {e['agent']:<9} {e['task'][:60]!r}\n{'':21}skills: {used}  learned: {len(e['learned']['facts'])} fact(s)")


def cmd_index(args):
    if args.add:
        added = index.add_sources(args.add)
        print("added source(s): " + ", ".join(map(str, added)) if added else "already listed")
    if not index.sources():
        print("no sources yet: `pai index --add <file-or-folder> ...`")
        return
    if args.list:
        _title("Indexed sources")
        print("\n".join(map(str, index.sources())))
        return
    st = index.build(rebuild=args.rebuild)
    s = index.stats()
    _title("Index updated")
    print(f"Indexed {_paint(st['indexed'], '1;32')} files · {st['skipped']} unchanged · {st['chunks']} new passages")
    print(f"Library: {s['docs']} documents · {s['chunks']} passages · {s['embedded']} with vectors")
    if not st["embedded"]:
        print("warning: Ollama was not reachable, so new chunks have no vectors (lexical search only). "
              "Start Ollama and run `pai index --rebuild`.")


def cmd_search(args):
    hits = index.search(" ".join(args.query), k=args.k, rerank=not args.no_rerank)
    _title("Search results")
    print(index.format_hits(hits))


def cmd_approvals(args):
    items = approvals.pending() if not args.all else approvals._load()
    _title("Approvals" if not args.all else "Approval history")
    if not items:
        print("Nothing waiting for approval." if not args.all else "No approval activity yet.")
        return
    print("\n".join(approvals.describe(i) for i in items) or "(nothing waiting for approval)")


def cmd_approve(args):
    for i in args.id:
        print(f"#{i}: {approvals.approve(i)}")


def cmd_reject(args):
    for i in args.id:
        approvals.reject(i)
        print(f"#{i}: rejected")


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="pai",
        description="A personal assistant that learns as you use it.",
        epilog="Start here:  pai chat   ·   pai agents   ·   pai run reader 'summarise examples/sample.pdf'",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("agents", help="list agents").set_defaults(fn=cmd_agents)

    s = sub.add_parser("show", help="show an agent's definition")
    s.add_argument("agent")
    s.add_argument("--prompt", action="store_true", help="print the full system prompt it would run with")
    s.set_defaults(fn=cmd_show)

    rv = sub.add_parser("revert", help="restore an agent's previous instructions")
    rv.add_argument("agent")
    rv.set_defaults(fn=cmd_revert)

    r = sub.add_parser("run", help="run one agent on one task")
    r.add_argument("agent")
    r.add_argument("task", nargs="+")
    r.set_defaults(fn=cmd_run)

    sub.add_parser("chat", help="interactive session, auto-routes to an agent").set_defaults(fn=cmd_chat)

    c = sub.add_parser("create-agent", help="create a new agent (interactive unless flags given)")
    c.add_argument("--name")
    c.add_argument("--description")
    c.add_argument("--skills", help="comma separated skill names")
    c.add_argument("--instructions", help="system instructions; omit to be asked, '' to let the AI draft them")
    c.set_defaults(fn=cmd_create_agent)

    d = sub.add_parser("delete-agent", help="delete an agent")
    d.add_argument("agent")
    d.set_defaults(fn=cmd_delete_agent)

    t = sub.add_parser("teach", help="store a fact directly (or a standing rule with --rule)")
    t.add_argument("text", nargs="+")
    t.add_argument("--rule", action="store_true")
    t.set_defaults(fn=cmd_teach)

    m = sub.add_parser("memory", help="show what the AI has learned")
    m.add_argument("--clear", action="store_true", help="forget everything")
    m.add_argument("--forget", metavar="TEXT", help="remove facts containing TEXT")
    m.set_defaults(fn=cmd_memory)

    sub.add_parser("rules", help="show the standing rules").set_defaults(fn=cmd_rules)

    ix = sub.add_parser("index", help="build/update the hybrid search index from index/sources.txt")
    ix.add_argument("--add", nargs="+", metavar="PATH", help="add files/folders to the source list first")
    ix.add_argument("--rebuild", action="store_true", help="re-index everything from scratch")
    ix.add_argument("--list", action="store_true", help="only list the sources")
    ix.set_defaults(fn=cmd_index)

    se = sub.add_parser("search", help="query the hybrid index directly")
    se.add_argument("query", nargs="+")
    se.add_argument("-k", type=int, default=5)
    se.add_argument("--no-rerank", action="store_true", help="skip the model reranker (faster, lower quality)")
    se.set_defaults(fn=cmd_search)

    ap = sub.add_parser("approvals", help="list actions waiting for your approval")
    ap.add_argument("--all", action="store_true", help="include approved/rejected ones")
    ap.set_defaults(fn=cmd_approvals)

    a1 = sub.add_parser("approve", help="run a queued action")
    a1.add_argument("id", type=int, nargs="+")
    a1.set_defaults(fn=cmd_approve)

    a2 = sub.add_parser("reject", help="drop a queued action")
    a2.add_argument("id", type=int, nargs="+")
    a2.set_defaults(fn=cmd_reject)

    lg = sub.add_parser("log", help="show recent runs")
    lg.add_argument("-n", type=int, default=10)
    lg.set_defaults(fn=cmd_log)

    args = p.parse_args(argv)
    try:
        args.fn(args)
    except (KeyError, ValueError, llm.LLMError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
