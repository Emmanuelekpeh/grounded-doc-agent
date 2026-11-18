from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

from . import ask as askmod
from .index import Index
from .ingest import ingest, load_chunks

STORE = "store"


def _index(name: str, refs: bool = False) -> Index:
    chunks = load_chunks(name, STORE)
    return Index(chunks, Path(STORE) / name, use_refs=refs)


def _log(name: str, rec: dict) -> None:
    with open(Path(STORE) / name / "session.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def cmd_ingest(a):
    out = ingest(a.pdf, a.name, STORE)
    meta = json.loads((out / "meta.json").read_text())
    print(f"Ingested {a.pdf} -> {out}\n  chunks: {meta['n_chunks']}  kinds: {meta['kinds']}")
    print("Next: python -m paperworld outline", a.name, " (check sections look right before trusting answers)")


def cmd_outline(a):
    secs: "OrderedDict[tuple, dict]" = OrderedDict()
    for c in load_chunks(a.name, STORE):
        s = secs.setdefault((c["section_id"], c["section_title"]), {"n": 0, "pages": set(), "cap": 0, "eq": 0})
        s["n"] += 1
        s["pages"].update(range(c["page_start"], c["page_end"] + 1))
        s["cap"] += c["kind"] == "caption"
        s["eq"] += c["kind"] == "equation"
    for (sid, title), s in secs.items():
        pg = f"p.{min(s['pages'])}-{max(s['pages'])}"
        print(f"{sid or '-':<6} {title[:60]:<60} {s['n']:>3} chunks  {pg:<10} captions:{s['cap']} eq:{s['eq']}")


def cmd_search(a):
    idx = _index(a.name)
    print(f"[{idx.mode}]")
    for c in idx.search(a.query, k=a.k, section=a.section):
        print(f"\n{c['id']}  {askmod.where(c)}  score={c['score']}\n  {c['text'][:300]}")


def _do_ask(idx, name, question, ctx, mode, k, no_llm):
    if no_llm:
        hits = idx.search(question, k=k, section=ctx.get("section", ""))
        for c in hits:
            print(f"\n{c['id']}  {askmod.where(c)}\n  {c['text'][:400]}")
        return None
    ans = askmod.answer_question(question, idx, ctx, mode, k)
    print(askmod.render(ans))
    _log(name, {"t": time.time(), "q": question, "ctx": ctx, "mode": mode,
                "retrieved": ans.retrieved, "supported": ans.supported, "stats": ans.stats(),
                "segments": [{"label": s.label, "text": s.text,
                              "evidence": [vars(e) for e in s.evidence]} for s in ans.segments]})
    return ans


def cmd_ask(a):
    idx = _index(a.name)
    ctx = {"section": a.section, "concept": a.concept}
    _do_ask(idx, a.name, a.question, ctx, a.mode, a.k, a.no_llm)


HELP = """Pause-and-ask. Type a question (even a vague one like 'wait what').
  :at 3.2        set the section you are paused at (see 'outline')
  :concept EMA   set the concept you are on
  :mode ste      normal | ste | intuitive | math | implementation | critical
  :show c0012    print a full passage
  :help  :quit"""


def cmd_pause(a):
    try:
        import readline  # noqa: F401
    except ImportError:
        pass
    idx = _index(a.name)
    ctx = {"section": "", "concept": "", "section_title": ""}
    mode = "normal"
    titles = {c["section_id"]: c["section_title"] for c in idx.chunks if c["section_id"]}
    print(f"[{idx.mode}]  paper={a.name}\n{HELP}")
    while True:
        try:
            line = input(f"\n[{ctx['section'] or '-'} | {ctx['concept'] or '-'} | {mode}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line.startswith(":"):
            cmd, _, arg = line[1:].partition(" ")
            arg = arg.strip()
            if cmd in ("quit", "q"):
                return
            elif cmd == "help":
                print(HELP)
            elif cmd == "at":
                ctx["section"], ctx["section_title"] = arg, titles.get(arg, "")
                print(f"paused at {arg} {ctx['section_title'] or '(unknown section id)'}")
            elif cmd == "concept":
                ctx["concept"] = arg
            elif cmd == "mode":
                if arg in askmod.MODES:
                    mode = arg
                else:
                    print("modes:", ", ".join(askmod.MODES))
            elif cmd == "show":
                c = idx.by_id.get(arg)
                print(f"{askmod.where(c)}\n{c['text']}" if c else "unknown chunk id")
            else:
                print("unknown command; :help")
            continue
        try:
            _do_ask(idx, a.name, line, ctx, mode, a.k, a.no_llm)
        except Exception as e:  # keep the session alive on API/JSON errors
            print(f"error: {e}")


def cmd_eval(a):
    idx = _index(a.name)
    qs = [q.strip() for q in Path(a.questions).read_text(encoding="utf-8").splitlines() if q.strip() and not q.startswith("#")]
    results, review = [], ["# Citation spot-check\n", "Mark each claim: does the quoted passage actually support it?\n"]
    tq = tv = unsup = 0
    for q in qs:
        try:
            ans = askmod.answer_question(q, idx, {}, "normal", a.k)
        except Exception as e:
            print(f"ERROR on {q!r}: {e}")
            continue
        st = ans.stats()
        tq += st["quotes"]
        tv += st["verified"]
        unsup += not ans.supported
        results.append({"q": q, "supported": ans.supported, **st})
        review.append(f"\n## {q}\nsupported={ans.supported}  quotes verified {st['verified']}/{st['quotes']}\n")
        for s in ans.segments:
            review.append(f"- **[{s.label}]** {s.text}")
            for e in s.evidence:
                chunk = idx.by_id.get(e.chunk, {})
                review.append(f"    - [ ] {'verified' if e.verified else 'NOT FOUND'} {e.where}: \u201c{e.quote}\u201d")
                review.append(f"      > {chunk.get('text', '')[:500]}")
        print(f"{st['verified']}/{st['quotes']} quotes verified | supported={ans.supported} | {q}")
    rate = (tv / tq) if tq else float("nan")
    summary = {"questions": len(results), "quotes": tq, "verified": tv, "verified_rate": rate,
               "answered_not_supported": unsup, "retrieval": idx.mode}
    out = Path(STORE) / a.name
    (out / "eval_report.json").write_text(json.dumps({"summary": summary, "results": results}, indent=1),
                                          encoding="utf-8")
    (out / "review.md").write_text("\n".join(review), encoding="utf-8")
    print(f"\nSUMMARY {summary}\nWrote {out/'eval_report.json'} and {out/'review.md'}")
    print("verified_rate only proves quotes exist in the paper. Read review.md to judge whether they support the claims.")


def main(argv=None):
    p = argparse.ArgumentParser(prog="paperworld")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest"); s.add_argument("pdf"); s.add_argument("--name", required=True); s.set_defaults(f=cmd_ingest)
    s = sub.add_parser("outline"); s.add_argument("name"); s.set_defaults(f=cmd_outline)
    s = sub.add_parser("search"); s.add_argument("name"); s.add_argument("query")
    s.add_argument("--k", type=int, default=6); s.add_argument("--section", default=""); s.set_defaults(f=cmd_search)
    s = sub.add_parser("ask"); s.add_argument("name"); s.add_argument("question")
    s.add_argument("--section", default=""); s.add_argument("--concept", default="")
    s.add_argument("--mode", default="normal", choices=list(askmod.MODES))
    s.add_argument("--k", type=int, default=6); s.add_argument("--no-llm", action="store_true"); s.set_defaults(f=cmd_ask)
    s = sub.add_parser("pause"); s.add_argument("name")
    s.add_argument("--k", type=int, default=6); s.add_argument("--no-llm", action="store_true"); s.set_defaults(f=cmd_pause)
    s = sub.add_parser("eval"); s.add_argument("name"); s.add_argument("questions")
    s.add_argument("--k", type=int, default=6); s.set_defaults(f=cmd_eval)

    a = p.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252; answers contain checkmarks and math
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    a.f(a)


if __name__ == "__main__":
    main(sys.argv[1:])
