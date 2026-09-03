#!/usr/bin/env python3
"""Extract plain text per section from main.tex and score with Pangram.

Usage:
  python scripts/pangram_check.py extract [--from-line N]   -> writes results/pangram/units/*.txt + units.json
  python scripts/pangram_check.py score <tag> [ids...]  -> scores units (all, or listed ids), writes results/pangram/scores_<tag>.json

Reads PANGRAM_API_KEY from the environment or from .env at the repo root.
"""
import json, os, re, subprocess, sys, time, pathlib, hashlib
import requests

ROOT = pathlib.Path("/Users/joshmuthu/thesis")
TEX = ROOT / "final_report/main.tex"
AUX = ROOT / "final_report/main.aux"
GEN = ROOT / "final_report/_generated"
OUT = ROOT / "results" / "pangram"; OUT.mkdir(parents=True, exist_ok=True)
UNITS = OUT / "units"
API = "https://text.external-api.pangram.com"
KEY = os.environ.get("PANGRAM_API_KEY") or [l.split("=", 1)[1].strip() for l in (ROOT / ".env").read_text().splitlines() if l.startswith("PANGRAM_API_KEY")][0]

MACRO_PREAMBLE = r"""
\newcommand{\corrhrp}{correlation-HRP}
\newcommand{\skelhrp}{skeleton-HRP}
\newcommand{\undirhrp}{undirected-HRP}
\newcommand{\semcovhrp}{semcov-HRP}
\newcommand{\topohrp}{topo-HRP}
\newcommand{\toposemcovhrp}{topo-semcov-HRP}
\newcommand{\oriented}{oriented}
\newcommand{\floatnote}[1]{#1}
"""


def load_labels():
    labels = {}
    for m in re.finditer(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", AUX.read_text()):
        labels[m.group(1)] = m.group(2)
    return labels


def strip_env(text, env):
    return re.sub(r"\\begin\{" + env + r"\*?\}.*?\\end\{" + env + r"\*?\}", "", text, flags=re.S)


def tex_to_plain(body, labels):
    body = re.sub(r"(?<!\\)%.*", "", body)  # comments
    body = re.sub(r"\\cite[tp]?\*?(\[[^\]]*\])*\{[^}]*\}", "", body)
    body = re.sub(r"\\label\{[^}]*\}", "", body)
    body = re.sub(r"\\(?:eq|auto|c)?ref\{([^}]*)\}", lambda m: labels.get(m.group(1), "X"), body)
    for env in ("figure", "table", "sidewaystable", "tabular", "minipage"):
        body = strip_env(body, env)
    body = re.sub(r"\\(clearpage|newpage|noindent|centering|small|footnotesize|midrule|toprule|bottomrule)\b", "", body)
    body = re.sub(r"\\paragraph\{([^}]*)\}", r"\n\n\1. ", body)
    body = re.sub(r"\\(?:section|subsection|subsubsection|chapter)\*?\{[^}]*\}", "", body)
    src = MACRO_PREAMBLE + (GEN / "robust_stats.tex").read_text() + (GEN / "oos_stats.tex").read_text() + "\n" + body
    r = subprocess.run(["pandoc", "-f", "latex", "-t", "plain", "--wrap=none"], input=src, text=True, capture_output=True)
    if r.returncode:
        sys.stderr.write(r.stderr)
    txt = r.stdout
    txt = txt.replace("\u00a0", " ")
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()


def extract(from_line=1):
    lines = TEX.read_text().splitlines()
    labels = load_labels()
    # find heading lines
    heads = []
    for i, l in enumerate(lines, 1):
        m = re.match(r"\\(chapter|section|subsection)\{([^}]*)\}", l)
        if m:
            heads.append((i, m.group(1), m.group(2)))
        if l.startswith("\\end{document}"):
            heads.append((i, "end", ""))
            break
    # section number lookup via following \label
    units = []
    ch, sec, sub = 0, 0, 0
    appendix = False
    for k, (ln, kind, title) in enumerate(heads[:-1]):
        nxt = heads[k + 1][0]
        if kind == "chapter":
            ch += 1; sec = 0; sub = 0
            if title.startswith("Supplementary"):
                appendix = True; ch = "A"
            num = f"{ch}"
        elif kind == "section":
            sec += 1; sub = 0; num = f"{ch}.{sec}"
        else:
            sub += 1; num = f"{ch}.{sec}.{sub}"
        if nxt <= from_line:
            continue
        body = "\n".join(lines[ln:nxt - 1])
        txt = tex_to_plain(body, labels)
        words = len(txt.split())
        units.append(dict(id=num, kind=kind, title=title, line=ln, end=nxt - 1, words=words, text=txt))
    merged = []
    for u in units:
        if merged and merged[-1]["words"] < 40 and merged[-1]["kind"] == "section" and u["kind"] == "subsection":
            prev = merged.pop()
            u["title"] = prev["title"] + " / " + u["title"]
            u["text"] = (prev["text"] + "\n\n" + u["text"]).strip()
            u["line"] = prev["line"]
            u["words"] = len(u["text"].split())
        elif u["words"] < 40:
            continue
        merged.append(u)
    UNITS.mkdir(exist_ok=True)
    for u in merged:
        safe = re.sub(r"[^A-Za-z0-9.]+", "_", f"{u['id']}_{u['title']}")[:80]
        u["file"] = str(UNITS / f"{safe}.txt")
        pathlib.Path(u["file"]).write_text(u["text"])
    (OUT / "units.json").write_text(json.dumps([{k: v for k, v in u.items() if k != "text"} for u in merged], indent=1))
    for u in merged:
        print(f"{u['id']:>8} {u['kind']:10} L{u['line']:<5} {u['words']:5}w  {u['title']}")


def score_text(text, model="pangram-4", tries=3):
    for t in range(tries):
        try:
            r = requests.post(f"{API}/task", headers={"x-api-key": KEY}, json={"text": text, "model": model, "public_dashboard_link": False}, timeout=60)
        except requests.exceptions.RequestException as e:
            sys.stderr.write(f"submit error {e}; retrying\n"); time.sleep(10); continue
        if r.status_code == 429:
            time.sleep(5 * (t + 1)); continue
        r.raise_for_status()
        tid = r.json()["task_id"]
        for _ in range(120):
            time.sleep(2)
            try:
                g = requests.get(f"{API}/task/{tid}", headers={"x-api-key": KEY}, timeout=60).json()
            except requests.exceptions.RequestException as e:
                sys.stderr.write(f"poll error {e}; retrying\n"); time.sleep(5); continue
            if g.get("stage") == "STAGE_SUCCESS":
                return g
            if g.get("stage") == "STAGE_FAILED":
                raise RuntimeError(f"task failed: {g}")
        raise RuntimeError("timeout")
    raise RuntimeError("rate limited")


def score(ids, tag):
    units = json.loads((OUT / "units.json").read_text())
    results = {}
    cache_f = OUT / f"scores_{tag}.json"
    if cache_f.exists():
        results = json.loads(cache_f.read_text())
    for u in units:
        if ids and u["id"] not in ids:
            continue
        text = pathlib.Path(u["file"]).read_text()
        h = hashlib.md5(text.encode()).hexdigest()
        if u["id"] in results and results[u["id"]].get("hash") == h:
            continue
        res = score_text(text)
        results[u["id"]] = dict(
            id=u["id"], title=u["title"], words=u["words"], hash=h,
            fraction_ai=res["fraction_ai"], fraction_ai_assisted=res["fraction_ai_assisted"],
            fraction_human=res["fraction_human"], headline=res["headline"], prediction=res["prediction"],
            windows=[dict(label=w["label"], ai=w["ai_assistance_score"], conf=w["confidence"], hum=w["humanizer_score"],
                          words=w["word_count"], start=w["start_index"], end=w["end_index"], text=w["text"]) for w in res["windows"]],
        )
        cache_f.write_text(json.dumps(results, indent=1))
        print(f"{u['id']:>8} ai={res['fraction_ai']:.2f} asst={res['fraction_ai_assisted']:.2f} human={res['fraction_human']:.2f}  {res['headline']:<20} {u['title']}", flush=True)
    return results


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "extract":
        fl = int(sys.argv[sys.argv.index("--from-line") + 1]) if "--from-line" in sys.argv else 1
        extract(fl)
    elif cmd == "score":
        tag = sys.argv[2]
        score(sys.argv[3:], tag)
