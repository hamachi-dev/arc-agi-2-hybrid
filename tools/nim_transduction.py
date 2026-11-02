"""nim_transduction.py - direct grid prediction via NIM API (no training, no GPU).

Prompt = train pairs + test input as digit text; model outputs the grid.
Parse digit-lines from free text; score against eval solutions.
Usage: python3 tools/nim_transduction.py --tasks ID,ID --out work/nim_trans.json
"""
import argparse
import json
import re
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "tools")
import arc_induction_v2 as legc  # noqa: E402
from nim_probe import get_key, chat  # noqa: E402

SYS = ("You solve ARC grid puzzles. Study the train pairs, then output ONLY "
       "the test output grid: one row per line, digits 0-9 with no spaces, "
       "no other text.")


def parse_grid(text):
    rows = []
    for line in text.splitlines():
        s = re.sub(r"\s+", "", line)
        if s and all(c.isdigit() for c in s):
            rows.append([int(c) for c in s])
    if not rows or len(rows) > 30 or max(map(len, rows)) > 30:
        return None
    w = max(map(len, rows))
    if any(len(r) != w for r in rows):
        return None
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", default="work/nim_trans.json")
    args = ap.parse_args()
    key = get_key()
    ch = json.load(open("data/arc-agi_evaluation_challenges.json"))
    sol = json.load(open("data/arc-agi_evaluation_solutions.json"))
    out = {}
    try:
        for tid in args.tasks.split(","):
            tid = tid.strip()
            task = ch[tid]
            for ti, t in enumerate(task["test"]):
                total, hits = 0, 0
                for att in range(2):
                    total += 1
                    prompt = legc.build_prompt(
                        {"train": task["train"], "test": [t]})
                    try:
                        resp = chat(args.model, SYS, prompt, key,
                                    max_tokens=4096, temp=0.8)
                    except Exception as e:
                        print(tid, ti, "API error", str(e)[:100])
                        continue
                    msg = resp["choices"][0]["message"]
                    txt = msg.get("content") or msg.get("reasoning") or ""
                    g = parse_grid(txt)
                    if g is not None and g == sol[tid][ti]:
                        hits += 1
                out.setdefault(tid, []).append(hits > 0)
                print(tid, ti, "HIT" if hits else "miss", flush=True)
    finally:
        key = None
    with open(args.out, "w") as f:
        json.dump(out, f)
    h = sum(1 for v in out.values() for x in v if x)
    t = sum(len(v) for v in out.values())
    print(f"transduction: {h}/{t}")


if __name__ == "__main__":
    main()
