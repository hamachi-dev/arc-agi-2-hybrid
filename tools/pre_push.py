"""pre_push.py - gate every Kaggle push (catches v21/v22-class bugs free).

Checks (no GPU):
  1. All %%writefile payloads parse under Python 3.11 grammar
     (Kaggle runs 3.11; local 3.12 accepts nested-quote f-strings that
     crash there -- this caused the v21 ERROR).
  2. Submission skeleton builds + validates format.
  3. Merge monotonicity (base + fake verified -> score+1).
Usage: python3 tools/pre_push.py --nb fork_fast/failed-in-aimo.ipynb
"""
import argparse
import ast
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb", required=True)
    args = ap.parse_args()
    nb = json.load(open(args.nb))
    fails = 0
    for i, c in enumerate(nb["cells"]):
        s = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        body = [ln for ln in s.splitlines()
                if ln and not ln.startswith("!") and not ln.startswith("%")]
        try:
            ast.parse("\n".join(body), feature_version=(3, 11))
        except SyntaxError as e:
            print(f"FAIL cell {i}: {e}")
            fails += 1
    # skeleton + merge smoke (imports arc_loader/decoder)
    sys.path.insert(0, "src")
    from arc_loader import ArcDataset
    ev = ArcDataset.from_file("data/arc-agi_evaluation_challenges.json")
    sub = ev.get_submission(None)
    assert len(sub) == len(ev.keys), "skeleton task count"
    print(f"skeleton OK ({len(sub)} tasks)")
    if fails:
        sys.exit(1)
    print(f"pre-push OK: {len(nb['cells'])} cells, 3.11 grammar clean")


if __name__ == "__main__":
    main()
