"""validate_submission.py - official-metric checker for ARC-AGI-2 submissions.

Checks BOTH halves of a valid entry:
  1. FORMAT: every challenge task_id present, exactly the right number of
     test outputs in order, attempt_1/attempt_2 present, rectangular int
     grids with values 0-9 and sides 1..30.
  2. SCORE (needs solutions file): per test output, 1.0 if EITHER attempt
     matches exactly, else 0.0; averaged over all test outputs.
     (matches the Evaluation page definition)

Usage:
    python3 tools/validate_submission.py \
        --challenges ../data/arc-agi_evaluation_challenges.json \
        --submission /tmp/skel.json \
        --solutions ../data/arc-agi_evaluation_solutions.json
"""
import argparse
import json
import sys


def check_grid(g):
    if not isinstance(g, list) or not g or len(g) > 30:
        return "not a non-empty list of <=30 rows"
    w = None
    for row in g:
        if not isinstance(row, list) or not row or len(row) > 30:
            return "row not a non-empty list of <=30 cells"
        if w is None:
            w = len(row)
        elif len(row) != w:
            return "ragged rows"
        for c in row:
            if not isinstance(c, int) or not (0 <= c <= 9):
                return f"bad cell {c!r} (need int 0..9)"
    return None


def validate(sub, challenges):
    errors = []
    if set(sub.keys()) != set(challenges.keys()):
        errors.append(
            f"task_id mismatch: missing={sorted(set(challenges) - set(sub))[:5]}"
            f" extra={sorted(set(sub) - set(challenges))[:5]}")
    for tid, task in challenges.items():
        want = len(task["test"])
        got = sub.get(tid)
        if not isinstance(got, list) or len(got) != want:
            errors.append(f"{tid}: want {want} outputs, got "
                          f"{len(got) if isinstance(got, list) else type(got)}")
            continue
        for i, entry in enumerate(got):
            for att in ("attempt_1", "attempt_2"):
                if not isinstance(entry, dict) or att not in entry:
                    errors.append(f"{tid}[{i}]: missing {att}")
                    continue
                err = check_grid(entry[att])
                if err:
                    errors.append(f"{tid}[{i}].{att}: {err}")
    return errors


def score(sub, sols):
    total, hit = 0, 0.0
    for tid, expected in sols.items():
        preds = sub.get(tid, [])
        for i, exp in enumerate(expected):
            total += 1
            if i < len(preds):
                e = preds[i]
                if e.get("attempt_1") == exp or e.get("attempt_2") == exp:
                    hit += 1
    return hit, total, (hit / total if total else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--challenges", required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--solutions", default=None)
    args = ap.parse_args()

    challenges = json.load(open(args.challenges))
    sub = json.load(open(args.submission))
    errors = validate(sub, challenges)
    if errors:
        print(f"FORMAT FAIL ({len(errors)} errors, first 10):")
        for e in errors[:10]:
            print("  -", e)
        sys.exit(1)
    print(f"FORMAT OK: {len(sub)} tasks, "
          f"{sum(len(t['test']) for t in challenges.values())} test outputs")

    if args.solutions:
        sols = json.load(open(args.solutions))
        hit, total, acc = score(sub, sols)
        print(f"SCORE: {hit:.0f}/{total} = {acc*100:.2f}%")


if __name__ == "__main__":
    main()
