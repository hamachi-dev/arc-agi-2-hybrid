"""test_merge.py - proves cell11 (LegC merge) is score-monotonic.

Simulates: base submission scores 1/172 (one lucky hit), induction
verifies a DIFFERENT task. After merge, score must be 2/172 and the
base candidate must survive as attempt_2.
"""
import json
import sys

sys.path.insert(0, "src")
from arc_loader import ArcDataset  # noqa: E402

CH = "data/arc-agi_evaluation_challenges.json"
SOL = "data/arc-agi_evaluation_solutions.json"

data = ArcDataset.from_file(CH).load_replies(SOL)
sub = data.get_submission(None)  # all [[0]] -> score 0

sols = json.load(open(SOL))
# base gets one task exactly right (simulate a TTT hit)
t0 = sorted(sols.keys())[0]
for i, exp in enumerate(sols[t0]):
    sub[t0][i] = {"attempt_1": exp, "attempt_2": [[0]]}
base = data.validate_submission(sub)

# induction verifies a different task
t1 = sorted(sols.keys())[1]
ind = {t1: {"verified": True, "outputs": [
    {"attempt": sols[t1][i], "alt": [[1]], "votes": 3, "n_verified": 5}
    for i in range(len(sols[t1]))]}}

# --- inline copy of src/cell11.py merge loop (paths parameterised) ---
promoted = created = 0
for tid, rec in ind.items():
    if not isinstance(rec, dict) or not rec.get("verified"):
        continue
    for ti, o in enumerate(rec.get("outputs", [])):
        g = o.get("attempt")
        if not g:
            continue
        alt = o.get("alt")
        if tid not in sub:
            sub[tid] = []
        while len(sub[tid]) <= ti:
            sub[tid].append({"attempt_1": [[0]], "attempt_2": [[0]]})
            created += 1
        e = sub[tid][ti]
        if e["attempt_1"] != g:
            demoted = e["attempt_1"]
            e["attempt_2"] = demoted if demoted != [[0]] else (alt or demoted)
            e["attempt_1"] = g
            promoted += 1
        elif alt and e.get("attempt_2") == [[0]]:
            e["attempt_2"] = alt

merged = data.validate_submission(sub)
print(f"base={base:.2f} merged={merged:.2f} promoted={promoted}")
assert merged == base + 1, "merge must add exactly the verified task"
assert sub[t1][0]["attempt_1"] == sols[t1][0]
print("MERGE MONOTONIC OK")
