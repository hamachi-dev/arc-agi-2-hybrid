# cell 11
# ===================== Leg C merge (run AFTER the submission cell) =====================
# Monotonic union: train-verified induction outputs pre-empt attempt_1, the
# base candidate is demoted to attempt_2 (cross-family decorrelation).
# Tasks without a verified program are left untouched.
import os, json

SUB = "submission.json"
IND = "/kaggle/working/induction_results.json"

with open(SUB) as f:
    sub = json.load(f)
ind = {}
if os.path.exists(IND):
    with open(IND) as f:
        ind = json.load(f)

rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")

def local_score(s):
    try:
        return data.validate_submission(s)   # `data` from the submission cell
    except Exception:
        return None

before = None if rerun_mode else local_score(sub)

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

with open(SUB, "w") as f:
    json.dump(sub, f)

after = None if rerun_mode else local_score(sub)
print(f"[LegC merge] promoted={promoted} created={created}")
if before is not None and after is not None:
    print(f"[LegC merge] local score: base {before:.2f} -> merged {after:.2f} (unit = tasks)")

