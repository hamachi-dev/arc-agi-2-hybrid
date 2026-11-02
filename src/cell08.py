# cell 8
# ===================== Leg C launch (BEFORE base pipeline) =====================
# Verified-induction pre-pass (Qwen2.5-Coder-7B-Instruct, 4 x L4 workers).
# Base is near time-saturated on the hidden set, so Leg C wall time is NOT free:
# keep LEGC_BUDGET_H small; the starter-level skip refunds verified tasks.
# arc_induction_v2.py exits safely (empty results) if the budget is < 5 min,
# if the coder model is not attached, or on any internal failure -> the merge
# becomes a no-op and the base score is untouched.
LEGC_BUDGET_H  = 1.0   # 0 disables Leg C (safe no-op)
BASE_RESERVE_H = 9.8   # wall hours guaranteed to remain for the base pipeline

LEGC_END_TS = f"{global_end_time - BASE_RESERVE_H * 3600:.0f}"
LEGC_BUDGET = f"{LEGC_BUDGET_H:.3f}"

!python arc_induction_v2.py --budget-h {LEGC_BUDGET} --end-ts {LEGC_END_TS}

