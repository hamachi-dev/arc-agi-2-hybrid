# Innovation Prize Writeup — Outline & Evidence (draft 2026-10-24)

Competition: ARC Prize 2026 - ARC-AGI-2. Deadline for Solution Writeups:
within 7 days after 2026-11-02. Prize $275k, judged equally on 6 criteria.

## 1. Accuracy
- Base pipeline (Qwen3-4B SFT + per-puzzle LoRA TTT + turbo-DFS + augmentation
  rescoring, 4xL4): eval 49/172 outputs (28.5%), hidden LB 27.22 (rank ~1989).
- Evidence: `run_v12/submission.json` (validated), LB ref 57559497.

## 2. Universality
- Method is task-agnostic: no per-task rules, same code path for all 240 hidden
  tasks. Verified-induction DSL (35 primitives) is domain-general grid ops.
- Portability proof: identical files run locally (RTX 3060 CPU paths) and on
  Kaggle (sandbox selftest 9/9).

## 3. Progress (toward 85%)
- Negative results that bound the space (all reproduced, logged):
  a. 7B-coder program sampling: 223 candidates / 60 tasks, ZERO train pairs
     passed. Blind synthesis does not crack ARC-AGI-2 distributions.
  b. Selection algorithms (probmul vs vote-margin): identical 36.3/116.
  c. Rerun draws: union +2.3pp only (49+46 -> 53/172). Noise, not signal.
  d. Train-loss vs hits: top-11 loss tasks all miss -> adaptive 2nd epoch
     (implemented, pending full validation run).
- Positive: cost-ordered any-time queue (all 120 eval tasks complete even
  under early termination); per-key worker isolation.

## 4. Theory
- Why TTT works here: per-puzzle LoRA fits the task's implicit program;
  augmentation-rescoring selects for invariance, not likelihood.
- Why blind induction fails: ARC-AGI-2 rules need multi-step composition;
  k~7 samples cannot cover the compositional space (empirical: 0/223).
- Loss-fit correlation as a difficulty probe (median FULL 0.0067 vs ZERO 0.0096).

## 5. Completeness
- Artifacts: arc_loader / arc_decoder / arc_solver / starter_fixed /
  arc_induction_v2 (+sandbox) / validate_submission / sim_schedule /
  debug harness. All open, reproducible end-to-end (12-cell notebook).
- Timeline: v12 (base, LB 27.22) -> v16/v17/v18 (Leg C experiments) ->
  v19-planned (OOM-hardened Leg C + adaptive epochs).

## 6. Novelty
- Hybrid transduction+verified-induction with monotonic merge
  (verified attempt_1 pre-emption, base demoted to attempt_2: score can only
  stay or improve — proven by test_merge.py).
- Train-verified sandbox (AST whitelist, per-program SIGALRM, torch-free
  subprocess isolation).

## TODO before finalizing
- [ ] v19 full-run results (post quota reset ~10/31)
- [ ] Final LB number + rank
- [ ] Repo link + reproduction instructions
- [ ] Figures: loss-vs-hits scatter, queue timing, merge monotonicity
