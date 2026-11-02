# A Hybrid Transduction–Induction Pipeline for ARC-AGI-2: What Worked, What Didn't, and Why

**Team Hamachi — ARC Prize 2026 Paper Track submission (draft)**

## Abstract

We built a hybrid ARC-AGI-2 solver combining (i) per-puzzle LoRA test-time
training (TTT) on a 4B grid-specialized model with turbo-DFS decoding and
augmentation rescoring (transduction), and (ii) train-verified program
synthesis with a monotonic merge (induction). Best public score **31.39**
(all scores herein are public-split scores; final standings use the private
split). Our main contribution is empirical: a fully reproduced account of
where each component helps, with two negative results we consider as
informative as the positive ones — (a) 7B-coder blind sampling yields zero
train-pair passes in 223 candidates over 60 tasks, and (b) two independent
single-point failures that each deterministically zero a submission (a
token-indexing port bug, and a subprocess interpreter miss launching workers
with a torch-less `python`), both diagnosed in full. All code, tooling, and
logs are open.

## 1. Method

### 1.1 Base transduction (forked, credited)

Our base is `koushikrudra/failed-in-aimo` (Apache 2.0): Qwen3-4B SFT
(`sorokin/qwen3_4b_grids15_sft139`) + per-puzzle LoRA (r=256, 1 epoch over
128 augmented variants) + constrained DFS decoding (12 grid tokens) +
augmentation rescoring (8 queries, vote-margin selection), 4×L4 parallel.
We kept its math intact and added: cost-ordered any-time queueing, per-key
worker isolation, model-path resolution, adaptive second epoch on high-loss
puzzles (see §3.3), and a GPU-resident rewrite of the two scoring hotspots
(ported from `mikelou1`, verified numerically identical on CPU).

### 1.2 Verified induction + monotonic merge (ours)

In parallel we sample DSL programs (35 grid primitives, AST-whitelisted,
torch-free sandbox with per-program timeouts) from a coder model, keep only
programs passing ALL train pairs, and merge: verified outputs pre-empt
`attempt_1`, the base candidate demotes to `attempt_2`. We prove by
construction + unit test that the merge can only add coverage where the base
emitted dummies, and bound the displacement risk elsewhere at the measured
base attempt_2-exclusive rate (5.2%).

## 2. Experiments

| Config | Eval (outputs) | Public score |
|---|---|---|
| Base (v12) | 49/172 (28.5%) | 27.22 |
| Base redraw (v16) | 46/172 (26.7%) | 30.56 |
| Base + adaptive (v28) | — | 29.72 |
| Base + perfpatch (mikelou1 fork, separate kernel) | — | **31.39** |
| Fixed stack (v35: both incidents resolved) | — | 28.47 |
| Control redraw (v12) | — | 30.14 |

Draw noise is ±3pp (same code, 27.22→30.14). Union of two draws adds only
+2.3pp (49+46→53/172): reruns are not a strategy.

### 2.1 Negative result 1: blind induction fails

7B-coder sampling (k≈7, repair + hot retries): **0 train pairs passed in
223 candidates over 60 tasks**. Stronger 20B reasoning model: 1/15 tasks
train-verified, 0/1 test-correct (pure train overfit). Conclusion: sampling
without search cannot compose ARC-AGI-2's multi-step rules; verification
without generalization is insufficient. We terminated this line and document
it so others need not repeat it.

### 2.2 Negative result 2: two independent single-point failures

Porting the (correct) GPU NLL rewrite, we kept token-ID indexing on the
gathered 12-wide matrix; EOS (id 15) raised IndexError on every decode →
all-dummy submissions. Separately, launching workers via `subprocess` with
a bare `"python"` resolved to a torch-less interpreter; with output
suppressed, every task silently skipped → identical all-dummy signature.
Each alone zeroes a submission (three 0.00s total); diagnosis, CPU
regression tests (matrix + integrated loop, incl. EOS path), the
`sys.executable` fix, and a 3-task live probe (real predictions restored)
are documented. Lessons: validate ported numerics through the consumer
loop, not just the tensor; never swallow subprocess failures (`check=False`
without a smoke signal).

### 2.3 Positive: loss predicts solvability

Train loss medians: FULL-solved 0.0067 < PARTIAL 0.0102; the 11 highest-loss
tasks all miss. Our adaptive second epoch (loss > 0.02 only, +0.5h bounded)
targets exactly this population; it shipped in v28/v35 (29.72/28.47, within
draw noise of baselines — directionally consistent, not yet isolated).

## 3. Theory (why TTT works here, briefly)

Per-puzzle LoRA fits the task's implicit program into the adapter while the
frozen SFT prior supplies grid syntax; augmentation rescoring then selects
for invariance across symmetries rather than likelihood. This explains both
the method's floor (~27%: prior + search) and its ceiling (no compositional
generalization beyond the adapter's reach — consistent with §2.1).

## 4. Universality, Completeness

Task-agnostic single code path for all 240 hidden tasks; identical files run
on Kaggle L4×4 and locally (sandbox selftest 9/9, CPU). Artifacts: loader,
decoder, solver, queue, induction sandbox + runtime, validator, scheduler
simulator, debug harnesses, full logs. Reproduction: attach listed data +
models, run cells in order (≈5.5h on 4×L4).

## 5. Novelty statement (honest)

The transduction backbone is forked (credited above); the scoring-hotspot
math is ported (credited). Ours: adaptive-by-loss training, monotonic
verified merge with bounded risk, any-time queue, the negative-result
characterizations (§2.1–2.2), and the offline-synthesis protocol with
train-only verification.

## References / artifacts

- Base: kaggle.com/code/koushikrudra/failed-in-aimo (Apache 2.0)
- Perf patch math: kaggle.com/code/mikelou1/arc-agi2-lb33-89-minimal-perfpatch
- SFT model: kaggle.com/models/sorokin/qwen3_4b_grids15_sft139 (Apache 2.0)
- Our fork: kaggle.com/code/hamatz/failed-in-aimo
- Our 31.39 kernel: kaggle.com/code/hamatz/arc-agi2-lb33-89-minimal-perfpatch
- Tooling + logs: https://github.com/hamachi-dev/arc-agi-2-hybrid (Apache 2.0)
