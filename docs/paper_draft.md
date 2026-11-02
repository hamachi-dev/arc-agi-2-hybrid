# A Practical Study of Test-Time Training and Verified Program Synthesis for ARC-AGI-2

**Team Hamachi (hamatz, Kaggle) — ARC Prize 2026 Paper Track submission (draft v2)**

## Abstract

We report a practical study of two approaches to ARC-AGI-2: (i) per-puzzle
LoRA test-time training (TTT) on a 4B grid-specialized model with turbo-DFS
decoding and augmentation rescoring (transduction — the effective
baseline), and (ii) train-verified program synthesis with a monotonic merge
(induction — investigated thoroughly, did not contribute to any scored
submission). Best public score **31.39** (all scores herein are public-split
scores: exact test-output matches divided by public outputs; final
standings use the private split). Our contribution is empirical: ablations
as far as the data allows, two fully diagnosed engineering failures, and
all code, tooling, and logs open. Our headline finding is a bounded one:
the tested induction approach fails systematically, and we document exactly
how, so others need not repeat it.

## 1. Method

### 1.1 Base transduction (forked, credited)

Base: `koushikrudra/failed-in-aimo` (Apache 2.0): Qwen3-4B SFT
(`sorokin/qwen3_4b_grids15_sft139`) + per-puzzle LoRA (r=256, 1 epoch over
128 augmented variants, lr 5e-5, seed 42 throughout) + constrained DFS
decoding (12 grid tokens, ~931 max new tokens) + augmentation rescoring
(8 queries, vote-margin selection), 4×L4 parallel, ~5.4h per 240-task run.
We kept its math intact and added: cost-ordered any-time queueing, per-key
worker isolation, model-path resolution, and an adaptive second epoch on
high-loss puzzles (see §2.3).

### 1.2 Scoring-hotspot rewrite (ported, credited, verified)

We ported the GPU-resident rewrite of the two scoring hotspots from
`mikelou1/arc-agi2-lb33-89-minimal-perfpatch`: NLL over ARC tokens computed
on-GPU with gather (vs full-vocab CPU log-softmax), and teacher-forced
rescoring with `use_cache=False` and target-only gather. Three distinct
claims, kept separate: (i) *scoring equivalence* — proven numerically
identical on CPU (random logits, atol 1e-4/1e-5, incl. the EOS path);
(ii) *output equivalence* — NOT established (no same-input comparison run);
(iii) *runtime improvement* — expected from reduced host transfer, not
measured end-to-end. Score deltas involving the perfpatch are therefore
draw noise unless stated otherwise.

### 1.3 Verified induction + merge (ours; did not score)

We sample DSL programs (35 primitives: grid ops, color ops, assembly,
objects, symmetry/physics-ish helpers; AST-whitelisted, no imports/loops/
classes; torch-free subprocess sandbox, 4s per-program timeout) from a
coder model, keep only programs passing ALL train pairs, and merge verified
outputs into `attempt_1` (base candidate demoted to `attempt_2`). Design
rationale and risk bound in §2.2.

## 2. Experiments

Seeds fixed at 42 for initialization, data shuffling, and augmentation
(all `seed=42` / `random_state=42` settings). This does NOT make runs
bit-identical: GPU kernels (unsloth fused ops, bf16 reductions) are
nondeterministic, and DFS sampling interacts with that noise — hence
±3pp across redraws of identical code (27.22→30.14). We report every draw
instead of selecting: submission IDs below.

| Config | Eval (172 outputs) | Public score |
|---|---|---|
| Base | 49 (28.5%) | 27.22 |
| Base redraw | 46 (26.7%) | 30.56 / 30.14 |
| Base + adaptive | — | 29.72 |
| Base + perfpatch (own fork of mikelou1) | — | **31.39** |
| Base + perfpatch + adaptive (fixed stack) | — | 28.47 |
| Zero-score incidents (v30/v31/v32) | — | 0.00 (refs 57726470/57786595/57820867) |

Submission-ID mapping: base 57559497 (v12); base redraws 57589119 (v16),
57754454 (control); adaptive 57617622 (v28); perfpatch-fork 57680657;
fixed stack 57854842 (v35); zeros above. All queryable on the competition
submissions page.

Reading guide: eval (172 local outputs) and public LB (hidden-test subset)
are different populations — compare within columns only. All public scores
27–31 lie within observed draw noise (±3pp on redraws: 27.22→30.14 same
code). The perfpatch is math-identical by construction, so 31.39 vs 27.22
is a draw difference, not a method effect. Union of two base draws adds
only +2.3pp (49+46→53/172): reruns are not a strategy.

Ablation status: Base / +Perfpatch / +Induction / +Both — induction added
zero verified outputs in every run, so +Induction ≡ Base everywhere
measured; +Perfpatch ≡ Base by construction (CPU proof). The honest summary
is a single distribution at ~27–31% with more than one way to sample it.

### 2.1 Induction fails systematically (bounded claim)

Under the tested configuration (7B coder, k≈4–8, temp 0.8–1.0, repair +
hot retries, up to ~5 rounds, 35-primitive DSL): **0 fully train-verified
programs in 305 candidates over the two 30-task smoke runs (112 + 193), and
none in the full 120-task run either** (i.e., no candidate passed ALL train
pairs of its task; best_partial was 0 on all 60 smoke tasks measured in
detail). With a stronger 20B reasoning model: 1/15
tasks train-verified, 0/1 test-correct (pure train overfit — e.g., hardcoded
triplet maps passing 2 train pairs, failing test). Statuses were ~99%
wrong-output (not crashes/timeouts): programs run, they just never match
even one pair. We claim only this: blind sampling under the tested
configuration failed to produce train-verified programs; whether search
would help is untested. Recorded details: per-task candidate counts (2–8),
best_partial = 0 on all 30 cheapest tasks, dedup rates, sample programs
(saved). We terminated this line.

### 2.2 Two independent single-point failures (both diagnosed)

(a) Token indexing: keeping token-ID indexing on the gathered 12-wide
matrix makes EOS (id 15) raise IndexError on every decode (vocab has 16
entries 0–15; gathered columns are positional 0–11). Fix: positional
indexing (one line), plus CPU regression test now covering the consumer
loop incl. the EOS path. (b) Subprocess interpreter: launching workers via
`subprocess` with bare `"python"` resolved to a torch-less interpreter;
with output suppressed, every task silently skipped. Fix: `sys.executable`
(proven by a 3-task live probe yielding real predictions), plus a
pre-push gate (3.11 grammar, skeleton, merge tests). Each bug alone zeroes
a submission (three 0.00s); the fixed stack scores normally (28.47).
Lesson: validate ported numerics through the consumer loop, and never
swallow subprocess failures.

### 2.3 Training loss and observed solvability (hypothesis, not claim)

Task-level aggregation (n=120 tasks; the 172 outputs belong to these tasks,
so the two views differ only in weighting). Train-loss medians: FULL-solved
(n=31) 0.0067 vs PARTIAL (n=11) 0.0102 vs ZERO (n=78) 0.0096; the 11
highest-loss tasks all miss (Spearman loss-vs-hits −0.17, a weak signal;
cost-vs-hits −0.02, so size does not explain it). Threshold 0.02 was chosen
as the visible elbow of sorted losses (≈15–18 tasks affected); the adaptive
second epoch (+0.5h bounded) ran inside the 28.47/29.72 submissions but was
never isolated — we report it as a hypothesis with supporting correlation,
not an established effect. Wording throughout: "correlates", never "predicts".

### 2.4 Merge design (scoped)

The merge promotes train-verified outputs to `attempt_1`, demoting base
`attempt_1` to `attempt_2` (base `attempt_2` is displaced). Three distinct
quantities: train verification (observed), test generalization (unknown —
our only verified program failed test), displacement loss (bounded by the
measured base attempt_2-exclusive rate, 5.2%, 9/172). Monotonicity holds
only where the base emitted dummies; elsewhere it is an expected-value bet
we did not get to settle (induction never verified on scored runs).

## 3. Theory (brief, scoped)

Per-puzzle LoRA fits the task's implicit program into the adapter while the
frozen SFT prior supplies grid syntax; augmentation rescoring then selects
for invariance across symmetries rather than likelihood. Consistent with
the observed floor (~27%), the loss correlation (§2.3), and the induction
failure (§2.1: no compositional generalization from sampling). Stated as an
interpretation, not a theorem.

## 4. Reproducibility and runtime coverage

Task-agnostic single code path for all tasks; sandbox selftest 9/9 (CPU).
Environment: Kaggle 4×L4, Python 3.11, unsloth+flash-patch bundle per
notebook metadata (exact pins in kernel metadata), SFT revision
`sorokin/qwen3_4b_grids15_sft139` Transformers/bfloat16/1, seeds fixed 42
(see nondeterminism note in §2); ~5.4h per 240-task run, i.e. ~81
wall-clock s/task (~322 GPU-s/task across 4×L4); weekly quota 108k GPU-s.
Artifacts: loader, decoder, solver, queue, induction sandbox + runtime, validator,
scheduler simulator, debug harnesses, full logs, per-task analyses.
Reproduction: attach listed data + models, run cells in order. Public-score
history with submission IDs is in §2.

## 5. Novelty statement (scoped)

Backbone forked (credited); hotspot math ported (credited). Ours: (1) a
verified-program integration design with bounded displacement risk; (2)
adaptive-by-loss compute allocation (hypothesis stage); (3) failure
characterizations — induction non-results under stated conditions and two
independently fatal port bugs with regression tests. (1) and (2) lack
isolated effect sizes, stated plainly above.

## 6. Related work (compact)

ARC-AGI-2 (Chollet et al., arXiv:2505.11831, 2025) extends ARC-AGI-1 with
harder compositional tasks. The NVARC 2024/2025 winning solutions
established per-task test-time training as the competitive backbone; our
transduction line descends from the public `failed-in-aimo` notebook
(K. Rudra, Kaggle, 2026) with the scoring-hotspot optimization of `mikelou1`
(Kaggle, 2026). Program synthesis for ARC spans enumerative DSL search and
recent LLM-guided attempts; ours differs in requiring full train-set
verification before use, and in reporting its failure quantitatively
(§2.1). Test-time training as a paradigm follows Sun et al. (ICML 2020,
arXiv:1909.13231); in-context adaptation theory follows Akyürek et al.
(arXiv:2211.15661, 2022). Our adaptive-by-loss scheduling (§2.3) is a small,
task-dependent extension, unevaluated in isolation.

## References / artifacts

- Base: kaggle.com/code/koushikrudra/failed-in-aimo (Apache 2.0)
- Perf math: kaggle.com/code/mikelou1/arc-agi2-lb33-89-minimal-perfpatch
- SFT model: kaggle.com/models/sorokin/qwen3_4b_grids15_sft139 (Apache 2.0)
- Our kernels: kaggle.com/code/hamatz/failed-in-aimo,
  kaggle.com/code/hamatz/arc-agi2-lb33-89-minimal-perfpatch
- Code + tooling + logs: https://github.com/hamachi-dev/arc-agi-2-hybrid (Apache 2.0)
