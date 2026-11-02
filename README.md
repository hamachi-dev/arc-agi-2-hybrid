# ARC-AGI-2 Hybrid: TTT + Verified Induction (Team Hamachi)

Kaggle ARC Prize 2026 (ARC-AGI-2) entry. Best public score **31.39**.
Paper Track submission companion repo (CC-BY-4.0 for the paper; code below
under Apache-2.0, see LICENSE).

## Credits (upstream, unmodified logic)

- Base pipeline: `koushikrudra/failed-in-aimo` (Apache 2.0)
- Scoring-hotspot math: `mikelou1/arc-agi2-lb33-89-minimal-perfpatch`
- SFT model: `sorokin/qwen3_4b_grids15_sft139` (Apache 2.0, Kaggle Models)
- Competition data: ARC Prize 2026 (Apache 2.0, Kaggle competition page)

## Ours

- `src/`: patched solver/loader/decoder/queue + induction sandbox
  (`arc_induction_v2.py`), model resolver, adaptive 2nd epoch, any-time queue
- `tools/`: validator, scheduler simulator, perfpatch equivalence tests,
  pre-push gate (Python 3.11 grammar), local debug harnesses, NIM/Go
  campaign scripts (need your own API keys; never committed)
- `docs/`: paper draft + writeup outline
- `evidence/`: scored submissions + analysis inputs (small, derived)

## Reproduce (Kaggle, 4×L4, ~5.5h)

1. Attach: competition `arc-prize-2026-arc-agi-2`, model
   `sorokin/qwen3_4b_grids15_sft139` (Transformers/bfloat16/1).
2. Run the notebook cells in order (`hamatz/failed-in-aimo` v35+ logic).
3. `submission.json` is produced; validate with
   `python3 tools/validate_submission.py --challenges <eval|test> --submission submission.json [--solutions ...]`.

## Results (public split)

| Config | Public score |
|---|---|
| Base | 27.22 |
| Base redraws | 30.56 / 30.14 |
| Base + adaptive | 29.72 |
| Base + perfpatch | **31.39** |
| Fixed stack (v35) | 28.47 |

See `docs/paper_draft.md` for the full account incl. negative results.
