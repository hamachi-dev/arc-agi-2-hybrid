"""infer_only.py - base-model greedy inference, NO training (tests H-A).

If the SFT base + formatting already solves tasks at ~Kaggle rates,
per-puzzle LoRA training is vacuous and strategy pivots to decode-side.
Usage: ./.venv-ttt/bin/python tools/infer_only.py --tasks ID,ID --max-new 931
"""
import argparse
import json
import sys

sys.path.insert(0, "src")
from arc_loader import ArcDataset, QwenFormatter  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/taizo/Kaggle/models/qwen3_4b_sft")
    ap.add_argument("--tasks", default="0934a4d8,135a2760,13e47133,142ca369,16de56c4")
    ap.add_argument("--max-new", type=int, default=931)
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16).to("cuda").eval()
    fmt = QwenFormatter(tok)
    ch = json.load(open("data/arc-agi_evaluation_challenges.json"))
    sol = json.load(open("data/arc-agi_evaluation_solutions.json"))
    hits = total = 0
    for tid in args.tasks.split(","):
        task = ch[tid]
        for ti, t in enumerate(task["test"]):
            total += 1
            prompt = fmt.fmt_train(task["train"]) + fmt.fmt_query([t])
            ids = tok.encode(prompt, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                out = model.generate(ids, max_new_tokens=args.max_new,
                                     do_sample=False,
                                     eos_token_id=tok.eos_token_id,
                                     pad_token_id=tok.eos_token_id)
            gen = out[0][ids.shape[1]:].tolist()
            arr = fmt.convert_tokens_to_array(gen + [tok.eos_token_id])
            import numpy as _np
            exp = _np.asarray(sol[tid][ti])
            ok = (arr is not None and arr.shape == exp.shape
                  and bool((arr == exp).all()))
            shape = None if arr is None else tuple(arr.shape)
            hits += int(ok)
            print(f"{tid}[{ti}] {'HIT' if ok else 'miss'} shape={shape} exp={tuple(exp.shape)}", flush=True)
    print(f"greedy inference-only: {hits}/{total}", flush=True)


if __name__ == "__main__":
    main()
