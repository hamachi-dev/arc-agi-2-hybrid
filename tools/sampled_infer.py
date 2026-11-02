"""sampled_infer.py - multi-sample decoding with UNTRAINED base (H-A test).

TTT training is vacuous locally (all-masked labels, loss 0). If sampled
decoding alone matches Kaggle per-task hits, training adds nothing and
strategy pivots to decode-side. Else training matters somehow.
Usage: ./.venv-ttt/bin/python tools/sampled_infer.py (background, ~40 min)
"""
import json
import sys

sys.path.insert(0, "src")
from arc_loader import ArcDataset, QwenFormatter  # noqa: E402

TASKS = ["0934a4d8", "135a2760", "13e47133", "142ca369", "16de56c4"]
N = 8


def main():
    import numpy as _np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/home/taizo/Kaggle/models/qwen3_4b_sft")
    model = AutoModelForCausalLM.from_pretrained(
        "/home/taizo/Kaggle/models/qwen3_4b_sft",
        dtype=torch.bfloat16).to("cuda").eval()
    fmt = QwenFormatter(tok)
    ch = json.load(open("data/arc-agi_evaluation_challenges.json"))
    sol = json.load(open("data/arc-agi_evaluation_solutions.json"))
    base = json.load(open("run_v12/submission.json"))
    out = {}
    for tid in TASKS:
        task = ch[tid]
        out[tid] = {}
        for ti, t in enumerate(task["test"]):
            prompt = fmt.fmt_train(task["train"]) + fmt.fmt_query([t])
            ids = tok.encode(prompt, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                outs = model.generate(
                    ids.repeat(N, 1), max_new_tokens=fmt.max_new_tokens(),
                    do_sample=True, temperature=0.8, top_p=0.95,
                    eos_token_id=tok.eos_token_id,
                    pad_token_id=tok.eos_token_id)
            exp = _np.asarray(sol[tid][ti])
            hit = False
            for s in range(N):
                arr = fmt.convert_tokens_to_array(
                    outs[s][ids.shape[1]:].tolist() + [tok.eos_token_id])
                if arr is not None and arr.shape == exp.shape and bool((arr == exp).all()):
                    hit = True
                    break
            p = base.get(tid, [])
            khit = ti < len(p) and (p[ti].get("attempt_1") == sol[tid][ti]
                                    or p[ti].get("attempt_2") == sol[tid][ti])
            out[tid][ti] = {"local8": hit, "kaggle": khit}
            print(f"{tid}[{ti}] local8={'HIT' if hit else 'miss'} kaggle={'HIT' if khit else 'miss'}",
                  flush=True)
    with open("work/sampled_infer.json", "w") as f:
        json.dump(out, f)
    lh = sum(v for t in out.values() for v in [x["local8"] for x in t.values()])
    kh = sum(v for t in out.values() for v in [x["kaggle"] for x in t.values()])
    print(f"local-sampled: {lh} vs kaggle-TTT: {kh} (same tasks)", flush=True)


if __name__ == "__main__":
    main()
