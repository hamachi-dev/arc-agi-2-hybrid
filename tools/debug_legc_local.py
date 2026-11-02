"""debug_legc_local.py - reproduce Leg C generation locally (3B coder, 1 GPU).

Runs the REAL gen_batch -> extract_code -> run_verifier path on N eval
tasks and prints generated programs + verifier statuses, so prompt or
DSL-compliance bugs are visible without spending Kaggle quota.

Usage: python3 tools/debug_legc_local.py [--model DIR] [--tasks 2] [--k 4]
"""
import argparse
import json
import sys

sys.path.insert(0, "src")
import arc_induction_v2 as legc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/taizo/Kaggle/models/qwen25coder3b")
    ap.add_argument("--tasks", type=int, default=2)
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    print("chat_template:", "YES" if getattr(tok, "chat_template", None) else "NO")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to("cuda").eval()

    ch = json.load(open("data/arc-agi_evaluation_challenges.json"))
    items = sorted(ch.items(), key=lambda kv: legc.est_cost(kv[1]))[:args.tasks]
    for tid, task in items:
        print("=" * 70)
        print("TASK", tid, "train pairs:", len(task["train"]))
        prompt = legc.build_prompt(task)
        print("prompt chars:", len(prompt))
        texts = legc.gen_batch(model, tok, prompt, args.k, 0.8, 512)
        codes = []
        for t in texts:
            c = legc.extract_code(t)
            print("-" * 40)
            print("RAW REPLY:", t[:600])
            print("EXTRACTED:", (c[:800] if c else "<EMPTY>"))
            if c:
                codes.append(c)
        res = legc.run_verifier(codes, task["train"],
                                [t["input"] for t in task["test"]],
                                4.0, "localdbg")
        for c, r in zip(codes, res):
            print(f"STATUS={r['status']} pairs={r.get('pairs_passed')}/"
                  f"{r.get('n_pairs')} err={r.get('error', '')[:120]}")


if __name__ == "__main__":
    main()
