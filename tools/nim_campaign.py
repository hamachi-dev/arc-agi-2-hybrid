"""nim_campaign.py - offline program synthesis via NIM + local verification.

For each task: sample K programs from a strong instruct model, verify
against ALL train pairs in the local sandbox, run verified programs on
(public) test inputs, save {tid: outputs} for Kaggle merge.

Usage: python3 tools/nim_campaign.py --tasks 20270e3b,aaaabbbb --k 6 --out work/nim_pilot.json
"""
import argparse
import json
import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, "tools")
import arc_induction_v2 as legc  # noqa: E402
from nim_probe import get_key, chat  # noqa: E402


def solve_one(model, task, k, temp, key):
    prompt = legc.build_prompt(task)
    seen, verified = set(), []
    attempts = 0
    while len(seen) < k and attempts < k + 3:
        attempts += 1
        try:
            resp = chat(model, legc.SYSTEM_MSG, prompt, key,
                        max_tokens=8192, temp=temp)
        except Exception as e:
            print("  API error:", str(e)[:120])
            break
        msg = resp["choices"][0]["message"]
        txt = msg.get("content") or msg.get("reasoning") or ""
        code = legc.extract_code(txt)
        if not code or code in seen:
            continue
        seen.add(code)
        res = legc.run_verifier([code], task["train"],
                                [t["input"] for t in task["test"]],
                                4.0, "camp")[0]
        if res["status"] == "verified":
            verified.append((code, res))
    return verified


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--out", default="work/nim_pilot.json")
    args = ap.parse_args()
    key = get_key()
    ch = json.load(open("data/arc-agi_evaluation_challenges.json"))
    out = {}
    t0 = time.time()
    try:
        for tid in args.tasks.split(","):
            tid = tid.strip()
            if tid not in ch:
                print(tid, "unknown"); continue
            task = ch[tid]
            print(f"== {tid} ({len(task['train'])} pairs) ==", flush=True)
            ver = solve_one(args.model, task, args.k, args.temp, key)
            print(f"  verified: {len(ver)}", flush=True)
            if ver:
                code, res = max(ver, key=lambda cr: len(cr[0]))
                outs = [{"attempt": t, "alt": None, "votes": len(ver),
                         "n_verified": len(ver)}
                        for t in res["test_outputs"]]
                out[tid] = {"verified": True, "n_verified": len(ver),
                            "outputs": outs, "n_pairs": len(task["train"]),
                            "elapsed": 0.0, "program": code[:4000]}
            else:
                out[tid] = {"verified": False, "n_verified": 0,
                            "outputs": [], "n_pairs": len(task["train"]),
                            "elapsed": 0.0}
    finally:
        key = None
    with open(args.out, "w") as f:
        json.dump(out, f)
    nv = sum(1 for r in out.values() if r["verified"])
    print(f"pilot: {nv}/{len(out)} verified in {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
