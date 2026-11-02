"""go_campaign.py - program synthesis via opencode-go (Responses API).

Same verify-then-keep protocol as nim_campaign.py. Uses the user's own
zen-go.key (never printed). Model default: muse-spark-1.3-contributor
($0.10/$0.20 per 1M — a full 240-task campaign costs on the order of $2).
Usage: python3 tools/go_campaign.py --tasks ID,ID --k 6 --out work/go_pilot.json
"""
import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, "src")
import arc_induction_v2 as legc  # noqa: E402

API = "https://opencode.ai/inference/go/openai/v1/responses"


def get_key():
    with open("/home/taizo/.config/opencode/secrets/zen-go.key") as f:
        return f.read().strip()


def ask(model, system, user, key, max_tokens=4096, temp=0.8):
    body = json.dumps({
        "model": model,
        "instructions": system,
        "input": user,
        "max_output_tokens": max_tokens,
        "temperature": temp,
    }).encode()
    req = urllib.request.Request(
        API, data=body,
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json",
                 "User-Agent": "agi2-pilot/1.0",
                 "x-opencode-session": os.environ.get("OPENCODE_SESSION_ID", "agi2-campaign")})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def extract(resp):
    texts = []
    for item in resp.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text" and c.get("text"):
                    texts.append(c["text"])
    return "\n".join(texts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="muse-spark-1.3-contributor")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--out", default="work/go_pilot.json")
    args = ap.parse_args()
    key = get_key()
    ch = json.load(open("data/arc-agi_evaluation_challenges.json"))
    out = {}
    t0 = time.time()
    try:
        for tid in args.tasks.split(","):
            tid = tid.strip()
            task = ch[tid]
            print(f"== {tid} ==", flush=True)
            prompt = legc.build_prompt(task)
            seen, verified = set(), []
            att = 0
            while len(seen) < args.k and att < args.k + 3:
                att += 1
                try:
                    resp = ask(args.model, legc.SYSTEM_MSG, prompt, key,
                               temp=args.temp)
                except Exception as e:
                    print("  API error:", str(e)[:150], flush=True)
                    break
                code = legc.extract_code(extract(resp))
                if not code or code in seen:
                    continue
                seen.add(code)
                res = legc.run_verifier(
                    [code], task["train"],
                    [t["input"] for t in task["test"]], 4.0, "gocamp")[0]
                if res["status"] == "verified":
                    verified.append((code, res))
            print(f"  {len(seen)} cands, verified: {len(verified)}", flush=True)
            if verified:
                code, res = max(verified, key=lambda cr: len(cr[0]))
                outs = [{"attempt": t, "alt": None, "votes": len(verified),
                         "n_verified": len(verified)}
                        for t in res["test_outputs"]]
                out[tid] = {"verified": True, "n_verified": len(verified),
                            "outputs": outs, "n_pairs": len(task["train"]),
                            "elapsed": 0.0, "program": code[:4000]}
            else:
                out[tid] = {"verified": False, "n_verified": 0, "outputs": [],
                            "n_pairs": len(task["train"]), "elapsed": 0.0}
    finally:
        key = None
    with open(args.out, "w") as f:
        json.dump(out, f)
    nv = sum(1 for r in out.values() if r["verified"])
    print(f"pilot: {nv}/{len(out)} in {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
