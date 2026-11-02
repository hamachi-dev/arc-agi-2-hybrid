"""nim_probe.py - test NIM-hosted strong models on one ARC task via Leg C prompt.

Key is read from the user's own opencode.jsonc (never printed).
Usage: python3 tools/nim_probe.py --model nvidia/llama-3.1-nemotron-70b-instruct --task 20270e3b
"""
import argparse
import json
import re
import sys
import urllib.request

sys.path.insert(0, "src")
import arc_induction_v2 as legc  # noqa: E402


def get_key():
    txt = open("/home/taizo/.config/opencode/opencode.jsonc").read()
    return re.search(r'"apiKey": "([^"]+)"', txt).group(1)


def chat(model, system, user, key, max_tokens=8192, temp=0.8):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "max_tokens": max_tokens,
        "temperature": temp,
        "reasoning_effort": "low",
    }).encode()
    req = urllib.request.Request(
        "http://localhost:8000/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--task", default="20270e3b")
    args = ap.parse_args()
    key = get_key()
    ch = json.load(open("data/arc-agi_evaluation_challenges.json"))
    task = ch[args.task]
    prompt = legc.build_prompt(task)
    print("prompt chars:", len(prompt), flush=True)
    try:
        resp = chat(args.model, legc.SYSTEM_MSG, prompt, key)
    finally:
        key = None
    msg = resp["choices"][0]["message"]
    txt = msg.get("content") or msg.get("reasoning") or ""
    print("REPLY:", txt[:1500], flush=True)
    code = legc.extract_code(txt)
    print("EXTRACTED:", (code[:1000] if code else "<EMPTY>"), flush=True)
    if code:
        res = legc.run_verifier([code], task["train"],
                                [t["input"] for t in task["test"]],
                                4.0, "nimprobe")
        print("VERIFY:", json.dumps(res[0])[:500], flush=True)


if __name__ == "__main__":
    main()
