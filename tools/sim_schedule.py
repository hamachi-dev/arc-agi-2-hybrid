"""sim_schedule.py - can the base TTT pipeline finish 240 test tasks?

Unknown: true per-puzzle wall time on L4 (train 128 aug samples LoRA-r256
on 4B + turbo_dfs + rescoring). So we anchor 3 scenarios to a MEAN time
and scale per-task time by est_cost (bigger grids -> more work), clipped
to [2min, 20min] (code caps: 1200s/puzzle, 540s/DFS call).

Compares queue orders under deadline: alphabetical (upstream) vs
cost-ascending (starter_fixed.py). Greedy 4-worker assignment in order.

Usage: python3 tools/sim_schedule.py [--base-hours 11] [--mean-min 11]
"""
import argparse
import json
import sys

sys.path.insert(0, "src")
from arc_induction_v2 import est_cost  # noqa: E402


def simulate(items, n_workers, deadline_s, mean_s):
    costs = [c for c, _ in items]
    mean_c = sum(costs) / len(costs)
    times = [min(1200, max(120, mean_s * c / mean_c)) for c in costs]
    avail = [0.0] * n_workers
    done = 0
    for t in times:
        w = min(range(n_workers), key=lambda i: avail[i])
        if avail[w] + t <= deadline_s:
            avail[w] += t
            done += 1
        # else: this worker is full; smaller later tasks could still fit,
        # but greedy-in-order leaves them -> models head-of-line blocking
    busy = sum(avail) / (n_workers * deadline_s)
    return done, busy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-hours", type=float, default=11.0)
    ap.add_argument("--mean-min", type=float, default=11.0)
    args = ap.parse_args()

    te = json.load(open("data/arc-agi_test_challenges.json"))
    items = [(est_cost(t), k) for k, t in te.items()]
    deadline = args.base_hours * 3600
    mean_s = args.mean_min * 60

    alpha = sorted(items, key=lambda x: x[1])          # upstream: alphabetical
    cheap = sorted(items, key=lambda x: x[0])          # fixed: cost-ascending

    for name, order in (("alphabetical", alpha), ("cost-ascending", cheap)):
        done, busy = simulate(order, 4, deadline, mean_s)
        print(f"{name:15s} completed {done:3d}/240  worker-busy {busy*100:4.1f}%")

    # how many of the cheapest tasks fit in one worker's day?
    cs = sorted(c for c, _ in items)
    t = 0
    n = 0
    for c in cs:
        step = min(1200, max(120, mean_s * c / (sum(cs) / len(cs))))
        if t + step <= deadline:
            t += step
            n += 1
    print(f"(single worker could do {n} cheapest tasks in "
          f"{args.base_hours}h at mean {args.mean_min}min)")


if __name__ == "__main__":
    main()
