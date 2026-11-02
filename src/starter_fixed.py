"""starter_fixed.py - patched copy of upstream starter.py.

Upstream issues fixed (see src/starter.py:58-61):
  1. DEBUG FILTER REMOVED: upstream only queued 4 hardcoded tasks when
     KAGGLE_IS_COMPETITION_RERUN is unset, so local eval runs scored
     ~4/120 tasks and told you nothing. This version queues ALL tasks
     in both modes (minus LegC-verified skips).
  2. PATHS VIA ENV: /kaggle/input, /kaggle/working, /kaggle/worker* and
     /kaggle/inference_outputs are Kaggle-only. Every path below honours
     ARC_DATA_DIR / ARC_WORK_DIR / ARC_OUT_DIR / ARC_FLAG_DIR with Kaggle
     defaults, so the same file runs locally and on Kaggle.
  3. NPROCS follows LEGC-style GPU count instead of hardcoded 4.

Usage (local smoke, no GPU work - just builds the queue):
    ARC_DATA_DIR=../data ARC_WORK_DIR=../work python3 starter_fixed.py --dry-run
Usage (Kaggle, replaces cell 9):
    !python starter_fixed.py --end-time {global_end_time}
"""
import os
import time
import json
import argparse

DATA_DIR = os.environ.get(
    "ARC_DATA_DIR",
    "/kaggle/input/competitions/arc-prize-2026-arc-agi-2")
WORK_DIR = os.environ.get("ARC_WORK_DIR", "/kaggle/working")
OUT_DIR = os.environ.get("ARC_OUT_DIR", "/kaggle/inference_outputs")
FLAG_DIR = os.environ.get("ARC_FLAG_DIR", "/kaggle")


def build_queue(data):
    """All tasks, minus LegC-verified skips, cheapest-first.

    Upstream queued alphabetically (and only 4 debug tasks off-rerun).
    Cheapest-first gives the best any-time behaviour: cell10 builds a
    valid submission from whatever is in OUT_DIR, so if the run is cut
    early (or a straggler tail task eats a worker), the most tasks are
    already done. Verified by tools/sim_schedule.py + tools/test_merge.py.
    """
    legc_skip = set()
    legc_path = os.path.join(WORK_DIR, "induction_results.json")
    if os.path.exists(legc_path):
        with open(legc_path) as f:
            legc_skip = {k for k, r in json.load(f).items()
                         if isinstance(r, dict) and r.get("verified")}
        print(f"[LegC] skipping {len(legc_skip)} verified task(s)"
              " in the base queue")
    keys = [k for k in data.keys() if k not in legc_skip]
    try:
        from arc_induction_v2 import est_cost
        keys.sort(key=lambda k: est_cost(data[k]))
        print("[queue] cost-ascending order")
    except Exception as e:
        keys = sorted(keys)
        print(f"[queue] alphabetical fallback ({e!r})")
    return keys


def local_worker(rank, queue, end_time):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    import torch
    torch.set_default_device("cpu")
    if rank > 0:
        while not os.path.exists(os.path.join(FLAG_DIR, f"worker{rank-1}")):
            time.sleep(5)
    from arc_solver import worker
    with open(os.path.join(FLAG_DIR, f"worker{rank}"), "w") as f:
        f.write("Ok")
    print(f"[Rank {rank}] start!")
    worker(rank, queue, end_time)
    print(f"[Rank {rank}] done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--end-time", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true",
                        help="print queue stats and exit (no GPU work)")
    args = parser.parse_args()

    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")
    if rerun_mode:
        test_path = os.path.join(DATA_DIR, "arc-agi_test_challenges.json")
    else:
        test_path = os.path.join(DATA_DIR,
                                 "arc-agi_evaluation_challenges.json")

    with open(test_path, "r") as f:
        data = json.load(f)

    keys = build_queue(data)
    print(f"[queue] mode={'RERUN' if rerun_mode else 'LOCAL_EVAL'}"
          f" tasks={len(keys)}/{len(data)}")
    if args.dry_run:
        print("[queue] first 5:", keys[:5])
        raise SystemExit(0)

    import torch.multiprocessing as mp
    try:
        import subprocess
        n_gpu = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                               text=True).stdout.count("GPU ")
    except Exception:
        n_gpu = 1
    nprocs = max(1, min(4, n_gpu or 1))

    queue = mp.Manager().Queue()
    for key in keys:
        queue.put(key)
    for _ in range(nprocs):
        queue.put(None)

    mp.spawn(local_worker, args=(queue, args.end_time), nprocs=nprocs)
