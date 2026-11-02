"""patch_fork.py - apply local fixes to hamatz/failed-in-aimo (base, 9 cells).

Patches (see KAGGLE_PATCH.md):
  P1 cell5 starter.py -> starter_fixed.py content (debug-filter removal,
     cost-ascending queue, LegC-skip, env paths)
  P2 cell6 run command -> starter_fixed.py
  P3 cell4 arc_solver.worker: per-key try/except (one bad puzzle must not
     kill a 4-GPU worker; partial outputs stay submittable)

Usage: python3 tools/patch_fork.py   (rewrites fork/failed-in-aimo.ipynb)
"""
import ast
import copy
import json

FORK = "fork/failed-in-aimo.ipynb"
FIXED_STARTER = "src/starter_fixed.py"


def main():
    nb = json.load(open(FORK))
    assert len(nb["cells"]) == 9, f"layout changed: {len(nb['cells'])} cells"
    cells = nb["cells"]
    src = lambda i: "".join(cells[i]["source"])
    assert src(4).startswith("%%writefile arc_solver.py"), src(4)[:60]
    assert src(5).startswith("%%writefile starter.py"), src(5)[:60]
    assert "python starter.py" in src(6), src(6)[:120]

    # ---- P1: starter cell ----
    fixed = open(FIXED_STARTER).read()
    # strip module docstring usage note? keep - harmless, documents itself
    cells[5]["source"] = ["%%writefile starter_fixed.py\n"] + [
        line + "\n" for line in fixed.splitlines()
    ]
    ast.parse(fixed)  # syntax check

    # ---- P2: run cell ----
    new_run = src(6).replace("python starter.py", "python starter_fixed.py")
    assert new_run != src(6)
    cells[6]["source"] = [new_run]

    # ---- P3: arc_solver per-key try/except ----
    lines = src(4).splitlines(keepends=True)
    # locate loop-body span: 'start_time = time.time()' .. 'finished {key}'
    start = next(i for i, l in enumerate(lines)
                 if l.strip() == "start_time = time.time()")
    end = next(i for i, l in enumerate(lines)
               if "finished {key} in" in l)
    assert 300 < start < end and end == len(lines) - 1, (start, end)
    body = []
    for l in lines[start:end + 1]:
        l = l.rstrip("\n")
        body.append(("    " + l + "\n") if l.strip() else "\n")
    patched = (
        lines[:start]
        + ["        try:\n"]
        + body
        + ["        except Exception as e:\n",
           "            print(f\"[Rank {rank}] ERROR on {key}: {e!r} -- skipping\")\n",
           "            gc.collect()\n",
           "            try:\n",
           "                torch.cuda.empty_cache()\n",
           "            except Exception:\n",
           "                pass\n",
           "            continue\n"]
    )
    patched_src = "".join(patched)
    ast.parse("\n".join(patched_src.splitlines()[1:]))  # skip %%writefile
    cells[4]["source"] = patched_src.splitlines(keepends=True)

    json.dump(nb, open(FORK, "w"), ensure_ascii=False, indent=1)
    print(f"P1 starter: {len(src(5))} -> {len(''.join(cells[5]['source']))} chars")
    print(f"P2 run cmd: {new_run.strip()[:100]}")
    print(f"P3 solver: indented lines {start}-{end}, try/except added")
    print("notebook rewritten:", FORK)


if __name__ == "__main__":
    main()
