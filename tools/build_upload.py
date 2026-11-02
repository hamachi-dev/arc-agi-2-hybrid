"""build_upload.py - rebuild fork notebook as clean line-array JSON for upload.

Why: save_notebook needs the FULL notebook text in one call. The pulled
file has 2 cells as giant single-line strings (unreadable/untransmittable
exactly), so we regenerate those + patched cells from exact local sources:
  cell2 <- src/arc_loader.py     (semantic identical to upstream)
  cell3 <- src/arc_decoder.py    (semantic identical to upstream)
  cell4 <- src/arc_solver.py + per-key try/except (KAGGLE_PATCH.md P3)
  cell5 <- src/starter_fixed.py  (KAGGLE_PATCH.md P1)
  cell6 <- run starter_fixed.py  (KAGGLE_PATCH.md P2)
Cells 0,1,7,8 pass through untouched. Execution metadata/outputs are
stripped (fresh-notebook state; Kaggle accepts this).

Verifies: every %%writefile payload parses with ast; notebook parses
as JSON with 9 code cells.
"""
import ast
import json

FORK = "/tmp/fork_backup.ipynb"  # pristine pull; patched file is rewritten below
OUT = "fork/upload.ipynb"


def payload_lines(path, header):
    body = open(path).read().rstrip("\n").split("\n")
    src = [header + "\n"] + [l + "\n" for l in body]
    src[-1] = src[-1].rstrip("\n")  # last line: no trailing newline
    ast.parse("\n".join(body))
    return src


def solver_lines():
    body = open("src/arc_solver.py").read().rstrip("\n").split("\n")
    start = next(i for i, l in enumerate(body)
                 if l.strip() == "start_time = time.time()")
    end = next(i for i, l in enumerate(body) if "finished {key} in" in l)
    assert end == len(body) - 1, (start, end)
    out = body[:start] + ["        try:"]
    out += [("    " + l) if l.strip() else "" for l in body[start:]]
    out += ["        except Exception as e:",
            "            print(f\"[Rank {rank}] ERROR on {key}: {e!r} -- skipping\")",
            "            gc.collect()",
            "            try:",
            "                torch.cuda.empty_cache()",
            "            except Exception:",
            "                pass",
            "            continue"]
    ast.parse("\n".join(out))
    src = ["%%writefile arc_solver.py\n"] + [l + "\n" for l in out]
    src[-1] = src[-1].rstrip("\n")
    return src


def main():
    nb = json.load(open(FORK))
    assert len(nb["cells"]) == 9
    get = lambda i: "".join(nb["cells"][i]["source"])
    assert get(4).startswith("%%writefile arc_solver.py")
    assert get(5).startswith("%%writefile starter.py")
    assert "python starter.py" in get(6)

    nb["cells"][2]["source"] = payload_lines(
        "src/arc_loader.py", "%%writefile arc_loader.py")
    nb["cells"][3]["source"] = payload_lines(
        "src/arc_decoder.py", "%%writefile arc_decoder.py")
    nb["cells"][4]["source"] = solver_lines()
    nb["cells"][5]["source"] = payload_lines(
        "src/starter_fixed.py", "%%writefile starter_fixed.py")
    run = get(6).replace("python starter.py", "python starter_fixed.py")
    assert run != get(6)
    nb["cells"][6]["source"] = [run]

    for c in nb["cells"]:
        c["outputs"] = []
        c["execution_count"] = None
        md = c.get("metadata", {})
        md.pop("execution", None)
        c["metadata"] = md

    json.dump(nb, open(OUT, "w"), ensure_ascii=False, indent=1)
    nb2 = json.load(open(OUT))
    assert len(nb2["cells"]) == 9
    heads = [( "".join(c["source"]).splitlines() or ["<empty>"])[0]
             for c in nb2["cells"]]
    print("cell heads:", heads)
    print("upload bytes:", len(open(OUT, encoding="utf-8").read()))


if __name__ == "__main__":
    main()
