"""
Scores of experiment 1 (docs/prereg/EXP1.md), the same for NAI and the
baselines, from their predictions for the test rows (in the harness's order:
groups sorted by name, rows as in the file).

  R2       1 − MSE/Var over all test rows, SLDBench's way (upstream/evaluator.py):
           a non-finite prediction makes it −1.
  RelFar   the mean |pred − y| / y over the rows of the largest N (the farthest
           extrapolation), in per cent.
  MAPE     the mean |pred − y| / y over all test rows, in per cent.
  Regret   porian_lrbsz only: for each token budget D among the test rows, the
           configuration (lr, batch) predicted best is taken, and the regret is
           its true loss minus the best true loss at that D; mean over D.

    python bench/exp1/evaluate.py --pred NAME=path/to/{task}.pred.txt ... [--data bench/exp1/tasks]
"""
import argparse, json, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def test_rows(data, task):
    d = json.load(open(os.path.join(data, f"{task}.test.json")))
    groups = {}
    for r in d["rows"]:
        groups.setdefault(r["group"], []).append(r)
    return [r for g in sorted(groups) for r in groups[g]]


def scores(task, rows, target, pred):
    y = np.array([r[target] for r in rows])
    p = np.array(pred, dtype=float)
    out = {}
    if len(p) != len(y) or not np.all(np.isfinite(p)):
        out["R2"] = -1.0
        return out
    out["R2"] = float(1 - np.mean((p - y) ** 2) / np.var(y))
    rel = np.abs(p - y) / np.abs(y)
    out["MAPE"] = float(100 * rel.mean())
    nmax = max(r["N"] for r in rows)
    far = [i for i, r in enumerate(rows) if r["N"] == nmax]
    out["RelFar"] = float(100 * rel[far].mean())
    if task == "porian_lrbsz":
        reg = []
        for D in sorted({r["D"] for r in rows}):
            idx = [i for i, r in enumerate(rows) if r["D"] == D]
            if len(idx) < 2:
                continue
            chosen = min(idx, key=lambda i: p[i])
            reg.append(y[chosen] - min(y[i] for i in idx))
        out["Regret"] = float(np.mean(reg)) if reg else float("nan")
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "tasks"))
    ap.add_argument("--pred", action="append", default=[],
                    help="NAME=pattern with {task} and {target}, e.g. B1=results/exp1/baselines/{task}.B1.pred.txt")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    tasks = json.load(open(os.path.join(a.data, "tasks.json")))
    table = {}
    for spec in a.pred:
        name, pattern = spec.split("=", 1)
        for task, t in tasks.items():
            target = t["targets"][0]
            path = pattern.format(task=task, target=target)
            rows = test_rows(a.data, task)
            pred = [float(v) for v in open(path).read().split()] if os.path.exists(path) else []
            table.setdefault(task, {})[name] = scores(task, rows, target, pred)
    for task, byname in table.items():
        print(f"\n{task}")
        for name, s in byname.items():
            print("  %-12s " % name + "  ".join(f"{k} {v:8.4f}" for k, v in s.items()))
    if a.json:
        json.dump(table, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
