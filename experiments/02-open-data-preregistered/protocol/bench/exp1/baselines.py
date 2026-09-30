"""
Baselines of experiment 1 (docs/EXPERIMENTS.md, docs/prereg/EXP1.md): the laws
as practitioners fit them, on the same rows NAI sees, with constants per group
(one form, its own constants for each group - the rule NAI follows).

  B1  Chinchilla: L = E + A/N^α + B/D^β   (one input: E + A/N^α)
      fitted as Hoffmann et al. 2022 (approach 3): Huber loss (δ = 10⁻³) on
      log L with L = exp(LSE(a − α log N, b − β log D, e)), L-BFGS from a grid
      of starting points, the best kept.
  B2  a pure power law without offset: log L = c − α log N − β log D
      (least squares in log space; Kaplan et al. 2020 style).
  B3  porian_lrbsz only: B1 plus parabolas in ln lr and ln batch about optima
      that move with N and D - the form of the lr/batch literature (DeepSeek
      LLM 2024, Step Law 2025): + c₁(ln lr − p₀ − p₁ ln N − p₂ ln D)²
      + c₂(ln B − q₀ − q₁ ln N − q₂ ln D)².

Each writes its predictions for the test rows (in the harness's order) and
bench/exp1/evaluate.py scores them with NAI's.

    python bench/exp1/baselines.py [--data bench/exp1/tasks] [--out bench/results/exp1/baselines]
    python bench/exp1/baselines.py --selftest      (a synthetic Chinchilla law: are its constants found?)
"""
import argparse, itertools, json, math, os, sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

HERE = os.path.dirname(os.path.abspath(__file__))


def huber(r, delta=1e-3):
    a = np.abs(r)
    return np.where(a <= delta, 0.5 * r * r, delta * (a - 0.5 * delta))


# ---- B1 -------------------------------------------------------------------

def b1_predict(p, X):
    if X.shape[1] == 2:
        a, b, e, al, be = p
        return np.exp(logsumexp(np.stack([a - al * np.log(X[:, 0]), b - be * np.log(X[:, 1]),
                                          np.full(len(X), e)]), axis=0))
    a, e, al = p
    return np.exp(logsumexp(np.stack([a - al * np.log(X[:, 0]), np.full(len(X), e)]), axis=0))


def huber_grad(r, delta=1e-3):
    return np.where(np.abs(r) <= delta, r, delta * np.sign(r))


def b1_fit(X, y):
    """Hoffmann et al.'s grid of starts (a, b ∈ 0…25, e ∈ −1…1, α, β ∈ 0…2),
    L-BFGS with the exact gradient of the Huber objective."""
    ly = np.log(y)
    two = X.shape[1] == 2
    lN = np.log(X[:, 0])
    lD = np.log(X[:, 1]) if two else None

    def obj(p):
        if two:
            a, b, e, al, be = p
            U = np.stack([a - al * lN, b - be * lD, np.full(len(X), e)])
        else:
            a, e, al = p
            U = np.stack([a - al * lN, np.full(len(X), e)])
        lse = logsumexp(U, axis=0)
        w = np.exp(U - lse)                               # softmax weights of the terms
        r = lse - ly
        h = huber_grad(r)
        if two:
            g = np.array([np.sum(h * w[0]), np.sum(h * w[1]), np.sum(h * w[2]),
                          -np.sum(h * w[0] * lN), -np.sum(h * w[1] * lD)])
        else:
            g = np.array([np.sum(h * w[0]), np.sum(h * w[1]), -np.sum(h * w[0] * lN)])
        return float(np.sum(huber(r))), g

    if two:
        grid = itertools.product(np.arange(0, 30, 5), np.arange(0, 30, 5), np.arange(-1, 1.5, 0.5),
                                 np.arange(0, 2.5, 0.5), np.arange(0, 2.5, 0.5))
    else:
        grid = itertools.product(np.arange(0, 30, 5), np.arange(-1, 1.5, 0.5), np.arange(0, 2.5, 0.5))
    best = None
    for p0 in grid:
        try:
            r = minimize(obj, np.array(p0, dtype=float), jac=True, method="L-BFGS-B")
        except (FloatingPointError, ValueError):
            continue
        if np.isfinite(r.fun) and (best is None or r.fun < best.fun):
            best = r
    return best.x


# ---- B2 -------------------------------------------------------------------

def b2_fit(X, y):
    A = np.column_stack([np.ones(len(X))] + [np.log(X[:, j]) for j in range(X.shape[1])])
    coef, *_ = np.linalg.lstsq(A, np.log(y), rcond=None)
    return coef


def b2_predict(c, X):
    A = np.column_stack([np.ones(len(X))] + [np.log(X[:, j]) for j in range(X.shape[1])])
    return np.exp(A @ c)


# ---- B3 (lr, batch, D, N) ---------------------------------------------------

def b3_predict(p, Z):
    lr, B, D, N = Z[:, 0], Z[:, 1], Z[:, 2], Z[:, 3]
    a, b, e, al, be, c1, p0, p1, p2, c2, q0, q1, q2 = p
    base = np.exp(logsumexp(np.stack([a - al * np.log(N), b - be * np.log(D), np.full(len(Z), e)]), axis=0))
    return base + c1 ** 2 * (np.log(lr) - p0 - p1 * np.log(N) - p2 * np.log(D)) ** 2 \
                + c2 ** 2 * (np.log(B) - q0 - q1 * np.log(N) - q2 * np.log(D)) ** 2


def b3_fit(Z, y, seed=0, starts=200):
    """B1's law of N and D from the rows nearest each (N, D)'s best (lr, batch)
    as the base, then 200 starts of the full form - optima centred on the
    sweep's range, slopes zero or small - each polished by L-BFGS, then the
    best refined by Nelder–Mead."""
    ly = np.log(y)
    lr, B, lD, lN = np.log(Z[:, 0]), np.log(Z[:, 1]), np.log(Z[:, 2]), np.log(Z[:, 3])
    # the base from the best run of each (N, D): the law without the parabolas
    best_rows = {}
    for i in range(len(Z)):
        k = (round(lN[i], 6), round(lD[i], 3))
        if k not in best_rows or y[i] < y[best_rows[k]]:
            best_rows[k] = i
    idx = np.array(sorted(best_rows.values()))
    base = b1_fit(Z[idx][:, [3, 2]], y[idx]) if len(idx) >= 5 else b1_fit(Z[:, [3, 2]], y)
    rng = np.random.default_rng(seed)
    obj = lambda p: float(np.sum(huber(np.log(np.maximum(b3_predict(p, Z), 1e-300)) - ly)))
    best = None
    for k in range(starts):
        sl = (lambda: 0.0) if k < starts // 4 else (lambda: rng.normal(0, 0.3))
        p0 = np.concatenate([base, [rng.uniform(0.02, 0.5), rng.uniform(lr.min(), lr.max()), sl(), sl(),
                                    rng.uniform(0.02, 0.5), rng.uniform(B.min(), B.max()), sl(), sl()]])
        # the optimum's intercept so that it sits at p0's point for the median N, D
        p0[6] -= p0[7] * np.median(lN) + p0[8] * np.median(lD)
        p0[10] -= p0[11] * np.median(lN) + p0[12] * np.median(lD)
        r = minimize(obj, p0, method="L-BFGS-B")
        if np.isfinite(r.fun) and (best is None or r.fun < best.fun):
            best = r
    r = minimize(obj, best.x, method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-14})
    return r.x if r.fun <= best.fun else best.x


# ---- running on the tasks --------------------------------------------------

def load(path):
    d = json.load(open(path))
    groups = {}
    for r in d["rows"]:
        groups.setdefault(r["group"], []).append(r)
    return dict(sorted(groups.items()))


def run(data, out):
    tasks = json.load(open(os.path.join(data, "tasks.json")))
    os.makedirs(out, exist_ok=True)
    for name, t in tasks.items():
        feats, target = t["features"], t["targets"][0]
        train, test = load(os.path.join(data, f"{name}.train.json")), load(os.path.join(data, f"{name}.test.json"))
        preds = {"B1": [], "B2": []}
        if name == "porian_lrbsz":
            preds["B3"] = []
        for g in [g for g in train if g in test]:
            tr, te = train[g], test[g]
            y = np.array([r[target] for r in tr])
            nd = [f for f in ("N", "D") if f in feats]                 # B1/B2 see size and data only
            Xtr, Xte = np.array([[r[f] for f in nd] for r in tr]), np.array([[r[f] for f in nd] for r in te])
            p1 = b1_fit(Xtr, y)
            preds["B1"] += list(b1_predict(p1, Xte))
            c2 = b2_fit(Xtr, y)
            preds["B2"] += list(b2_predict(c2, Xte))
            if "B3" in preds:
                Ztr = np.array([[r[f] for f in feats] for r in tr])
                Zte = np.array([[r[f] for f in feats] for r in te])
                p3 = b3_fit(Ztr, y)
                preds["B3"] += list(b3_predict(p3, Zte))
        for b, p in preds.items():
            with open(os.path.join(out, f"{name}.{b}.pred.txt"), "w") as f:
                f.write("".join("%.17g\n" % v for v in p))
        print(name, {b: len(p) for b, p in preds.items()})


def selftest():
    """A Chinchilla law with known constants: B1 must find them."""
    rng = np.random.default_rng(1)
    N = np.exp(rng.uniform(np.log(1e7), np.log(1e10), 200))
    D = np.exp(rng.uniform(np.log(1e9), np.log(1e11), 200))
    L = 1.69 + 406.4 / N ** 0.34 + 410.7 / D ** 0.28
    y = L * (1 + rng.normal(0, 0.002, len(L)))
    a, b, e, al, be = b1_fit(np.column_stack([N, D]), y)
    print("true  E 1.69  A 406.4  B 410.7  alpha 0.34  beta 0.28")
    print("B1    E %.3f  A %.1f  B %.1f  alpha %.3f  beta %.3f" % (math.exp(e), math.exp(a), math.exp(b), al, be))
    Z = np.column_stack([np.exp(rng.uniform(np.log(1e-4), np.log(1e-2), 300)), np.exp(rng.uniform(np.log(32), np.log(1024), 300)),
                         np.exp(rng.uniform(np.log(1e9), np.log(1e11), 300)), np.exp(rng.uniform(np.log(1e7), np.log(1e9), 300))])
    base = 1.69 + 406.4 / Z[:, 3] ** 0.34 + 410.7 / Z[:, 2] ** 0.28
    yy = base + 0.02 * (np.log(Z[:, 0]) - (-2.0 - 0.25 * np.log(Z[:, 3]))) ** 2 + 0.01 * (np.log(Z[:, 1]) - (-3.0 + 0.33 * np.log(Z[:, 2]))) ** 2
    p = b3_fit(Z, yy)
    r = np.log(b3_predict(p, Z)) - np.log(yy)
    print("B3 on a synthetic lr/batch law: rms log error %.2e (0 = exact)" % float(np.sqrt(np.mean(r * r))))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "tasks"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "exp1", "baselines"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    np.seterr(all="ignore")
    if a.selftest:
        selftest()
    else:
        run(a.data, a.out)
