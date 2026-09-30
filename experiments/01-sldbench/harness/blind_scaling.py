"""
A blind suite of scaling-law tables, in SLDBench's format, for sldbench.py
--suite blind. Generated once (2026-09-27), before a round of work on the
engine, from laws of the literature with their published constants and a
realistic noise; run only at the end of the round. Whatever the round tuned
on SLDBench, it never saw these.

  chinchilla  L = E + A/N^α + B/D^β                 (Hoffmann et al. 2022)
  kaplan      L = (N_c/N)^α_N, three datasets       (Kaplan et al. 2020)
  constrained L = E + A/N^α + B/D'^β, D' from repeated data (Muennighoff et al. 2023)
  lr          L = L₀(N) + k·(ln lr − ln lr*(N))², lr* ∝ N^−0.25
  moe         ln L = a ln N + b ln E + c ln N ln E + d (Clark et al. 2022)
  finetune    L = E + A/(D^α + B), six models        (the SFT law of SLDBench's authors' form)
  lrbsz       Chinchilla + parabolas in ln lr, ln B about DeepSeek LLM's optima (added 2026-09-29)

Each test split extrapolates, like SLDBench's: bigger models or more data.

    python blind_scaling.py            → blind/scaling/*.json, tasks.json
"""
import json, math, os, random

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "blind", "scaling")


def logspace(a, b, n):
    return [math.exp(math.log(a) + (math.log(b) - math.log(a)) * i / (n - 1)) for i in range(n)]


def noisy(rng, v, rel):
    return v * (1 + rel * rng.gauss(0, 1))


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(20260927)
    tasks = {}

    def emit(name, feats, target, extrap, limit, rows, cut):
        """rows: dicts with group, feats, target; test = the rows beyond cut on extrap."""
        train = [r for r in rows if r[extrap] <= cut]
        test = [r for r in rows if r[extrap] > cut]
        for split, rs in (("train", train), ("test", test)):
            json.dump({"features": ["group"] + feats + [target], "rows": rs},
                      open(os.path.join(OUT, f"{name}.{split}.json"), "w"))
        tasks[name] = {"features": feats, "targets": [target], "extrap": extrap, "limit": limit}
        print(name, len(train), len(test))

    # Chinchilla: N 1e7…1e10 (test above 1e9), D 1e9…1e11.
    rows = []
    for N in logspace(1e7, 1e10, 13):
        for D in logspace(1e9, 1e11, 7):
            L = 1.69 + 406.4 / N ** 0.34 + 410.7 / D ** 0.28
            rows.append({"group": "all", "N": round(N), "D": round(D), "loss": round(noisy(rng, L, 0.005), 4)})
    emit("chinchilla", ["N", "D"], "loss", "N", 5, rows, 1.01e9)

    # Kaplan: L = (N_c/N)^0.076, three datasets with their own N_c.
    rows = []
    for g, Nc in (("web", 8.8e13), ("code", 2.1e13), ("books", 4.0e14)):
        for N in logspace(1e6, 1e10, 14):
            rows.append({"group": g, "N": round(N), "loss": round(noisy(rng, (Nc / N) ** 0.076, 0.004), 4)})
    emit("kaplan", ["N"], "loss", "N", 2, rows, 1.01e9)

    # Data-constrained: U unique tokens repeated to D; D' = U + U·R*·(1 − e^(−R/R*)), R = D/U − 1.
    rows = []
    for N in logspace(1e8, 3e9, 6):
        for U in logspace(1e9, 1e11, 5):
            for epochs in (1, 2, 4, 8, 16, 40):
                D = U * epochs
                R = D / U - 1
                Dp = U + U * 15.4 * (1 - math.exp(-R / 15.4))
                L = 1.87 + 521 / N ** 0.35 + 1488 / Dp ** 0.35
                rows.append({"group": "all", "unique_tokens": round(U), "params": round(N), "tokens": round(D),
                             "loss": round(noisy(rng, L, 0.005), 4)})
    emit("constrained", ["unique_tokens", "params", "tokens"], "loss", "params", 7, rows, 1.1e9)

    # Learning rate: a parabola in ln lr about lr*(N) = 0.8·N^−0.25, depth L₀(N) = 1.8 + 30/N^0.2.
    rows = []
    for N in logspace(1e7, 3e9, 8):
        for lr in logspace(1e-4, 3e-2, 9):
            L0 = 1.8 + 30 / N ** 0.2
            L = L0 + 0.02 * (math.log(lr) - math.log(0.8 * N ** -0.25)) ** 2
            rows.append({"group": "all", "lr": lr, "N": round(N), "loss": round(noisy(rng, L, 0.004), 4)})
    emit("lr", ["lr", "N"], "loss", "N", 6, rows, 1.1e9)

    # MoE: ln L = a ln N + b ln Ê + c ln N ln Ê + d, E experts.
    rows = []
    for N in logspace(1.5e7, 1.3e9, 8):
        for E in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
            lnL = -0.082 * math.log(N / 1e9) - 0.108 * math.log(E) * 0.1 + 0.009 * math.log(N / 1e9) * math.log(E) + 0.75
            rows.append({"group": "all", "num_experts": E, "dense_parameter_count": round(N),
                         "loss_validation": round(noisy(rng, math.exp(lnL), 0.004), 4)})
    emit("moe", ["num_experts", "dense_parameter_count"], "loss_validation", "dense_parameter_count", 6, rows, 5e8)

    # Fine-tuning: six models, each its own E, A, α, B.
    rows = []
    for g in range(6):
        E, A, al, B = rng.uniform(0.5, 1.5), rng.uniform(2, 8), rng.uniform(0.2, 0.5), rng.uniform(2, 30)
        for D in [200 * 2 ** i for i in range(13)]:
            rows.append({"group": "model%d" % g, "sft_data_size": D,
                         "sft_loss": round(noisy(rng, E + A / (D ** al + B), 0.01), 4)})
    emit("finetune", ["sft_data_size"], "sft_loss", "sft_data_size", 4, rows, 410000)

    # Added 2026-09-29 (after the first rounds; the rows above are unchanged -
    # these draws come after theirs): four inputs, like a learning-rate and
    # batch-size sweep. Loss: Chinchilla's; the optima of lr and batch from
    # DeepSeek LLM (Bi et al. 2024): lr* = 0.3118·C^−0.125, B* = 0.292·C^0.3271
    # tokens, C = 6ND; about them, parabolas in ln lr and ln B (curvatures
    # chosen here, not from a paper). The test: bigger models.
    rows = []
    for N in logspace(5e7, 3e9, 6):
        for D in logspace(2e9, 1e11, 4):
            C = 6 * N * D
            lr_opt, B_opt = 0.3118 * C ** -0.125, 0.292 * C ** 0.3271
            for lr in logspace(lr_opt / 8, lr_opt * 8, 7):
                for B in logspace(B_opt / 4, B_opt * 4, 5):
                    L = (1.69 + 406.4 / N ** 0.34 + 410.7 / D ** 0.28
                         + 0.015 * (math.log(lr) - math.log(lr_opt)) ** 2 + 0.01 * (math.log(B) - math.log(B_opt)) ** 2)
                    rows.append({"group": "all", "lr": lr, "bsz": round(B), "data_size": round(D), "N": round(N),
                                 "loss": round(noisy(rng, L, 0.004), 4)})
    emit("lrbsz", ["lr", "bsz", "data_size", "N"], "loss", "N", 10, rows, 1.1e9)

    json.dump(tasks, open(os.path.join(OUT, "tasks.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
