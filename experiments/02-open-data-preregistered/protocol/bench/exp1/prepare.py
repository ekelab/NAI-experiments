"""
Experiment 1 (docs/EXPERIMENTS.md): open scaling data NAI has never seen,
turned into SLDBench's format so bench/sldbench.py --suite exp1 runs NAI on it
exactly as on SLDBench. Each task: bench/exp1/tasks/<name>.{train,test}.json
({"features": ["group", …, target], "rows": [...]}) and tasks.json.

The held-out rows are fixed here, by a rule stated before any run
(docs/prereg/EXP1.md): the largest models of each set, as SLDBench's test
extrapolates. None of the sources of SLDBench is used (its tasks come from
Chen 2025, Tao 2024, Lin 2024, Ye 2024, Krajewski 2024, Muennighoff 2023,
Li 2025, Wu & Lo 2024).

    python bench/exp1/prepare.py        → bench/exp1/tasks/*.json
"""
import json, math, os, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "tasks")

SEQ_LEN = 2048          # Porian et al.: every run


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(OUT, exist_ok=True)
    tasks = {}

    def emit(name, feats, target, extrap, limit, monotone, rows, held, note):
        """rows: dicts with group, feats, target; held(row) → True for the test split."""
        train = [r for r in rows if not held(r)]
        test = [r for r in rows if held(r)]
        for split, rs in (("train", train), ("test", test)):
            json.dump({"features": ["group"] + feats + [target], "rows": rs},
                      open(os.path.join(OUT, f"{name}.{split}.json"), "w"))
        tasks[name] = {"features": feats, "targets": [target], "extrap": extrap, "limit": limit,
                       "monotone": monotone, "note": note}
        groups = sorted({r["group"] for r in rows})
        print(f"{name:22s} train {len(train):5d}  test {len(test):5d}  groups {len(groups)}  {note}")

    # E1 - Chinchilla, as reconstructed by Epoch AI (Besiroglu et al. 2024) from
    # Hoffmann et al.'s Figure 4: model size N, training FLOP C; D = C / 6N.
    # Held out: N > 2·10⁹ (about the largest fifth).
    e = pd.read_csv(os.path.join(SRC, "epoch_chinchilla_svg_extracted_data.csv"))
    rows = [{"group": "all", "N": float(r["Model Size"]), "D": float(r["Training FLOP"]) / (6 * float(r["Model Size"])),
             "loss": float(r["loss"])} for _, r in e.iterrows()]
    emit("chinchilla_epoch", ["N", "D"], "loss", "N", 7, ["N", "D"], rows, lambda r: r["N"] > 2e9,
         "Epoch AI reconstruction of Chinchilla (N, D → loss); test N > 2e9")

    # Porian et al. 2024 (MIT): one run a row; N = 12·depth·width² (non-embedding),
    # D = steps · batch · 2048 tokens.
    df = pd.read_pickle(os.path.join(SRC, "porian_experiment_results.pickle.xz"))
    df["N"] = 12.0 * df["depth"] * df["width"] ** 2

    # E2 - the cosine ("chinchilla") decay runs with base hyperparameters, on two
    # corpora (a group each): the final validation loss of each run.
    # Held out: the four largest widths (N ≥ 2.8·10⁸).
    c = df[(df.hparams == "base") & (df.decay == "chinchilla")]
    rows = []
    for _, r in c.iterrows():
        v = r["val/loss"]
        rows.append({"group": r["dataset"], "N": float(r["N"]), "D": float(r["max_step"] * r["bs"] * SEQ_LEN),
                     "loss": float(v.iloc[-1])})
    emit("porian_chinchilla", ["N", "D"], "loss", "N", 7, ["N", "D"], rows, lambda r: r["N"] >= 2.8e8,
         "Porian et al. cosine-decay runs, RefinedWeb and OpenWebText2 (groups); test N >= 2.8e8")

    # E3 - the learning-rate and batch-size sweep (constant schedule): each run's
    # validation loss at its evaluation step; lr and batch free, N and D scale.
    # Held out: the largest width of the sweep (N = 1.7·10⁸).
    s = df[df.hparams == "sweep"]
    rows = []
    for _, r in s.iterrows():
        v = r["val/loss"]
        step = int(v.index[-1])
        rows.append({"group": "all", "lr": float(r["lr"]), "bsz": float(r["bs"]),
                     "D": float(step * r["bs"] * SEQ_LEN), "N": float(r["N"]), "loss": float(v.iloc[-1])})
    nmax = max(r["N"] for r in rows)
    emit("porian_lrbsz", ["lr", "bsz", "D", "N"], "loss", "N", 10, ["D", "N"], rows, lambda r: r["N"] >= nmax,
         "Porian et al. lr x batch sweep; test: the largest width")

    # E4 - DataDecide (AI2 2025, ODC-BY): 25 pretraining corpora (the groups),
    # 14 model sizes 4M…1B, up to three seeds, all trained to 5× Chinchilla.
    # Target, fixed before any fit: the mean over the ten OLMES tasks (MMLU as
    # the mean of its 57 subjects) of −ln(correct_prob_per_char) at each run's
    # final checkpoint - the cross-entropy per character of the correct answer,
    # a loss (it falls with scale). bits_per_byte_corr, the first choice, is
    # missing at the final checkpoints of the large models.
    # Held out: the 1B models (DataDecide's own target scale).
    import pyarrow.parquet as pq
    import re
    size = {"4M": 3.7e6, "6M": 6e6, "8M": 8e6, "10M": 10e6, "14M": 14e6, "16M": 16e6, "20M": 20e6, "60M": 60e6,
            "90M": 90e6, "150M": 150e6, "300M": 300e6, "530M": 530e6, "750M": 750e6, "1B": 1e9}
    parts = [pq.read_table(os.path.join(SRC, "datadecide", f"{i}.parquet"),
                           columns=["params", "data", "task", "step", "seed", "metrics"]).to_pandas() for i in range(4)]
    dd = pd.concat(parts, ignore_index=True)
    dd = dd.loc[dd.groupby(["params", "data", "seed", "task"])["step"].idxmax()]
    # metrics: a Python dict's repr (single quotes) - the one number read directly.
    pat = re.compile(r"'correct_prob_per_char':\s*([-+0-9.eE]+|nan|inf)")
    def prob(m):
        x = pat.search(m)
        return float(x.group(1)) if x else float("nan")
    dd["p"] = dd["metrics"].map(prob)
    dd = dd[np.isfinite(dd["p"]) & (dd["p"] > 0)]
    dd["nll"] = -np.log(dd["p"])
    dd["family"] = dd["task"].map(lambda t: "mmlu" if t.startswith("mmlu_") else t)
    fam = dd.groupby(["params", "data", "seed", "family"])["nll"].mean().reset_index()
    runs = fam.groupby(["params", "data", "seed"]).agg(nll=("nll", "mean"), n=("family", "size")).reset_index()
    runs = runs[runs.n == 10]                               # all ten tasks present
    rows = [{"group": r["data"], "N": size[r["params"]], "nll": float(r["nll"])} for _, r in runs.iterrows()]
    emit("datadecide", ["N"], "nll", "N", 6, ["N"], rows, lambda r: r["N"] >= 1e9,
         "DataDecide: OLMES-10 mean -ln(correct_prob_per_char) vs model size, 25 corpora (groups); test: 1B")

    json.dump(tasks, open(os.path.join(OUT, "tasks.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
