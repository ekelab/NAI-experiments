# Experiment 02 - a pre-registered test on open scaling-law data

**Question.** On open data NAI has never seen, does it predict the loss of the largest models from smaller ones better than the standard practice (fitting a law of a form chosen in advance) and than general-purpose symbolic regression?

**Status.** The protocol was fixed on 30 September 2026, before any run: hypotheses, data with held-out rows, code, baselines and the NAI version. The run followed it with no deviation. The protocol is in [PROTOCOL.md](PROTOCOL.md) and the detailed report in [REPORT.md](REPORT.md).

**Main result.** NAI's mean R² over the four tasks is **0.733**, against **−0.412** for general symbolic regression (the engine of PySR) at equal time. Of the three registered hypotheses, **H2 and H3 are supported and H1 is not**.

## 1. Hypotheses (registered)

| | Hypothesis | Test |
|---|---|---|
| **H1** | Where the standard form is right (E1, E2), NAI is not worse than the Chinchilla fit | NAI's mean R² ≥ B1 − 0.02 on each of E1 and E2 |
| **H2** | Where the standard form is incomplete (E3: learning rate × batch), NAI is better than the Chinchilla fit and not worse than the literature form | NAI > B1 and NAI ≥ B3 − 0.02 on E3 |
| **H3** | NAI is better than general symbolic regression at equal time | mean over E1–E4 of NAI's R² > that of B4 |

E4 is exploratory. The full protocol is in [PROTOCOL.md](PROTOCOL.md).

## 2. Data

None of the sources is used by SLDBench. In every task the held-out rows are the largest models.

| Task | Source (license) | Input → target | Groups | Train / test | Held out |
|---|---|---|---|---|---|
| E1 | Epoch AI's reconstruction of the Chinchilla data (Besiroglu et al. 2024; no license stated) | N, D → loss | 1 | 188 / 57 | N > 2·10⁹ |
| E2 | Porian et al. 2024, cosine-decay runs (MIT) | N, D → loss | 2 corpora | 164 / 36 | N ≥ 2.8·10⁸ |
| E3 | Porian et al. 2024, learning-rate × batch sweep (MIT) | lr, batch, D, N → loss | 1 | 471 / 95 | the largest width |
| E4 | DataDecide, AI2 2025 (ODC-BY) | N → mean over 10 OLMES tasks of −ln(correct_prob_per_char) | 25 data recipes | 975 / 75 | 1B models |

The prepared task files are in [`protocol/bench/exp1/tasks/`](protocol/bench/exp1/tasks/). They were produced by `protocol/bench/exp1/prepare.py` from the raw sources, which [`reproduce/fetch_sources.sh`](reproduce/fetch_sources.sh) downloads.

## 3. Methods compared

| | Method |
|---|---|
| **NAI v9** | The version and settings of SLDBench v9, no adjustment to these tasks (public build in `cli/`). Three replicates, each pooling three searches (seeds {1,2,3}, {4,5,6}, {7,8,9}). |
| **B1** | Chinchilla form E + A/N^α + B/D^β, fitted as in Hoffmann et al. 2022 (Huber loss on log L, L-BFGS, grid of starts) |
| **B2** | Power law without offset (least squares in log space) |
| **B3** | E3 only: Chinchilla form plus parabolas in ln lr and ln batch around optima that move with N and D (DeepSeek LLM / Step Law form) |
| **B4** | SymbolicRegression.jl 2.5.0 (PySR's engine), NAI's alphabet and inputs, one form with per-group parameters. **Budget: the same seconds per task as NAI.** Three replicates. |

All methods fit one form per task with per-group constants, on the same training rows.

## 4. Results

| Task | NAI v9 (mean ± sd) | B1 Chinchilla | B2 power law | B3 literature form | B4 PySR engine (mean) |
|---|---|---|---|---|---|
| E1 | **0.892 ± 0.046** | 0.813 | 0.524 | - | −2.460 |
| E2 | 0.775 ± 0.225 | **0.930** | 0.644 | - | 0.649 |
| E3 | 0.500 ± 0.005 | −0.000 | −0.375 | **0.504** | −0.368 |
| E4 | **0.765 ± 0.010** | 0.730 | 0.732 | - | 0.530 |
| Mean | **0.733** | - | - | - | −0.412 |

The replicates, secondary metrics (relative errors, regret of the learning-rate/batch choice), the analysis of where NAI is stronger and weaker, the comparison with SLDBench and the caveats are in [REPORT.md](REPORT.md).

## 5. Verifying and reproducing

The whole protocol (about 11 hours on a 12-thread PC):

```
JULIA=julia PY=python3 sh reproduce/reproduce.sh
```

The choice of form of a replicate, from its published candidates (seconds):

```
cd protocol/bench
mkdir -p ../../results-repro/work
cp -r ../../results/work/sldbench-exp1-p123 ../../results-repro/work/
python3 sld_rules.py --suite exp1 --cli ../../cli/nai_cli.exe --out ../../results-repro --tags p123 --rules all-mean \
  --common "--engine-groups --jobs 1 --threads 12 --transform both --select nested --monotone --delta 0.003"
# compare ../../results-repro/work/sldbench-exp1-p123-all-mean/ with ../../results/work/sldbench-exp1-p123-all-mean/
```

The files in `protocol/` are those of the run, with changes that do not affect the results: the run script takes NAI from `cli/` and Julia and Python from the environment; the data preparation writes the task files only; and the benchmark runner (`sldbench.py`, `sld_rules.py`) is reduced to the group mode and rules used here and reads NAI's English output. The protocol itself is [PROTOCOL.md](PROTOCOL.md).

## 6. Files

| Path | Content |
|---|---|
| `PROTOCOL.md` | The protocol, as fixed before the run |
| `REPORT.md` | The detailed report |
| `cli/` | The binary, [README](cli/README.md), [PARAMETERS](cli/PARAMETERS.md), examples |
| `protocol/` | Scripts (`bench/`), task data (`bench/exp1/tasks/`) and B4's Julia environment (`bench/exp1/sr/`) |
| `results/` | All outputs of the run (section 11 of the report) |
| `reproduce/` | `fetch_sources.sh` (raw sources), `reproduce.sh` (the protocol) |
