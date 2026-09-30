# Experiment 02 - protocol (fixed before the run)

This is the protocol of experiment 02 as it was fixed on 30 September 2026, before any run of NAI or of the baselines on the experiment's data. It fixed the hypotheses, the data with their held-out rows, the scripts and the NAI version.

- **Language.** This file is the English text of the registration document, which was written in another language and is not included here.
- **Naming.** "Experiment 1" in the registration is "experiment 02" in this repository.
- **Edits.** References to files have been adapted to this repository; the substance of the protocol is unchanged.
- **Paths.** They are relative to `protocol/`. `docs/EXPERIMENTS.md` and `docs/REPORT.md` refer to the author's working documents, which are not part of this repository; the report of this experiment is [REPORT.md](REPORT.md).

## 1. Question and hypotheses

Does NAI predict the behaviour of large models from small ones, on open data it has not seen, better than the standard fit of a law whose form is chosen in advance?

- **H1 (not worse where the standard is right).** On tasks E1 and E2 (loss as a function of N and D), NAI's R², averaged over replicates, is not below the R² of baseline B1 minus 0.02.
- **H2 (better where the standard form is incomplete).** On task E3 (learning rate and batch size), NAI's mean R² is above B1's, and not below B3's minus 0.02.
- **H3 (better than general symbolic regression).** The mean over tasks E1–E4 of NAI's R² (averaged over replicates) is above that of B4 (SymbolicRegression.jl, the engine of PySR), at equal time.
- **E4 (DataDecide)** is not part of H1 or H2. It has no hypothesis of its own; its result is reported as is and enters only the mean for H3.

The threshold 0.02 was chosen in advance as the margin of "not worse", given NAI's spread between replicates on SLDBench (±0.026).

## 2. Data

None of the sources is used in SLDBench, whose tasks come from Chen 2025, Tao 2024, Lin 2024, Ye 2024, Krajewski 2024, Muennighoff 2023, Li 2025 and Wu & Lo 2024.

| Task | Source | License | Input → target | Groups | Held out (test) |
|---|---|---|---|---|---|
| E1 `chinchilla_epoch` | Epoch AI, Chinchilla reconstruction (Besiroglu et al. 2024) | not stated (data from a figure of the paper) | N, D → loss | 1 | N > 2·10⁹ (57 of 245) |
| E2 `porian_chinchilla` | Porian et al. 2024, cosine runs with base hyperparameters | MIT | N, D → loss | 2 (RefinedWeb, OpenWebText2) | N ≥ 2.8·10⁸, the 4 largest widths (36 of 200) |
| E3 `porian_lrbsz` | Porian et al. 2024, lr × batch grid | MIT | lr, batch, D, N → loss | 1 | the largest width, N = 1.7·10⁸ (95 of 566) |
| E4 `datadecide` | DataDecide (AI2 2025) | ODC-BY | N → mean over 10 OLMES tasks of −ln(correct_prob_per_char) | 25 (data recipes) | 1B models (75 of 1050) |

Transformation: `bench/exp1/prepare.py`, where the rules for the held-out rows are written. For Porian, N = 12·depth·width² and D = steps · batch · 2048; for Epoch, D = C / 6N. For E4 the target was chosen before any look at its values: `bits_per_byte_corr` is missing at the final checkpoints of the large models, so −ln(correct_prob_per_char) is used, the cross-entropy of the correct answer per character.

## 3. NAI

- **Version frozen:** NAI v9, the same as for SLDBench v9.
- **Settings:** as in SLDBench v9, with no adjustment to these tasks:
  `bench/sldbench.py --suite exp1 --engine-groups --holdout all --jobs 2 --threads 12 --seconds 180 --seconds-single 900 --memory-single 3072 --total-memory 4096 --transform both --select nested --monotone --delta 0.003`.
- **Assumptions** (declared in `tasks.json`): the loss does not rise with N and D; lr and batch are free. Limit on the number of constants: 7 for E1 and E2, 10 for E3, 6 for E4.
- **Method v9:** three searches with seeds s, s+1 and s+2, whose candidates are pooled (`bench/sld_rules.py`, rule `all-mean`). **Three replicates:** seeds {1, 2, 3}, {4, 5, 6} and {7, 8, 9}. The result is the mean and spread over replicates.

## 4. Baselines (`bench/exp1/baselines.py`)

Each group has its own constants, as for NAI.

- **B1:** the Chinchilla form L = E + A/N^α + B/D^β (for E4: E + A/N^α). Fitted as in Hoffmann et al. 2022, approach 3: Huber loss (δ = 10⁻³) on log L, L-BFGS from a grid of starts (a, b ∈ {0, 5, …, 25}, e ∈ {−1, −0.5, …, 1}, α, β ∈ {0, 0.5, …, 2}), the best result kept. For E3 it sees only N and D.
- **B2:** a power law without offset, log L linear in log N and log D (least squares).
- **B3** (E3 only): B1 plus parabolas in ln lr and ln batch around optima that depend on N and D (the form of the laws of DeepSeek LLM 2024 / Step Law 2025). The base is B1 fitted on the best run of each (N, D) pair. Then 200 starts (seed 0; a quarter with zero slopes of the optima), L-BFGS, and the best refined by Nelder–Mead.
- **B4:** general symbolic regression. SymbolicRegression.jl 2.5.0 (the engine of PySR, Cranmer 2023; the versions of all packages are in `bench/exp1/sr/Manifest.toml`), Julia 1.13.1, `bench/exp1/sr/b4_sr.jl`.
  - Operations: + − × ÷ ^, exp, log, sqrt, sin, cos (NAI's alphabet on continuous data).
  - Inputs: the variables and ln of the scale variables, as in NAI's second search.
  - With several groups: one form with 4 parameters of its own for each group (a template expression).
  - maxsize 30; default choice (choose_best); 12 threads.
  - **Time:** for each task in each replicate, as many seconds as NAI spent on that task in that replicate (the sum of `search_seconds` of its three seeds). Replicate r uses seed r (1, 2, 3).

The implementation was checked before registration on synthetic data with known constants (`baselines.py --selftest`: B1 recovers the Chinchilla constants; B3 recovers a synthetic law with optima to 10⁻¹³) and on the blind synthetic suite (B4: R² 0.947 on Chinchilla and 0.999 on fine-tuning in 60 s). It was not checked on the experiment's data.

## 5. Metrics (`bench/exp1/evaluate.py`)

- **R²** on the held-out rows, as in SLDBench (a non-numeric prediction gives −1).
- **RelFar:** the mean relative error on the rows with the largest N, in %.
- **MAPE:** the mean relative error on all held-out rows, in %.
- **Regret** (E3): for each D among the held-out rows, the configuration (lr, batch) that is best by the prediction is taken. The regret is its true loss minus the best true loss at that D, averaged over D.

The main metric for the hypotheses is R². All the others are reported, in any case: for NAI (mean and spread over three replicates), for B1–B3 (deterministic) and for B4 (mean and spread over three replicates).

## 6. Order and rules

1. Make a dated public record of the protocol before the run.
2. Run NAI (9 seeds) and pool the replicates; then B1–B3; then B4, with the time budget taken from NAI's log. The whole protocol, step by step, is `bench/exp1/run.sh`.
3. Compute the metrics and write the report in `docs/REPORT.md`, including negative and unexpected results.
4. No changes to NAI's method after publication. If a run fails for a technical reason (memory, a crash), it is repeated with the same binary and the same seed, and this is noted in the report.

## 7. What the experiment will not show

The data are published. NAI has not seen them, but this is not a prediction of the future. Baseline B3 knows the form of the law from the literature. A check on truly new data is experiment 2 (`docs/EXPERIMENTS.md`).
