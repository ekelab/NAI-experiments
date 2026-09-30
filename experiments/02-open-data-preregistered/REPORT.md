# Experiment 02 - a pre-registered test on open scaling-law data: report

*Report date: 30 September 2026. Run: 30 September 2026, 00:33–11:11. The protocol, fixed before the run, calls this study "experiment 1".*

## Summary

**NAI** (No-Ansatz Inference) is a symbolic-regression system: it searches for the functional form of a law in data, rather than fitting the constants of a form chosen in advance. In this experiment it predicted the loss of the largest models from smaller ones, on four open datasets it had never seen. It was compared with the way scaling laws are fitted in practice and with a general-purpose symbolic-regression engine.

- **Mean R² over the four tasks: NAI 0.733**, versus −0.412 for the PySR engine (SymbolicRegression.jl) given the same wall-clock time.
- **On the learning-rate × batch-size sweep**, the standard Chinchilla fit scores R² ≈ 0. NAI finds the dependence on the learning rate by itself and **matches the literature form** that was given that dependence in advance (0.500 vs 0.504).
- **On one Chinchilla-type dataset NAI lost** to the Chinchilla fit (0.775 vs 0.930): one of its three replicates chose a form that fits the training range but oscillates beyond it.
- **Two of three pre-registered hypotheses were confirmed** (H2, H3); H1 was not.

## 1. Pre-registration

Before any run on these data, the hypotheses, the datasets with their held-out rows, all code (data preparation, baselines, scoring, run script), the Julia environment of the PySR engine and the NAI binary were frozen, and a dated public record of them was made.

| Item | Status |
|---|---|
| Protocol | [PROTOCOL.md](PROTOCOL.md), the English text of the protocol |
| Frozen files changed after the run | none (checked after the run) |
| Deviations from the protocol | none |
| Crashed or repeated runs | none |
| Checks done before registration | synthetic data and training rows only, never the held-out rows |

## 2. Data and tasks

None of the sources is used by SLDBench (whose tasks come from Chen 2025, Tao 2024, Lin 2024, Ye 2024, Krajewski 2024, Muennighoff 2023, Li 2025 and Wu & Lo 2024). In every task the held-out rows are the **largest models**, so every task is an extrapolation.

| Task | Source (license) | Inputs → target | Groups | Train / test rows | Held out |
|---|---|---|---|---|---|
| E1 `chinchilla_epoch` | Epoch AI reconstruction of Chinchilla (Besiroglu et al. 2024; no license stated) | N, D → loss | 1 | 188 / 57 | N > 2·10⁹ |
| E2 `porian_chinchilla` | Porian et al. 2024, cosine-decay runs (MIT) | N, D → loss | 2 corpora (RefinedWeb, OpenWebText2) | 164 / 36 | 4 largest widths, N ≥ 2.8·10⁸ |
| E3 `porian_lrbsz` | Porian et al. 2024, learning-rate × batch sweep (MIT) | lr, batch, D, N → loss | 1 | 471 / 95 | largest width, N = 1.7·10⁸ |
| E4 `datadecide` | DataDecide, AI2 2025 (ODC-BY) | N → mean over 10 OLMES tasks of −ln(correct_prob_per_char) | 25 data recipes | 975 / 75 | 1B models |

N for Porian et al. is 12·depth·width² (non-embedding) and D = steps · batch · 2048; for Epoch, D = C / 6N.

## 3. Methods

All methods fit **one form per task, with constants of its own for each group**.

| Method | What it is |
|---|---|
| **NAI v9** | The frozen version and settings of the SLDBench v9 run (public build in `cli/`): three searches with different random splits of the rows, candidates pooled, one form chosen by its accuracy on held-out rows beyond the fitted range. The only declared assumption is that loss does not rise with N and D. Three replicates: seeds {1,2,3}, {4,5,6}, {7,8,9}. |
| **B1 Chinchilla** | L = E + A/N^α + B/D^β (E4: E + A/N^α), fitted as in Hoffmann et al. 2022 (approach 3): Huber loss on log L, L-BFGS from their grid of starts. On E3 it sees N and D only, as in practice. |
| **B2 power law** | log L linear in log N and log D (least squares). |
| **B3 literature lr/batch form** (E3 only) | B1 plus parabolas in ln lr and ln batch around optima that move with N and D (the form of DeepSeek LLM 2024 / Step Law 2025); 200 starts, L-BFGS, then Nelder–Mead. |
| **B4 general symbolic regression** | SymbolicRegression.jl 2.5.0, the engine of PySR (Julia 1.13.1), with NAI's operator alphabet and inputs, a template with 4 parameters per group, and default model selection. **Budget: the same wall-clock seconds per task as NAI spent in that replicate.** Three replicates. |

Implementations were checked on synthetic data with known constants before registration: B1 recovers the Chinchilla constants, B3 recovers a synthetic lr/batch law to 10⁻¹³, and B4 reaches R² 0.95–0.999 on the synthetic blind suite.

## 4. Hypotheses (as registered)

- **H1** - where the standard form is right (E1, E2), NAI's mean R² is not below B1's minus 0.02.
- **H2** - where the standard form is incomplete (E3), NAI's mean R² is above B1's and not below B3's minus 0.02.
- **H3** - NAI's mean R² over E1–E4 is above B4's, at equal time.

The margin of 0.02 was chosen in advance, from NAI's replicate spread on SLDBench (±0.026). E4 is exploratory: it has no hypothesis of its own and enters only the mean for H3.

## 5. Results

### 5.1 R² on the held-out largest models

| Task | NAI r1 | NAI r2 | NAI r3 | **NAI mean ± sd** | B1 | B2 | B3 | B4 r1 | B4 r2 | B4 r3 | **B4 mean** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 Chinchilla (Epoch) | 0.877 | 0.943 | 0.854 | **0.892 ± 0.046** | 0.813 | 0.524 | - | 0.875 | −9.058 | 0.802 | −2.460 |
| E2 Porian, Chinchilla | 0.920 | 0.888 | 0.516 | 0.775 ± 0.225 | **0.930** | 0.644 | - | 0.573 | 0.500 | 0.874 | 0.649 |
| E3 Porian, lr × batch | 0.497 | 0.498 | 0.505 | 0.500 ± 0.005 | −0.000 | −0.375 | **0.504** | 0.065 | −0.948 | −0.219 | −0.368 |
| E4 DataDecide | 0.766 | 0.774 | 0.754 | **0.765 ± 0.010** | 0.730 | 0.732 | - | 0.671 | 0.208 | 0.711 | 0.530 |
| **Mean over tasks** | 0.765 | 0.776 | 0.657 | **0.733 ± 0.065** | - | - | - | 0.546 | −2.325 | 0.542 | −0.412 |

### 5.2 Hypotheses

| Hypothesis | Test | Result | Verdict |
|---|---|---|---|
| H1, E1 | NAI ≥ B1 − 0.02 | 0.892 ≥ 0.793 | pass |
| H1, E2 | NAI ≥ B1 − 0.02 | 0.775 < 0.910 | **fail** |
| **H1** | both tasks | - | **not supported** |
| **H2** | NAI > B1 and NAI ≥ B3 − 0.02 on E3 | 0.500 > −0.000; 0.500 ≥ 0.484 | **supported** |
| **H3** | mean NAI > mean B4 | 0.733 > −0.412 (0.544 without B4's worst replicate) | **supported** |

### 5.3 Secondary metrics

MAPE is the mean relative error over all held-out rows. "Far" is the mean relative error on the rows of the largest N. "Regret" (E3) is how much worse, in loss, the configuration of learning rate and batch predicted best is than the truly best one at each token budget. In E3 and E4 all held-out rows belong to one model size, so "far" equals MAPE.

| Task | Metric | NAI r1 / r2 / r3 | B1 | B2 | B3 | B4 r1 / r2 / r3 |
|---|---|---|---|---|---|---|
| E1 | MAPE, % | 1.89 / 1.39 / 2.09 | 2.12 | 4.51 | - | 1.89 / 5.69 / 2.66 |
| E1 | far, % | 2.05 / 1.83 / 2.47 | 2.14 | 6.27 | - | 1.26 / 0.74 / 2.39 |
| E2 | MAPE, % | 1.85 / 2.08 / 5.04 | 1.94 | 4.34 | - | 5.29 / 4.69 / 2.35 |
| E2 | far, % | 3.03 / 3.25 / 8.72 | **1.93** | 3.15 | - | 8.40 / 7.66 / 3.67 |
| E3 | MAPE, % | 1.67 / 1.71 / 1.56 | 2.50 | 3.19 | 1.61 | 2.58 / 3.98 / 2.98 |
| E3 | regret | 0.018 / 0.018 / 0.012 | 0.079 | 0.079 | **0.012** | 0.073 / 0.039 / 0.050 |
| E4 | MAPE, % | 7.59 / 7.52 / 8.01 | 8.33 | **4.93** | - | 9.44 / 16.34 / 7.45 |

### 5.4 The nine single NAI searches (R², search time in seconds)

Each replicate of NAI v9 pools three of these.

| Seed | E1 | E2 | E3 | E4 |
|---|---|---|---|---|
| 1 | 0.855 (99) | 0.516 (79) | 0.605 (876) | 0.763 (990) |
| 2 | 0.877 (140) | 0.920 (84) | 0.497 (893) | 0.774 (1067) |
| 3 | 0.829 (130) | −0.071 (108) | 0.605 (875) | 0.766 (998) |
| 4 | 0.845 (136) | 0.541 (73) | 0.498 (867) | 0.721 (917) |
| 5 | 0.841 (94) | 0.888 (77) | 0.605 (838) | 0.774 (991) |
| 6 | 0.943 (93) | 0.828 (84) | 0.605 (837) | 0.739 (983) |
| 7 | 0.836 (91) | 0.881 (74) | 0.598 (835) | 0.754 (983) |
| 8 | 0.850 (90) | 0.740 (79) | 0.605 (850) | 0.774 (981) |
| 9 | 0.854 (92) | 0.516 (79) | 0.505 (839) | 0.715 (996) |
| **Mean** | 0.859 | 0.640 | 0.569 | 0.753 |

Mean over tasks: 0.705 for a single search, 0.733 for the pooled method.

### 5.5 Compute

Everything ran on one Windows PC (12 threads, 16 GB RAM), with no GPU and no language model. One NAI search of all four tasks takes about 35 minutes, and one replicate of NAI v9 is three searches. B4 received the same seconds per task and replicate: 232–369 s on E1 and E2, 2524–3055 s on E3 and E4.

## 6. Where NAI is stronger - and why

**1. When the standard form of the law is incomplete (E3).** With the learning rate and batch size as inputs, a Chinchilla fit cannot express their effect and scores R² ≈ 0; its choice of configuration costs 0.079 in loss. NAI reaches the literature form's R² (0.500 vs 0.504), and its regret is close to it (0.012–0.018 vs 0.012). The literature form was told where the optimum is and how it moves. NAI found such a dependence by itself: for example, the factors 1.0087^((ln lr)²) · 1.0056^(ln lr · ln N) in one replicate's form are a valley in ln lr whose minimum moves with the model size. **Why:** NAI searches the space of forms, including logarithms of the inputs and their products, instead of fitting one ansatz. Terms that practitioners add from domain knowledge can be discovered.

**2. Stable extrapolation where data are rich (E1, E4).** On E1 NAI beats the Chinchilla fit (0.892 vs 0.813) on the very data the Chinchilla law was made for. On E4, 25 data recipes share one form: NAI beats both B1 and B2 on R² (0.765 vs 0.730 / 0.732), with a replicate spread of only 0.010. **Why:** each form is judged on held-out training rows beyond the range it was fitted on. That favours forms that extrapolate, not forms that interpolate best. Pooling three searches with different splits reduces the dependence on one split.

**3. Against general symbolic regression at equal time (E1–E4).** The PySR engine is far less stable: its R² on E1 ranges from 0.875 to −9.06 across replicates. On E3 it never recovers the learning-rate dependence (−0.95 to 0.07). **Why:** B4 chooses its equation by training loss and complexity. Nothing in that choice tests extrapolation, so an equation that fits the training range can explode beyond it. NAI's choice of formula is built for extrapolation. The difference is in the choice as much as in the search.

## 7. Where NAI is weaker - and why

**1. A Chinchilla-type dataset where the standard form is right (E2).** NAI's mean is 0.775 against 0.930. Two replicates are close to the Chinchilla fit (0.920, 0.888); the third chose a form with a term oscillating in D (a sine of ln D), and it misses beyond the data (0.516). Single searches range from −0.07 to 0.92 on this task. **Why:** the held-out validation rows lie inside the training range. Two forms that agree there but diverge beyond it cannot be told apart, and flexible oscillating terms fit small data (two groups, 164 rows) well. The Chinchilla fit has the correct inductive bias built in, so there is nothing for it to get wrong. NAI does not have that bias.

**2. The choice, not the search, is the bottleneck.** On E3, five of the nine single searches found a form scoring 0.605 on the held-out models, but the pooled choice took forms scoring 0.50. On E2 a good form was in the pool in every replicate. **Why:** the internal selection signal is noisy. On E1 and E2 the chosen forms score 0.95–0.99 on the internal held-out rows, but that signal cannot rank forms by how far out they stay correct.

**3. Not every metric agrees.** On E4 the plain power law (B2) has a lower relative error than NAI (4.9 % vs 7.7 %) despite a lower R². On E2 the Chinchilla fit is more accurate on the largest models (1.9 % vs 5.0 %). On E1 B4 is more accurate on the largest models in two replicates. **Why:** NAI and the selection rule optimise squared error (R²). A law fitted in log space, like B2, optimises relative error instead. NAI's advantage is in R², the metric of the registration and of SLDBench, and not uniformly in every metric.

**4. Interpretability.** The forms NAI chose here are approximations, not clean laws, for example: 21.29 − 0.245·(ln N)^(−sin √ln D) − 2.946·√(ln D + sin sin ln D). They extrapolate better than the standard fit on three tasks of four, but a human cannot read them, whereas the Chinchilla law and the literature lr/batch form are readable by construction. **Why:** on noisy real data NAI works in its "approximation" mode, where no exact law is found and the best-extrapolating expression is kept. Its operator set includes sin and cos, and in this mode they act as flexible basis functions. NAI's promise of "a formula, not a black box" holds here only formally.

## 8. Comparison with SLDBench - and its caveats

The same frozen method (NAI v9) was run on SLDBench ([experiment 01](../01-sldbench/REPORT.md)), whose protocol matches this one: R² on extrapolated test rows, one form with per-group constants, the test never used in any choice, and the mean over repeated runs reported.

| SLDBench task | NAI v9 (3 replicates) | SLDAgent / GPT-5 (5 runs) | Human law |
|---|---|---|---|
| parallel | 0.985 | 1.000 | 1.000 |
| vocab_size | 0.971 | 0.987 | 0.966 |
| sft | 0.982 | 0.993 | 0.957 |
| domain_mix | **0.991** | 0.988 | 0.671 |
| moe | 0.656 | 0.773 | 0.703 |
| d_constrain | 0.768 | 0.944 | 0.911 |
| lr&bsz | 0.372 | 0.604 | −0.076 |
| u_shape | −0.076 | −0.305 | −1.000 |
| **Mean** | **0.706 ± 0.026** | 0.748 | 0.517 |

Other systems in the SLDBench paper: Goose 0.695, SLDAgent with Claude Sonnet 4.5 0.590, CodeX 0.550.

**What the two tests have in common:**
- **Same strength:** NAI does well where no simple textbook form fits, as in domain mixtures on SLDBench and the lr/batch sweep here.
- **Same weakness:** the choice of form is unstable on some tasks - moe, d_constrain and u_shape on SLDBench, E2 here. The search finds good forms more often than the choice picks them.
- **Different picture on learning rate and batch:** on SLDBench NAI trails SLDAgent on lr&bsz (0.372 vs 0.604), while here it matches the literature form (0.500 vs 0.504). The tasks differ: four inputs from the Step Law runs versus a constant-schedule sweep from Porian et al. So this is not evidence either way about SLDAgent on these data.

**Caveats of the comparison:**
- **Different tasks, not comparable R².** R² depends on how far the test extrapolates and how much the target varies. 0.733 here and 0.706 on SLDBench measure different things.
- **SLDAgent was not run here.** The only LLM-agent result is SLDBench's own; this experiment compares NAI with classical fits and with a general symbolic-regression engine.
- **This experiment is the cleaner evidence.** SLDBench was used during NAI's development, and its weak tasks set the direction of work. Rules were validated on a blind synthetic suite, but that is still a mild form of benchmark overfitting. This experiment was pre-registered on data NAI had never seen.
- **Different prior knowledge.** SLDAgent's language model may have read the papers SLDBench is built from. NAI has no domain knowledge. The literature form B3 here also starts from known structure.
- **Different compute.** SLDAgent uses about 30–60 minutes per task plus GPT-5 calls (its README); NAI uses one PC with no GPU or LLM, and one NAI v9 replicate costs three searches.

## 9. Caveats of this experiment

- **Small scale.** Four tasks and three replicates; the H1 and H2 margins (0.02) are of the order of NAI's replicate spread on some tasks.
- **Imperfect data.** E1 was digitised from a figure of the Chinchilla paper, so it carries extraction noise. N and D for Porian et al. are approximations (non-embedding 12·d·w², tokens from steps × batch × 2048).
- **E4's target was forced by availability.** The first choice, bits per byte, is missing at the final checkpoints of large models. The replacement was fixed before registration and before any look at its values, but it remains a proxy of downstream quality.
- **B1 on E3 is weak by construction.** It sees N and D only, as a Chinchilla fit would; the fair comparison there is B3.
- **B4 ran with default settings, not tuned.** A tuned PySR, or one with an extrapolation-aware model choice, could do better. Equal wall-clock time also does not mean equal effective compute: the two programs parallelise differently.
- **One machine, time-limited searches.** NAI's single searches depend slightly on machine load, since time-limited searches reach different depths.
- **Not a prediction of the future.** The data are public; NAI had not seen them, but the literature form had. A prospective test, with the prediction published before the large models are trained, is planned as a separate experiment.

## 10. Conclusions

1. **NAI adds value where the textbook ansatz is incomplete.** On the learning-rate × batch sweep it discovers the kind of dependence practitioners add from domain knowledge, and it matches that hand-made form.
2. **Where the textbook ansatz is right, NAI is competitive but not reliable.** It is better on one dataset, worse on another, and the gap comes from an unstable choice of form, not from a missing form.
3. **Against general symbolic regression, NAI is far more stable at equal time.** The main reason is its extrapolation-aware choice of form.
4. **The next work is the same as on SLDBench:** a more robust choice of form, and cleaner, readable laws instead of oscillating approximations.

## 11. Reproducing

With Git Bash or MSYS2, Python 3 (numpy, pandas, scipy) and Julia 1.13:

```
JULIA=julia PY=python3 sh reproduce/reproduce.sh      # the whole protocol, ~11 h on one PC
```

`reproduce/reproduce.sh` installs the Julia packages of B4 at the versions of the run and runs `protocol/bench/exp1/run.sh`, the protocol step by step. NAI is the public build in `cli/`, which reproduces the published choices of form exactly ([README](README.md#5-verifying-and-reproducing)). The task files are included; to regenerate them from the raw sources, run `sh reproduce/fetch_sources.sh`, then `python3 protocol/bench/exp1/prepare.py`, and compare the result with the included files (`git status` shows any difference).

| What | Where |
|---|---|
| Scores (all methods, replicates, metrics) | `results/scores.json` |
| Per-search results of NAI | `results/nai_sldbench_exp1-s*.csv`, `.log` |
| NAI's replicates (pooled) | `results/nai_sldbench_exp1-p*-all-mean.csv`, `chosen_exp1-p*-all-mean.json` |
| NAI's chosen forms with per-group constants, candidates, predictions | `results/work/sldbench-exp1-*/` |
| Baselines B1–B3 predictions | `results/baselines/` |
| B4 equations (full Pareto front) and predictions | `results/b4-r1/`, `b4-r2/`, `b4-r3/` |
| The log of the whole run | `results/run.log` |
| Data preparation (reads the sources, writes the tasks) | `protocol/bench/exp1/prepare.py` |
