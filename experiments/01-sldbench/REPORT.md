# Experiment 01 - SLDBench: report

*Version: NAI v9, frozen before the runs; `cli/nai_cli.exe` is its public build. Runs: 29 September 2026. All scores re-computed with SLDBench's official evaluator.*

## 1. Summary

On the eight tasks of SLDBench, NAI v9 reaches a mean extrapolation R² of **0.706 ± 0.026** (mean ± standard deviation over three independent replicates). It uses no language model: one PC, 12 threads, about 3.4 hours of search per replicate.

- **Ranking.** Among the systems reported in the SLDBench paper, this places NAI between SLDAgent with GPT-5 (0.748) and Goose (0.695), above the human-derived laws (0.517).
- **Strong tasks.** NAI is at the level of the best system on five tasks (parallel, vocabulary, fine-tuning, domain mixture, U-shaped). It exceeds SLDAgent on domain mixture (0.991 vs 0.988) and on the U-shaped task (−0.076 vs −0.305).
- **Weak tasks.** Most of the gap comes from three tasks: mixture of experts (0.656 vs 0.773), data-constrained (0.768 vs 0.944) and learning rate × batch size (0.372 vs 0.604). In each of them the laws of the literature are larger than the formulas NAI reaches, and NAI's choice between its candidates varies across replicates.
- **Caveat.** SLDBench was used during NAI's development, so this result is not a blind test. Section 9 explains what that implies. Experiment 02 is the blind test.

## 2. Benchmark

SLDBench (Lin et al., ICLR 2026) collects eight scaling-law studies. For each, a system receives the training rows and must output one law: a functional form with constants fitted per group, at most the task's number of constants. The law is scored on held-out rows that extrapolate: larger models, more data or new mixtures. The score is R² = 1 − MSE/Var over the concatenated test rows of all groups. A non-finite prediction counts as R² = −1. The benchmark's score is the mean of the eight task scores.

The task list, sources, groups and constant limits are in the [README](README.md#2-tasks-and-data).

## 3. Method

NAI v9 in group mode ([docs/METHOD.md](../../docs/METHOD.md#group-mode-one-form-constants-per-group)):

- **Search.** Each group is searched within a time budget: 180 s per search, 900 s for single-group tasks.
- **Candidates.** All candidate forms of three searches (seeds s, s+1, s+2) are pooled.
- **Choice.** One form is chosen for all groups by its accuracy on held-out training rows beyond the range it was fitted on. Fewer constants are preferred when scores are close. Forms whose loss rises with model or data size beyond the data are rejected (declared assumption).
- **Final fit.** The chosen form's constants are refitted per group on all training rows.

Replicates: seeds {1,2,3}, {4,5,6} and {7,8,9}. The test rows enter no step.

## 4. Protocol and verification

1. The binary was frozen **before** the nine searches were run.
2. The nine searches were run once each, one after another (29 September 2026, 04:27–14:38; about 68 minutes of search per seed). One task in one search failed for lack of memory during a preliminary run (v8, seed 3). In v9 a total memory limit of 4 GB prevents this, and no search failed.
3. The three replicates were formed from the saved candidates, and their test predictions computed.
4. **Official evaluator.** Every chosen form was converted into a program in SLDBench's format and scored with the benchmark's official `evaluator.py`, unchanged. All three replicates and all nine single searches agree with our computation to four decimals, with one exception on a single search: seed 9, u_shape, official −0.5580 vs ours −0.5527. There the official refit of a trigonometric form converges slightly differently. Official values are used throughout.
5. **Data.** The local copies of the data were compared value for value with the Hugging Face source. A fresh download on 30 September 2026 was identical.
6. **The published binary.** The public build in `cli/` was checked against the build used in the runs: re-running the choice of the three replicates reproduces their forms, constants and predictions byte for byte.

## 5. Results

### 5.1 Main result: NAI v9, three replicates

| Task | Replicate 1 (seeds 1–3) | Replicate 2 (4–6) | Replicate 3 (7–9) | **Mean ± sd** | Human law | SLDAgent / GPT-5 |
|---|---|---|---|---|---|---|
| parallel | 0.9934 | 0.9935 | 0.9681 | **0.985 ± 0.015** | 1.000 | 1.000 |
| vocab_size | 0.9663 | 0.9716 | 0.9741 | **0.971 ± 0.004** | 0.966 | 0.987 |
| sft | 0.9849 | 0.9830 | 0.9793 | **0.982 ± 0.003** | 0.957 | 0.993 |
| domain_mix | 0.9918 | 0.9934 | 0.9880 | **0.991 ± 0.003** | 0.671 | 0.988 |
| moe | 0.6906 | 0.7608 | 0.5166 | **0.656 ± 0.126** | 0.703 | 0.773 |
| d_constrain | 0.6636 | 0.7434 | 0.8983 | **0.768 ± 0.119** | 0.911 | 0.944 |
| lr&bsz | 0.4997 | 0.3104 | 0.3060 | **0.372 ± 0.111** | −0.076 | 0.604 |
| u_shape | 0.0265 | −0.0464 | −0.2093 | **−0.076 ± 0.121** | −1.000 | −0.305 |
| **Mean** | **0.727** | **0.714** | **0.678** | **0.706 ± 0.026** | 0.517 | 0.748 |

### 5.2 The nine single searches (the previous method, v8)

| Seed | parallel | vocab | sft | domain_mix | moe | d_constrain | lr&bsz | u_shape | Mean |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.999 | 0.933 | 0.983 | 0.990 | 0.458 | 0.664 | 0.519 | 0.209 | 0.719 |
| 2 | 0.999 | 0.966 | 0.989 | 0.993 | 0.691 | 0.958 | 0.500 | −0.289 | 0.726 |
| 3 | 0.993 | 0.967 | 0.985 | 0.990 | 0.495 | 0.743 | 0.020 | 0.027 | 0.653 |
| 4 | 0.999 | 0.978 | 0.975 | 0.995 | 0.511 | 0.743 | 0.519 | −0.046 | 0.709 |
| 5 | 0.994 | 0.972 | 0.972 | 0.990 | 0.761 | 0.663 | −0.123 | −0.361 | 0.609 |
| 6 | 0.994 | 0.979 | 0.983 | 0.991 | 0.587 | 0.885 | 0.310 | 0.209 | 0.742 |
| 7 | 0.993 | 0.968 | 0.983 | 0.990 | 0.485 | 0.703 | 0.354 | 0.027 | 0.688 |
| 8 | 1.000 | 0.974 | 0.979 | 0.992 | 0.514 | 0.898 | 0.020 | −0.209 | 0.646 |
| 9 | 0.968 | 0.967 | 0.983 | 0.986 | 0.517 | 0.663 | 0.306 | −0.558 | 0.604 |
| **Mean** | 0.993 | 0.967 | 0.981 | 0.991 | 0.557 | 0.769 | 0.270 | −0.110 | **0.677 ± 0.047** |

**Pooling three searches** raises the mean from 0.677 to 0.706 and reduces the spread of the benchmark score from 0.047 to 0.026. The gain comes from mixture of experts (+0.10), learning rate × batch (+0.10) and the U-shaped task (+0.03). On data-constrained pooling changes nothing (0.768 vs 0.769).

### 5.3 Position among published systems

Scores of other systems are those of the SLDBench paper (Tables 2–3). Agents are averaged over five runs; human laws are fixed.

| System | Uses an LLM | Mean R² |
|---|---|---|
| SLDAgent (GPT-5) | yes | 0.748 |
| **NAI v9** (this work, 3 replicates) | **no** | **0.706 ± 0.026** |
| Goose | yes | 0.695 |
| SLDAgent (Claude Sonnet 4.5) | yes | 0.590 |
| CodeX | yes | 0.550 |
| Human-derived laws | - | 0.517 |
| SLDAgent (Gemini 2.5 Flash) | yes | 0.506 |

### 5.4 Chosen forms (replicate 1)

The laws NAI chose, with the constants of one group:

| Task | Chosen form (constants of the first group) | Constants |
|---|---|---|
| parallel | 112 / (ln N)^1.3296 | 1 |
| moe | 1.5314 + 31143 · (ln N − ln E) / (ln N)^4.4963 | 2 |
| u_shape | 16.924 − 17.467 · 1.01063^(−log_flops) | 3 |
| vocab_size | −5.310 + 80783/√C + 107.60·√(V/C) − 4.602·10⁸/C − 3.121·10⁻⁸·√(N·V) | 5 |
| domain_mix, domain 1 | 4.1969 · 0.72756^(cos p₂ + p₁^¼) | 2 |
| sft | 4.4214 · 0.999827^((ln D)^3.2652) · 1.00401^exp(sin ln D) | 4 |
| d_constrain | 2.1733 + 12.977·(ln ln ln D)^(−ln N) + 7191.1·√exp(cos ln D − ln U) | 3 |
| lr&bsz | 3.357 − 2.9·10⁻⁴·(ln N)^((ln D)^0.314) + 7.74·10⁵·B/D + 63.56·2^(ln lr − ln B) + 0.0484·(sin ln D − cos ln lr) | 5 |

Notation: N parameters, E experts, C characters, V vocabulary, D data, U unique tokens, B batch, pᵢ domain proportions. All forms with constants for every group are in `results/v9/chosen_*.json`.

**The forms fall into two kinds.**
- **Readable.** Parallel, mixture of experts and the U-shaped task have short forms with few constants.
- **Approximations.** The others use flexible terms, some trigonometric in logarithms of the inputs (sin ln D, cos ln lr). These act as basis functions, not as mechanisms, and should not be read as laws.

## 6. Development and validation

### 6.1 History of versions on SLDBench

| Version | Date | What changed | Mean R² | Runs |
|---|---|---|---|---|
| v1 | 27 Sep | one form per task, constants per group (the benchmark's rules) | 0.623 | 1 |
| v2 | 27 Sep | approximation mode: choice by held-out extrapolation | 0.656 | 1 |
| round 3 | 28 Sep | sin and cos on continuous data; other engine changes | 0.655 | 1 |
| v3 | 28 Sep | choice by extrapolation on held-out training rows | 0.678 | 1 |
| v4 | 28 Sep | a second search on transformed inputs | 0.679 | 1 |
| v5 | 28 Sep | 900 s for single-group tasks; monotonicity assumption; preference for fewer constants | 0.717 → 0.731 | 1 each |
| v8 | 28–29 Sep | choice inside the engine; search sped up 30 % (identical results) | 0.726 / 0.550 / 0.717 = **0.664 ± 0.10** | 3 |
| **v9** | 29 Sep | broader held-out criterion; three searches pooled | **0.706 ± 0.026** | 3 (9 searches) |

**v1–v5 are single runs made while developing against the test.** The replicates of v8 showed that the 0.731 of v5 was a favourable run: the same configuration spreads from 0.55 to 0.73 across seeds. From v8 on, only replicated results are reported, and v9's changes were chosen without consulting SLDBench's test (section 6.2).

### 6.2 How v9's changes were chosen

v9's two changes were compared, together with alternatives, **on a blind synthetic suite**: laws from the literature with published constants and noise, generated before this work. Each rule was re-applied to the same saved candidates, so that the rules differ and nothing else does.

| Rule of choice (blind suite, 6 laws) | Single searches (6 seeds) | Pooled, 3 searches (2 replicates) |
|---|---|---|
| v8's held-out criterion | 0.9687 ± 0.019 | 0.9782 ± 0.005 |
| **v9's broader held-out criterion** | **0.9721 ± 0.015** | **0.9782 ± 0.005** |
| a stricter, worst-case score | 0.9613 (3 seeds) | - |
| + a stronger extrapolation constraint (two variants) | 0.9654 / 0.9603 | 0.9731 / 0.9739 |
| + a restricted alphabet for the chosen form | 0.9538 | 0.9749 |
| combining 6 searches | - | 0.9801 (1 replicate) |

After v9 was frozen and scored, further changes were tested on the blind suite (with an added four-input learning-rate × batch law) and on an internal split of SLDBench's **training** rows: the top 30 % along the extrapolation variable held out, and the test file not opened.

| Change | Blind suite | Internal split | Decision |
|---|---|---|---|
| larger formulas from a growth strategy, variant 1 | 0.9640 vs 0.9721 | - | rejected |
| larger formulas, variant 2 (four-input law) | 0.968 vs 0.983 | - | rejected |
| more free constants per form | - | 0.760 vs 0.810 | rejected |
| combining 6 searches | 0.9801 vs 0.9782 | 0.883 vs 0.845 | not adopted (1 replicate, twice the cost) |

The internal split reproduces the pattern of the test. Its single searches spread widely (0.29–0.86), pooling helps (0.724 → 0.845), and the weak tasks are the same. So the weaknesses are properties of the search and choice, not of chance in the test rows. The full protocol is in [docs/VALIDATION-PROTOCOL.md](../../docs/VALIDATION-PROTOCOL.md).

## 7. Where NAI is stronger, where weaker - and why

**Stronger.**
- **Smooth, well-sampled laws** (parallel 0.985, vocabulary 0.971, fine-tuning 0.982, domain mixture 0.991). They are recovered at the level of the best system. Their structure is within reach of NAI's search, and the replicates agree (sd ≤ 0.015).
- **Domain mixture: above SLDAgent** (0.991 vs 0.988, and 0.671 for the human law). There is no textbook form for the dependence of each domain's loss on five proportions; NAI finds compact forms for each of the five targets (two constants each).
- **U-shaped task: better than every published system** (−0.076 vs −0.305 for SLDAgent and −1.0 for the human law). The test extrapolates across a change of trend that the training rows barely show, and no system predicts it well. NAI's choice rule rejects the forms that diverge beyond the data, so its errors stay moderate.

**Weaker.**
- **Data-constrained** (0.768 vs 0.944). The literature law (Muennighoff et al.) uses an *effective data* term, U + U·R*·(1 − e^(−R/R*)) with R = D/U − 1, inside a power law. The effective-data term alone is about 15 nodes, and it sits inside a power law: larger than the formulas NAI's search reaches within its budget. NAI approximates it with flexible terms (cos ln D) that fit the training range. Which approximation is chosen varies across replicates (0.66–0.90).
- **Learning rate × batch size** (0.372 vs 0.604). The loss has an optimum in learning rate and batch size that moves with model and data size, in four inputs. The chosen forms capture the monotone parts and only partly the optima. Pooling helps (0.270 → 0.372), but the forms that extrapolate well are rare among the candidates.
- **Mixture of experts** (0.656 vs 0.773). The chosen forms are readable, but the test extrapolates the dense parameter count 3.5 times beyond the training rows. Forms that are indistinguishable on the held-out training rows diverge there (0.52–0.76 across replicates).

**The common cause** is the gap between what the internal held-out score can see and what the test asks. The held-out validation rows lie inside the training range. Two forms that agree there but diverge beyond it cannot be ranked. The search often finds the better form: on moe and d_constrain, individual searches reach 0.76 and 0.96. But the choice does not reliably pick it.

## 8. Comparison with competitors - and caveats

| Aspect | NAI v9 | SLDAgent (GPT-5) | Human laws |
|---|---|---|---|
| Mean R² | 0.706 ± 0.026 (3 replicates) | 0.748 (5 runs) | 0.517 |
| Prior knowledge | none beyond operations and monotonicity | a large language model that may have read the source papers | the authors' domain knowledge |
| Output | a formula + per-group constants | a program (formula + fitting code) | a formula |
| Compute | ~3.4 h CPU per replicate, one PC, offline | ~30–60 min per task per run with GPT-5 calls (per its README) | - |
| Tasks at the level of the best system | 5 of 8 | - | - |

**Caveats of the comparison:**
- **Developer-level overfitting.** NAI was developed while observing SLDBench's test results up to v5. From v8 on, the rules in section 6 kept the test out of every decision, but the direction of work had already been set by what the test showed.
- **Different notions of "a run".** An agent's run is one evolutionary search. A replicate of NAI v9 pools three searches, at three times the cost of one; the single-search figure is 0.677.
- **Different knowledge.** SLDAgent's language model may have seen the eight source papers; NAI has seen none.
- **No confidence intervals for the published systems.** The SLDBench paper reports the means of its systems without per-run variation, so the difference 0.748 − 0.706 cannot be tested for significance here.

## 9. Caveats

- **Not pre-registered.** See section 8 above; the pre-registered test is experiment 02.
- **Time-limited searches.** A search's depth depends slightly on machine load, so a reproduction yields comparable statistics, not identical formulas. The choice from the published candidates is exactly reproducible.
- **One machine.** All runs used one Windows PC (12 threads, 16 GB RAM).
- **Approximations.** Most chosen forms are approximations; they should be used for prediction, not read as mechanisms.

## 10. Files and reproduction

See the [README](README.md#5-reproducing). Key result files:

| File | Content |
|---|---|
| `results/v9/nai_sldbench_p123-all-mean.csv` (and p456, p789) | the three replicates: per-task R², held-out score, constants, form |
| `results/v9/verify_p*-all-mean.txt`, `verify_s*.txt` | the official evaluator's scores next to ours |
| `results/v9/chosen_*.json` | the chosen forms, as scored by the official evaluator |
| `results/v9/work/sldbench-s*/` | per search and task: all candidate forms, the chosen form with per-group constants, predictions |
| `results/blind-suite-rules/`, `results/dev-split/` | the runs of section 6 with the v8 and v9 rules and with six searches combined; rejected variants are reported by their scores only |
