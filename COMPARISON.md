# NAI compared with other methods

This note collects every comparison made in the experiments of this repository, states what each comparison can and cannot show, and does not combine numbers that are not comparable.

## 1. The methods

| Method | Kind | Needs an LLM | Domain knowledge used | Output | How the final formula is chosen | Where compared |
|---|---|---|---|---|---|---|
| **NAI v9** | symbolic regression (search over formulas, description-length scoring) | no | operations; declared monotonicity in model and data size | formula, constants per group | accuracy on held-out data beyond the fitted range; three searches pooled | 01, 02 |
| SLDAgent (GPT-5, Claude Sonnet 4.5, Gemini 2.5 Flash) | LLM-driven evolution of programs | yes | whatever the LLM has learned, including possibly the source papers | program: formula + fitting code | best score on the training data | 01 (published numbers) |
| Goose, CodeX | general coding agents | yes | as above | program | agent's own | 01 (published numbers) |
| Human-derived laws | the forms published in the source studies | - | the authors' expertise | formula | by the authors | 01 (published numbers) |
| B1: Chinchilla fit | fixed ansatz E + A/N^α + B/D^β | no | the ansatz | formula | - (form fixed) | 02 |
| B2: power law | fixed ansatz, no offset | no | the ansatz | formula | - | 02 |
| B3: literature lr/batch form | fixed ansatz with moving optima (DeepSeek LLM, Step Law) | no | the ansatz, including the optimum's dependence | formula | - | 02 (E3) |
| B4: SymbolicRegression.jl 2.5.0 (engine of PySR) | symbolic regression (evolution) | no | operations only | formula, per-group parameters | SymbolicRegression.jl's default: best score by training loss and complexity | 02 |

## 2. Against LLM-agent systems (experiment 01, SLDBench)

Mean extrapolation R² over the eight SLDBench tasks. Agents: mean of five runs (SLDBench paper). NAI: mean of three replicates.

| System | Mean R² |
|---|---|
| SLDAgent (GPT-5) | 0.748 |
| **NAI v9** | **0.706 ± 0.026** |
| Goose | 0.695 |
| SLDAgent (Claude Sonnet 4.5) | 0.590 |
| CodeX | 0.550 |
| Human-derived laws | 0.517 |
| SLDAgent (Gemini 2.5 Flash) | 0.506 |

Per task, against the best system and the human laws:

| Task | NAI v9 | SLDAgent (GPT-5) | Human law | NAI vs best |
|---|---|---|---|---|
| parallel | 0.985 | 1.000 | 1.000 | −0.015 |
| vocab_size | 0.971 | 0.987 | 0.966 | −0.016 |
| sft | 0.982 | 0.993 | 0.957 | −0.011 |
| domain_mix | **0.991** | 0.988 | 0.671 | **+0.003** |
| moe | 0.656 | 0.773 | 0.703 | −0.117 |
| d_constrain | 0.768 | 0.944 | 0.911 | −0.176 |
| lr&bsz | 0.372 | 0.604 | −0.076 | −0.232 |
| u_shape | **−0.076** | −0.305 | −1.000 | **+0.229** |

**Reading.** Without a language model, NAI is second among the published systems. It is within about 0.016 of the best system on four tasks and ahead of it on two, domain mixture and the U-shaped task. The whole gap to SLDAgent lies in three tasks whose literature laws are large, and it is concentrated there. The reasons are analysed in the [report](experiments/01-sldbench/REPORT.md#7-where-nai-is-stronger-where-weaker---and-why).

## 3. Against standard practice and general symbolic regression (experiment 02, pre-registered)

Mean extrapolation R² on four open datasets never seen by NAI. NAI and B4: mean of three replicates. B1–B3 are deterministic.

| Task | NAI v9 | B1 Chinchilla | B2 power law | B3 literature form | B4 PySR engine |
|---|---|---|---|---|---|
| E1 Chinchilla (Epoch) | **0.892** | 0.813 | 0.524 | - | −2.460 |
| E2 Porian, Chinchilla-type | 0.775 | **0.930** | 0.644 | - | 0.649 |
| E3 Porian, lr × batch | 0.500 | −0.000 | −0.375 | **0.504** | −0.368 |
| E4 DataDecide | **0.765** | 0.730 | 0.732 | - | 0.530 |
| Mean | **0.733** | - | - | - | −0.412 |

**Reading:**
- **Against a fixed ansatz.** NAI is better on two tasks of three where the ansatz applies (E1, E4) and worse on one (E2). Where the ansatz lacks a dependence (E3), NAI matches the literature form that encodes it.
- **Against general symbolic regression at equal time.** NAI is better on every task, on average. The difference is mostly one of stability: B4's R² on E1 ranges from −9.06 to 0.875 across replicates. NAI tests extrapolation in its choice of formula; B4's default choice does not.
- **Not on every metric.** On E4 the plain power law has a lower mean relative error than NAI (4.9 % vs 7.7 %). On E2 the Chinchilla fit is more accurate on the largest models. See the [report](experiments/02-open-data-preregistered/REPORT.md#53-secondary-metrics).

## 4. What these comparisons cannot show

- **Numbers from different experiments are not comparable.** R² depends on how far the test extrapolates and on how much the target varies. 0.706 on SLDBench and 0.733 in experiment 02 measure different things.
- **SLDAgent was not run on experiment 02's data.** No statement about LLM agents on those data is made.
- **Developer-level overfitting on SLDBench.** NAI was developed while observing SLDBench's test results in early versions. Experiment 02, which is pre-registered, is the cleaner evidence.
- **Different notions of a "run".** A replicate of NAI v9 pools three searches, at three times the cost of one. The single-search figure on SLDBench is 0.677.
- **B4 was run with default settings.** A tuned configuration, or one with an extrapolation-aware choice of equation, may do better. The comparison is at equal wall-clock time, which is not equal effective compute: the two programs parallelise differently.
- **Unequal prior knowledge.** LLM agents may have read the papers behind the benchmark; B3 encodes the literature's form. NAI uses neither. This is part of what is being compared, not a flaw of the comparison, but it should be kept in mind when reading "NAI below SLDAgent" or "NAI equal to B3".
- **No significance tests against published systems.** The SLDBench paper reports means without per-run variation, so the gap to SLDAgent (0.042) cannot be tested for significance here.
