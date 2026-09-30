# NAI - No-Ansatz Inference: experiments and reproduction materials

This repository publishes the experiments run with **NAI**, a system that infers the functional form of a law from tabular data. It contains the protocols, the data (or scripts that fetch them), the exact program binaries used, the raw results and detailed reports, so that every reported number can be checked and reproduced independently. **The source code of NAI is not part of this repository.**

## What NAI is

In empirical science and in machine-learning engineering, a law is usually fitted to data by choosing its functional form first, the *ansatz* (for instance the Chinchilla scaling law L = E + A/N^α + B/D^β), and then estimating its constants. The quality of every prediction then rests on that choice, which is made by a person, in advance.

NAI does not require the form in advance. Given a table (variables → value), it searches the space of formulas built from a set of operations and returns a formula together with measures of its accuracy and complexity. NAI still makes assumptions: its alphabet of operations, a preference for short descriptions (minimum description length, MDL) and a model of the noise. What it does not assume is the shape of the law. Hence the name: *No-Ansatz Inference*.

Two kinds of answer are distinguished explicitly:

- **Law.** A formula that reproduces the data within their precision and is the shortest such description found (e.g. `1 - (5/6)^n` for the probability of at least one six in n rolls).
- **Approximation.** When no law is found, NAI returns the formula that extrapolates best on held-out data and labels it as an approximation (not the law). All results on real scaling-law data in this repository are approximations in this sense.

A brief description of the method is in [docs/METHOD.md](docs/METHOD.md).

## Intended uses

- **Scaling laws of neural networks.** Predicting the loss of a large training run from small ones, and checking whether a textbook form of the law is adequate for a new setting.
- **Research mathematics, probability and combinatorics.** Exact closed forms and recurrences for sequences and probabilities, which can then be proven.
- **Empirical science in general.** Interpretable formulas for tabular data where a model must be checked, not only used.

This repository currently covers the first use.

## Strengths - and the evidence for them

| Strength | Evidence in this repository |
|---|---|
| Finds structure that a standard ansatz lacks | On a learning-rate × batch-size sweep, where the Chinchilla fit scores R² ≈ 0, NAI matches the hand-made literature form (0.500 vs 0.504) without being given it ([experiment 02](experiments/02-open-data-preregistered/REPORT.md)). On SLDBench's domain-mixture task it is above the best LLM agent (0.991 vs 0.988) ([experiment 01](experiments/01-sldbench/REPORT.md)). |
| Extrapolation-aware choice of the formula | At equal time, NAI's mean R² is 0.733 against −0.412 for the engine of PySR, whose default choice does not test extrapolation ([experiment 02](experiments/02-open-data-preregistered/REPORT.md)). |
| Competitive with LLM-agent systems, without an LLM | On SLDBench: 0.706 ± 0.026, above Goose (0.695) and human-derived laws (0.517), below SLDAgent with GPT-5 (0.748) ([experiment 01](experiments/01-sldbench/REPORT.md)). |
| Lightweight and offline | One PC (12 threads), no GPU, no language model, no network access during the search. The binary is self-contained. |
| Verifiable | The published binary reproduces every published choice of formula exactly; a protocol fixed before the run (experiment 02); results re-scored with the benchmark's own evaluator. |

## Limitations

These are stated as prominently as the strengths, because they bound what the results mean.

- **The choice of form is unstable on some tasks.** Independent searches can choose different forms. On several tasks one of three replicates chose a form that fits the data range but fails beyond it. Pooling three searches reduces this, but does not remove it.
- **Approximations are not readable laws.** On noisy real data, the chosen formulas often contain terms such as sin(ln D) acting as flexible basis functions. They extrapolate well on most tasks, but they are not laws a person can interpret. A textbook form is readable by construction.
- **Below the best LLM agent on SLDBench.** The gap is 0.042 in mean R², concentrated in the data-constrained, mixture-of-experts and learning-rate/batch tasks.
- **Closed source, Windows only.** The published binary runs on Windows 10/11 x86-64.

## Experiments

| # | Experiment | Question | Main result |
|---|---|---|---|
| [01](experiments/01-sldbench/) | **SLDBench** (Lin et al., ICLR 2026): 8 scaling-law discovery tasks | Does NAI, without an LLM, reach the level of LLM-agent systems in discovering laws that extrapolate? | 0.706 ± 0.026 over 3 replicates; SLDAgent/GPT-5 0.748, Goose 0.695, humans 0.517 |
| [02](experiments/02-open-data-preregistered/) | **Pre-registered test on open data** never seen by NAI (Chinchilla reconstruction, Porian et al. 2024, DataDecide) | Does NAI predict large models from small ones better than the standard practice and than general symbolic regression? | Mean R² 0.733 vs PySR engine −0.412. H2 and H3 confirmed; H1 not. |

A cross-experiment comparison with other methods is in [COMPARISON.md](COMPARISON.md). How changes to NAI were validated without touching test data is described in [docs/VALIDATION-PROTOCOL.md](docs/VALIDATION-PROTOCOL.md).

## Repository layout

```
README.md                  this file
COMPARISON.md              NAI against other methods, across experiments, with caveats
LICENSE.md, NOTICE.md      licenses of this repository and of third-party data and software
CITATION.cff               how to cite
docs/
  METHOD.md                how NAI works, at the level needed to interpret the results
  VALIDATION-PROTOCOL.md   blind suites, internal splits and pre-registration
experiments/
  01-sldbench/             README (design), REPORT (results), cli/ (binary + manual),
                           harness/, results/, reproduce/
  02-open-data-preregistered/
                           README, REPORT, PROTOCOL (fixed before the run), cli/,
                           protocol/ (scripts, task data), results/, reproduce/
```

Each experiment folder contains the NAI binary (`cli/nai_cli.exe`, the public build of the version used; see [below](#the-published-binary)), a manual (`cli/README.md`) and a parameter reference (`cli/PARAMETERS.md`).

## Requirements for reproduction

- Windows 10/11 x86-64, for `nai_cli.exe`.
- A POSIX shell with `curl` and `git`: Git Bash or MSYS2.
- Python 3.10+ with numpy; for experiment 02 also pandas, scipy and pyarrow. The runs used MSYS2 ucrt64 Python 3.14 with numpy 2.5.2, pandas 3.0.5, scipy 1.18.1 and pyarrow 25.0.1.
- For experiment 02, baseline B4: Julia 1.13 (packages pinned by `Manifest.toml`).
- Time: about 10–11 hours per experiment on a 12-thread PC.

## The published binary

The experiments were run with NAI v9. The binary in the `cli/` folders is the public build of that version: the same search and the same choice of formula, with shorter output (the result rather than the course of the search), in English, and without developer options. It was checked against the build used in the runs:

- re-running the choice of formula for every published replicate reproduces the published forms, per-group constants and predictions byte for byte (experiment 01: 3 replicates × 12 targets; experiment 02: 3 × 4);
- complete searches without a time limit give identical results with both builds (23 of the 29 built-in problems, and a table in group mode).

Scores are re-computed with the benchmark's own evaluator where one exists (SLDBench), and the data are compared with their sources value for value.

## Language

All documents and files in this repository are in English.

## Author, citation and license

NAI and these experiments are by **Mikhail Smirnov** (lab@mindscan.org). To cite this work, see [CITATION.cff](CITATION.cff); for licensing, see [LICENSE.md](LICENSE.md) and [NOTICE.md](NOTICE.md).

**The NAI binaries in this repository are free for non-commercial use:** reproducing the published experiments, and evaluating NAI on your own data for research or teaching. **Commercial use of any kind requires the author's written permission:** contact lab@mindscan.org. Third-party datasets keep their own licenses; data without a stated license are not redistributed but fetched from their source by the scripts in `reproduce/`.
