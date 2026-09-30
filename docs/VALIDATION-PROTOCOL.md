# Validation protocol

Improving a method against a benchmark's test set, even only by deciding which change to keep after seeing test results, turns the test set into a training set, and the reported score then overstates the method. This note describes the rules NAI's development followed from September 2026 onward to avoid that, and where they fall short.

## Rules

1. **The test rows of a benchmark enter no choice.** They are not used to choose the formula, the constants, the settings of NAI or which change to NAI to keep. They are used once, to score a frozen version.
2. **A change must first win on data that are not the benchmark.** A candidate change (a new rule for choosing a form, a new search strategy) is accepted only if it:
   - is not worse on a **blind synthetic suite**, and
   - is better on an **internal split of the benchmark's training rows**, when the synthetic suite is too easy to discriminate.
3. **Selection rules are compared on the same candidates.** Every search saves all of its candidate forms (`--candidates-out`). A rule of choice can therefore be re-applied to the same candidates in seconds (`--candidates-in`), without a new, time-dependent search. Differences between rules are then differences of the rules alone.
4. **Frozen versions, replicated.** A version is frozen before it is scored. It is scored on independent replicates (different seeds), and every replicate is reported. The best replicate is never presented as the result.
5. **Scores are recomputed by the benchmark's own code.** For SLDBench, NAI's chosen forms are turned into programs in the benchmark's format and scored by its official `evaluator.py`, unchanged. The data are compared with the published source value for value.
6. **Pre-registration for new tests.** For a test on new data, the hypotheses, data, held-out rows, baselines, metrics and code are fixed before any run (experiment 02).

## The blind synthetic suite

It contains scaling laws from the literature with their published constants and realistic noise: Chinchilla, Kaplan, data-constrained, learning-rate optimum, mixture of experts, fine-tuning, and, added later, a four-input learning-rate × batch law. Test splits extrapolate to larger models or more data, as in SLDBench.

The suite was generated before the rounds of work it judged, and its six original tasks were not changed afterwards. Generator: `experiments/01-sldbench/harness/blind_scaling.py`.

**Limitation.** The synthetic laws are smooth and the noise is small. The suite reliably rejects harmful rules, but it is too easy to show improvements that matter on real data: searches often end early, having reached the noise level.

## The internal split of training rows (`--dev 0.3`)

In each group of a benchmark task, the training rows are split by the variable along which the benchmark's test extrapolates. The top values, about 30 % of the rows (whole values, so that no model size appears on both sides), play the role of the test; the rest play the role of the training data. The benchmark's test file is not opened in this mode.

This gives a development signal as hard as the benchmark's own data, without contact with its test.

## How the rules were applied

| Candidate change (by kind) | Blind suite | Internal split | Decision |
|---|---|---|---|
| A broader held-out criterion for the choice of form | 0.9687 → 0.9721 | - | accepted (v9) |
| Combining several independent searches before the choice | 0.9721 → 0.9782 (spread 0.015 → 0.005) | 0.724 → 0.845 (checked after v9 was frozen) | accepted (v9), on the blind suite |
| A stricter, worst-case score | 0.9613 | - | rejected |
| A stronger extrapolation constraint (two variants) | 0.9654 / 0.9603 | - | rejected |
| A restricted alphabet for the chosen form | 0.9538 | - | rejected |
| Larger formulas from a growth strategy (two variants) | 0.9640; 0.968 vs 0.983 on the 4-input task | - | rejected |
| More free constants per form | - | 0.810 → 0.760 | rejected |
| Combining more searches | 0.9801 vs 0.9782 | 0.883 vs 0.845 | not adopted: one replicate, twice the cost |

The numbers are mean R² over seeds (single searches) or over pooled replicates. Raw outputs are published for the runs of the adopted rules (v8 and v9) and of combining six searches; the rejected variants are reported by these scores only. The internal split was introduced after version v9 had been frozen and scored on SLDBench. The changes tried after that point were rejected, so SLDBench's test was not consulted again. Details are in [experiment 01's report](../experiments/01-sldbench/REPORT.md#6-development-and-validation).

## Where the protocol falls short

- **SLDBench was used during development before these rules existed, and its results set the direction of later work** (which tasks were weak). This is a mild form of benchmark overfitting at the level of the developer, which the rules above reduce but do not undo. Experiment 02 is the cleaner evidence: pre-registered, on data NAI had never seen.
- **The internal split comes from the same distribution as the benchmark.** It protects the test rows, not against over-adaptation to the benchmark's domain.
