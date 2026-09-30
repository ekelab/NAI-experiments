# How NAI works - an overview

This note describes NAI at the level needed to read the experiments: what it searches, how a result should be interpreted, and what it does not use. It is not a description of the implementation.

## Principles

1. **Search over formulas, not over the constants of one formula.** NAI searches expressions built from an alphabet of operations over the input variables, starting with the simplest. In the scaling-law experiments the alphabet is `+ − × ÷ ^ neg exp log sqrt sin cos`. Other alphabets, for example combinatorial operations for counted data, are chosen according to the data.
2. **Description length decides.** A candidate is scored by how briefly it describes the data: its accuracy under a model of the noise, weighed against its complexity. Among formulas that fit equally well, the shorter one wins.
3. **Growth when the search stalls.** When larger formulas stop improving the fit, NAI switches to strategies that build larger structures from what it has already found.
4. **Laws and approximations are kept apart.** A **law** is a formula that reproduces the data within their precision. When no law is found, NAI returns the formula that extrapolates best on held-out data and labels it **APPROXIMATION (not the law)**. All results on real scaling-law data in this repository are approximations.

## Group mode: one form, constants per group

Scaling-law benchmarks require one functional form per task, with its own constants for each group (for example one form for six base models, each with its own constants). In this mode NAI:

- **Searches the groups,** and collects candidate forms from all of them.
- **Chooses one form** by its accuracy on held-out training rows that lie beyond the range the constants were fitted on. Where the scores are close it prefers the form with fewer constants, and it can apply a declared assumption: that the loss does not rise with model or data size.
- **Refits the constants** of the chosen form separately for each group, on all of its training rows.

In version v9, one result (a *replicate*) combines the candidates of three independent searches before the choice. Its cost is therefore three searches.

## What NAI does not use

- **No language model and no pre-trained knowledge of scaling-law papers.** NAI knows only its operations and the data it is given.
- **No test data.** The form, the constants and the choice use only the training rows.
- **No hand-written form of the law.** The only priors are the alphabet of operations, the preference for short descriptions, the noise model and, in group mode, the declared monotonicity in model and data size.

## Repeatability

Searches run within a time budget, so how far a search gets depends slightly on machine load. Two runs with the same seed are therefore not guaranteed to produce the same formula, and the experiments report means over independent replicates. Given the same candidate forms, the choice of form is deterministic. The published binaries can re-run it from saved candidates, so this step reproduces exactly.
