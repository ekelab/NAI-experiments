# `nai_cli` - command-line interface of NAI (v9, public build)

`nai_cli.exe` searches a formula for a table of data: variables → value. The experiments in this repository were run with NAI v9, and this file is the public build of that version. All of its parameters are listed in [PARAMETERS.md](PARAMETERS.md).

> **License.** This binary is free for non-commercial use: to reproduce the experiments in this repository, and to evaluate NAI on your own data for research or teaching. Commercial use of any kind requires the written permission of the author, Mikhail Smirnov (lab@mindscan.org). See [LICENSE.md](../../../LICENSE.md).

| | |
|---|---|
| File | `nai_cli.exe` |
| Version | NAI v9 (frozen 2026-09-29), public build (2026-09-30) |
| Platform | Windows 10/11, x86-64. Statically linked: it needs only the system C runtime (UCRT), no installation. |
| Hardware | Any x86-64 CPU; all threads are used by default. RAM: 1–5 GB for the settings used here. |
| Source code | Not published. |

## The public build

The build used in the runs and this public build have the same search and the same choice of formula. The public build differs only in what it prints and accepts:

- its output states the result, not the course of the search;
- its output is in English;
- developer options (switching off parts of the search, diagnostics) are not included.

It was checked against the build used in the runs:

- **The choice of formula.** Re-running the choice for every published replicate, from its saved candidates, reproduces the published forms, per-group constants and predictions byte for byte: 3 replicates × 12 targets in experiment 01, 3 × 4 in experiment 02.
- **The search.** Complete searches without a time limit give identical results with both builds (the same Pareto front and answer; in group mode the same candidates, constants and predictions): 23 of the 29 built-in problems, and a table in group mode.

## Input table format

A text table: a header line, then one row per observation.

```
# comment lines start with '#'
n,P
1,0.16666666666666666
2,0.3055555555555556
...
```

- **Separators.** Comma, semicolon, tab or spaces, whichever the header uses. With semicolons, tabs or spaces a decimal comma is accepted too.
- **Columns.** The last column is the value to explain; the others are variables.
- **Special columns** (any case):
  - `sigma` (or `sd`, `se`, `err`, `error`): the standard error of the value.
  - `trials`: the number of trials behind a frequency.
  - `group`: a label. Rows with the same label form a group, and the table is then searched in group mode: one form, constants per group.
- **Precision.** Without a `sigma` column, the precision is read from how the numbers are printed. A value printed with 4 decimals is known to half a unit of the fourth decimal. Integers, and numbers with 15 or more significant digits, are exact. Print exact data in full (for example `%.17g` or Python's `repr`) so that NAI treats them as exact.
- **Size.** At least 8 rows: they are split into rows the formula is fitted on, rows its choice is validated on, and internal test rows.

## Basic use

The general form of a call is:

```
nai_cli.exe --data table.csv [--seconds 300] [--threads 0]
```

The program describes the table, searches, and then prints the Pareto front of the forms it found (size in nodes, description length (MDL), χ² per degree of freedom, and maximum relative error on the internal test rows), the time, and the result. The last line is the result:

| Result line | Meaning |
|---|---|
| `SOLVED: <formula>` | An exact law: it reproduces the data within their precision and is the shortest such description found. |
| `WITHIN THE NOISE (not the law): <formula>` | It describes noisy data within the noise, but is not claimed to be the law. |
| `APPROXIMATION (not the law): <formula>` | No law was found. This is the formula that extrapolates best on held-out rows. It is used for prediction and should not be interpreted as a law. |
| `RECURRENCE (no closed form): <recurrence>` | An exact recurrence was found, but no closed form. |
| `unsolved` | Nothing adequate within the budget. |

### Example 1 - an exact law

[`examples/dice.csv`](examples/dice.csv) holds the probability of at least one six in n rolls of a fair die, printed exactly:

```
> nai_cli.exe --data examples/dice.csv --seconds 60
dice: n (integer) → P (probability); exact values

Pareto front (nodes, MDL, χ²/dof, test max rel., formula):
  ...
    7         14.67    2.62e-12   0.00e+00  1 - (5/6)^n

recurrence: 6·P(n+1) - 5·P(n) - 1 = 0

done in 0.50 s

SOLVED: 1 - (5/6)^n
```

### Example 2 - group mode, prediction of held-out points

[`examples/finetune_groups.csv`](examples/finetune_groups.csv) holds synthetic fine-tuning losses of six models (the `group` column). One functional form is sought for all six, each with its own constants. The losses at larger data sizes are then predicted for the points in [`examples/finetune_points.csv`](examples/finetune_points.csv):

```
> nai_cli.exe --data examples/finetune_groups.csv --probability no ^
      --seconds 60 --threads 12 --jobs 2 --max-constants 4 --extrap sft_data_size ^
      --monotone sft_data_size --holdout-vars all ^
      --predict examples/finetune_points.csv pred.txt --forms forms.txt
finetune_groups: sft_data_size (integer) → sft_loss; values with errors
groups: 6
  group model0: 21.7 s
  group model0 (transformed inputs): 20.6 s
  ...
one form for every group; constants: 3, held-out R² 0.9962, forms judged 175
  model0: 0.607995 + 0.297135*0.978971^(log(sft_data_size)*(log(sft_data_size) - 3))
  ...
ONE FORM FOR THE GROUPS: 0.607995 + 0.297135*0.978971^(log(sft_data_size)*(log(sft_data_size) - 3))
```

`pred.txt` receives one prediction per row of the points file. `forms.txt` receives the form in prefix notation and the constants of every group.

## Settings used in the experiments

The experiments call `nai_cli` through a harness, `sldbench.py` (in `harness/` in experiment 01, in `protocol/bench/` in experiment 02). For a table with groups, the harness issues:

```
nai_cli.exe --data <train table> --lang en --probability no --no-learn --threads 12 --seconds <180 or 900> \
    --predict <test points> <predictions> --forms <forms> --max-constants <task limit> \
    --delta 0.003 --seed <s> --jobs <2, or 1 for single-group tasks> --extrap <variable> \
    [--monotone <variables>] [--max-memory 3072] --max-total-memory 4096 \
    --holdout-vars all --candidates-out <candidates>
```

(`--lang en` and `--no-learn` were needed by the build used in the runs; this build accepts them and needs neither.) A **replicate of the method "NAI v9"** consists of three such searches, with seeds s, s+1 and s+2. The final form is chosen from their pooled candidates with `--candidates-in` (see the experiment READMEs).

## Practical notes

- **Time-limited searches are not bit-for-bit repeatable.** With `--seconds`, how far a search gets depends slightly on machine load. The same seed gives the same split of the rows, but not necessarily the same depth of search. The experiments therefore report means over independent replicates.
- **Memory.** Raising `--max-memory` without `--max-total-memory` can exhaust the memory of the machine, because the values stored by the earlier stages of a search are kept until it ends.
