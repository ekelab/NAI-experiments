# `nai_cli` - parameter reference (NAI v9, public build)

This reference covers the binary in this folder, the public build of NAI v9 ([README](README.md#the-public-build)). It lists the 30 command-line parameters that this build recognises; any other parameter is an error (exit code 2).

Conventions: `N`, `K` are integers, `S`, `D` are real numbers, `MB` is mebibytes, `FILE` is a path. Unless stated otherwise, a parameter is off by default.

## 1. Input and mode

| Parameter | Argument | Default | Meaning |
|---|---|---|---|
| `--data` | `FILE` | - | Search a law for the table in `FILE` (format: [README](README.md#input-table-format)). A table with a `group` column runs in **group mode** (one form, constants per group; section 5). |
| `--task` | `ID` | - | Search one of the built-in demonstration problems instead of a table. |
| `--list` | - | - | Print the built-in problems and exit. |
| `--lang` | `en` | `en` | Accepted for compatibility with the harness; this build writes English only. |

## 2. Data handling

| Parameter | Argument | Default | Meaning |
|---|---|---|---|
| `--extrap` | `VAR` | first variable | The variable along which the law must extrapolate; used for validation. |
| `--probability` | `yes` \| `no` | guessed | Whether the value is a probability, which affects the noise model and the admissible forms. |
| `--seed` | `N` | `1` | Seed of the random split of the rows into fitted, validation and internal-test parts, and of the other random choices of the search. Different seeds give different, equally valid searches. |

## 3. Budget and search space

| Parameter | Argument | Default | Meaning |
|---|---|---|---|
| `--seconds` | `S` | `0` (no limit) | Wall-clock budget. When it runs out, the best answer found so far is returned. |
| `--threads` | `N` | `0` (all hardware threads) | Worker threads of the search. |
| `--max-size` | `N` | `11` | Formula size, in nodes, up to which the search is systematic. Larger formulas can still be found. |
| `--max-memory` | `MB` | `768` | Memory limit for one stage of the search. When it is reached, the stage stops growing. |
| `--max-total-memory` | `MB` | `0` (no limit) | The same limit for all stages of one search together. Recommended whenever `--max-memory` is raised. |
| `--literals` | `a,b,…` | `1,2,3` | Integer literals available to the search. Other constants are fitted, and replaced by exact values where the data allow it. |
| `--exclude` | `op,op,…` | - | Remove operations from the alphabet. Names: `+ - * / ^ neg exp log sqrt fact binom ffact sum prod`. |
| `--no-stop` | - | stop | Keep searching after an exact law is found, until the budget ends. |

The alphabet of operations:

| Operations | Used on |
|---|---|
| `+ − × ÷ neg ^ exp log sqrt` | all data |
| `fact binom ffact`, `Σ Π` (sums and products over an index) | counted data: small integer variables |
| `sin cos` | continuous data |

## 4. The answer

| Parameter | Argument | Default | Meaning |
|---|---|---|---|
| `--no-approx` | - | on | Disable approximation mode. Without it, when no exact law exists, the model of least description length (MDL) is returned instead of the best-extrapolating one. |
| `--all-ops` | - | off | Use every operation on any data, instead of choosing the alphabet by the type of the data (section 3). |
| `--no-learn` | - | - | Accepted for compatibility with the harness; no effect in this build. |

## 5. Group mode (a table with a `group` column)

A single functional form is chosen for all groups; each group gets its own values of the constants. Candidate forms are judged by their accuracy on held-out rows.

| Parameter | Argument | Default | Meaning |
|---|---|---|---|
| `--max-constants` | `K` | `0` (no limit) | Largest number of constants a form may have. |
| `--delta` | `D` | `0.003` | Tie margin of the held-out score, within which a form with fewer constants is preferred. |
| `--monotone` | `v1,v2,…` | - | Declared assumption: the prediction must not rise with these variables beyond the data. Forms that violate it are rejected. |
| `--no-log-inputs` | - | on | Disable the second search on transformed inputs. |
| `--holdout-vars` | `v1,v2,…` \| `all` | extrapolation variable | Variables along which held-out validation rows are taken (`all`: every input). |
| `--holdout-score` | `mean` \| `min` | `mean` | How the held-out scores are combined. |
| `--restarts` | `K` | `1` | Run `K` searches with seeds `seed … seed+K−1` and pool their candidates before the choice. |
| `--jobs` | `N` | `1` | Searches run concurrently, each on `threads/N` threads. |
| `--candidates-out` | `FILE` | - | Write every candidate form, in prefix notation, for later re-selection. |
| `--candidates-in` | `FILE` | - | Skip the search and judge the forms in `FILE` (re-selection). |

## 6. Output

| Parameter | Argument | Meaning |
|---|---|---|
| `--predict` | `POINTS OUT` | Evaluate the final formula at the rows of the table `POINTS` (same column names; in group mode a `group` column selects the constants) and write one prediction per line to `OUT` (`%.17g`). |
| `--forms` | `OUT` | Single table: the Pareto front, one form per line (`nodes`, `MDL`, form in prefix notation), and a recurrence line if one was found. Group mode: `form` followed by the chosen form with the constants every group's fit starts from, then one `group <name> <c0,c1,…>` line per group. |

## 7. Exit codes

| Code | Meaning |
|---|---|
| `0` | An exact law was found (single table), or a form was chosen (group mode). |
| `1` | No exact law: approximation, recurrence or unsolved (single table), or no form fits every group (group mode). This is a normal outcome. |
| `2` | Usage or input error. |
