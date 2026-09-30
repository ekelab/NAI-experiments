# Experiment 01 - SLDBench: scaling-law discovery

**Question.** Can NAI, a symbolic-regression system without a language model, discover scaling laws that extrapolate as well as LLM-agent systems do?

**Benchmark.** SLDBench (Lin et al., *Can Language Models Discover Scaling Laws?*, ICLR 2026, arXiv:2507.21184). It has eight tasks built from published scaling-law studies. For each task a system must propose one law that is fitted on smaller models or less data and predicts the held-out larger ones.

**Main result.** NAI v9 reaches a mean R² of **0.706 ± 0.026** over three independent replicates. SLDAgent with GPT-5 reaches 0.748 (mean of 5 runs), Goose 0.695, and the human-derived laws 0.517. The detailed report is in [REPORT.md](REPORT.md).

## 1. What the experiment tests

- **Primary quantity:** extrapolation accuracy. R² of the predictions on each task's held-out test rows, averaged over the eight tasks, exactly as SLDBench's official evaluator computes it.
- **Secondary quantities:**
  - the spread of that score over independent replicates of NAI (seeds);
  - the per-task comparison with human-derived laws and with LLM agents, as published in the SLDBench paper.

This experiment was **not pre-registered**. SLDBench was used during NAI's development (see [Caveats](REPORT.md#9-caveats)). The pre-registered test is [experiment 02](../02-open-data-preregistered/).

## 2. Tasks and data

| Task | Source study | Inputs → target | Groups | Test extrapolates in | Constant limit |
|---|---|---|---|---|---|
| parallel | Chen et al. 2025 | N, parallel size → loss | 2 | parallel size | 4 |
| vocab_size | Tao et al. 2024 | non-vocab params, vocab size, characters → unigram-normalised loss | 1 | vocab size | 7 |
| sft | Lin et al. 2024 | fine-tuning data size → loss | 42 | data size | 4 |
| domain_mix | Ye et al. 2024 | 5 domain proportions → 5 domain losses | 4 | (other mixtures) | 7 per target |
| moe | Krajewski et al. 2024 | experts, dense params → loss | 1 | dense params | 6 |
| d_constrain | Muennighoff et al. 2023 | unique tokens, params, tokens → loss | 1 | tokens | 7 |
| lr&bsz | Li et al. 2025 (Step Law) | lr, batch, data, params → loss | 1 | params | none |
| u_shape | Wu & Lo 2024 | log FLOPs → Brier score | 9 | log FLOPs | 6 |

- **Data.** Hugging Face dataset `pkuHaowei/sldbench`, fetched by `reproduce/fetch_data.sh`. The dataset states no license, so it is not redistributed here. A fresh download on 2026-09-30 was identical to the copy used; the row counts are in [section 5](#5-reproducing).
- **Rules** (SLDBench's own):
  - one functional form per task, with constants of its own for each group;
  - at most the task's number of constants;
  - R² computed over the concatenated test rows of all groups;
  - the test rows are never used to fit or choose anything.

## 3. Method under test

**NAI v9** in group mode. [cli/nai_cli.exe](cli/) is its public build: the same search and choice ([cli README](cli/README.md#the-public-build)). One replicate is three searches with seeds s, s+1 and s+2, whose candidate forms are pooled before the choice. Settings per search:

| Setting | Value |
|---|---|
| Time | 180 s per search; 900 s for single-group tasks |
| Threads | 12 (two searches at a time for multi-group tasks) |
| Memory | 768 MB of stored values per stage of a search (3072 MB for single-group tasks), 4096 MB in total |
| Inputs | the task's variables |
| Choice | by accuracy on held-out training rows beyond the fitted range; fewer constants preferred when scores are close |
| Assumption | loss does not rise with model size and data size (declared per task: `MONOTONE` in `harness/sldbench.py`) |

Replicates: seeds {1, 2, 3}, {4, 5, 6} and {7, 8, 9}. The nine single searches are also reported: they are what the previous version, v8, would output.

## 4. Verification

- **Data.** `harness/sld_verify_data.py` re-downloads every file from Hugging Face and compares it with the local copy value for value, including the column schema of the official data loader.
- **The published binary.** Re-running the choice of the three replicates with [cli/nai_cli.exe](cli/) reproduces their forms, constants and predictions byte for byte.
- **Scores.** `harness/sld_verify_r2.py` turns each chosen form into a program in SLDBench's format (`fit_scaling_law`, `scaling_law_func`) and scores it with SLDBench's official `evaluator.py`, unchanged; only the data loader reads the verified local copy. All three replicates of v9 agree with our own computation to four decimals (`results/v9/verify_*.txt`).

## 5. Reproducing

From this folder, with Git Bash or MSYS2, Python 3 and numpy:

```
sh reproduce/fetch_data.sh      # SLDBench data + official evaluator
sh reproduce/run_v9.sh          # nine searches, three pooled replicates, official scores (~10 h)
```

The data fetched should have these row counts (`fetch_data.sh` prints them; `harness/sld_verify_data.py` compares the copy with the source value for value):

| Task | Train rows | Test rows |
|---|---|---|
| parallel_scaling_law | 36 | 12 |
| vocab_scaling_law | 1080 | 120 |
| sft_scaling_law | 504 | 42 |
| domain_mixture_scaling_law | 80 | 24 |
| moe_scaling_law | 193 | 28 |
| data_constrained_scaling_law | 161 | 21 |
| lr_bsz_scaling_law | 2702 | 117 |
| easy_question_scaling_law | 389 | 127 |

Time-limited searches depend slightly on machine load, so a reproduction gives a comparable mean and spread, not identical formulas. The **choice** step is exactly reproducible from the published candidates, and it takes seconds:

```
cd harness                                     # after reproduce/fetch_data.sh
mkdir -p ../results-repro/work
cp -r ../results/v9/work/sldbench-p123 ../results-repro/work/
python3 sld_rules.py --suite sldbench --cli ../cli/nai_cli.exe --out ../results-repro --tags p123 --rules all-mean \
  --common "--engine-groups --jobs 1 --threads 12 --transform both --select nested --monotone --delta 0.003"
# compare ../results-repro/nai_sldbench_p123-all-mean.csv with ../results/v9/nai_sldbench_p123-all-mean.csv
```

To score the published forms with the official evaluator directly:

```
cd harness
python3 sld_verify_r2.py --chosen ../results/v9/chosen_p123-all-mean.json --csv ../results/v9/nai_sldbench_p123-all-mean.csv
```

## 6. Files

| Path | Content |
|---|---|
| `cli/` | The binary, [README](cli/README.md), [PARAMETERS](cli/PARAMETERS.md), examples |
| `harness/` | `sldbench.py` (the benchmark runner), `sldfit.py` (the forms as Python programs), `sld_rules.py` (re-selection), `sld_verify_data.py`, `sld_verify_r2.py`, `blind_scaling.py` with the blind synthetic suite in `blind/scaling/` |
| `results/v9/` | Nine searches (`*_s1…s9`), three pooled replicates (`*_p123/p456/p789-all-mean`), official scores (`verify_*.txt`), chosen forms with per-group constants (`chosen_*.json`), all candidate forms and predictions (`work/`) |
| `results/v8/` | The previous version, v8, on three seeds: `v8` (seed 1), `v8s2`, `v8s3`, and `v8s3moe`, the mixture-of-experts task of seed 3 re-run after it ran out of memory (summaries) |
| `results/blind-suite-rules/` | The runs of the v8 and v9 selection rules on the blind suite, and of combining six searches ([validation protocol](../../docs/VALIDATION-PROTOCOL.md)); the suite's tables are in `harness/blind/scaling/`. The pooled runs (`bpool*`) predate the suite's seventh law, `lrbsz`, which appears in their tables with R² = −1 and is not part of the reported means over six laws |
| `results/dev-split/` | The internal split of SLDBench's training rows (predictions and candidates; the training rows themselves are not redistributed) |
| `reproduce/` | `fetch_data.sh`, `run_v9.sh` |
