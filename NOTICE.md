# Third-party data and software

This repository uses the following third-party material. Each keeps its own license.

## Data

| Material | Source | License | In this repository |
|---|---|---|---|
| SLDBench data | Hugging Face `pkuHaowei/sldbench`; Lin et al., *Can Language Models Discover Scaling Laws?*, ICLR 2026 | not stated | **Not redistributed.** Fetched by `experiments/01-sldbench/reproduce/fetch_data.sh`. Results derived from it (formulas, predictions, scores) are included. |
| Chinchilla reconstruction | Epoch AI, `github.com/epoch-research/analyzing-chinchilla` (Besiroglu et al. 2024, *Chinchilla Scaling: A replication attempt*); extracted from Hoffmann et al. 2022 | not stated | The raw file is not redistributed (fetched by `experiments/02-open-data-preregistered/reproduce/fetch_sources.sh`). The derived task files (model size, tokens, loss) are included because they are registered and needed to verify the pre-registration. |
| Porian et al. 2024 experiment results | `github.com/formll/resolving-scaling-law-discrepancies` (*Resolving Discrepancies in Compute-Optimal Scaling of Language Models*, NeurIPS 2024) | MIT | Raw file fetched; derived task files included, with attribution. |
| DataDecide evaluation results | Hugging Face `allenai/DataDecide-eval-results` (Magnusson et al. 2025, AI2) | ODC-BY 1.0 | Raw files fetched; derived task files included, with attribution. |
| ObsScaling evaluation results | `github.com/ryoungj/ObsScaling` (Ruan et al. 2024) | Apache-2.0 | Fetched only (listed among the sources of experiment 02); not used by any task. |

## Software

| Software | Source | License | Use |
|---|---|---|---|
| SLDBench official evaluator (`evaluator.py`, `data_loader.py`) | `github.com/linhaowei1/SLD` | MIT (per the repository) | **Not redistributed.** Fetched by `fetch_data.sh` to re-score NAI's formulas. |
| SymbolicRegression.jl 2.5.0 and dependencies | `github.com/MilesCranmer/SymbolicRegression.jl` | Apache-2.0 | Baseline B4 in experiment 02. Installed by Julia from `Manifest.toml`; not redistributed. |
| Julia 1.13 | julialang.org | MIT | Runs baseline B4. |
| Python, numpy, pandas, scipy, pyarrow | - | BSD/Apache-style | Harness, baselines, scoring. |

## Citations of the data sources

- Lin, H. et al. *Can Language Models Discover Scaling Laws?* ICLR 2026. arXiv:2507.21184.
- Hoffmann, J. et al. *Training Compute-Optimal Large Language Models.* NeurIPS 2022.
- Besiroglu, T., Erdil, E., Barnett, M., You, J. *Chinchilla Scaling: A replication attempt.* arXiv:2404.10102, 2024.
- Porian, T., Wortsman, M., Jitsev, J., Schmidt, L., Carmon, Y. *Resolving Discrepancies in Compute-Optimal Scaling of Language Models.* NeurIPS 2024.
- Magnusson, I. et al. *DataDecide: How to Predict Best Pretraining Data with Small Experiments.* arXiv:2504.11393, 2025.
- Cranmer, M. *Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl.* arXiv:2305.01582, 2023.
