# Experiments

| # | Folder | Question | Pre-registered | NAI version | Main result |
|---|---|---|---|---|---|
| 01 | [01-sldbench](01-sldbench/) | Does NAI, without an LLM, reach LLM-agent systems on SLDBench's eight scaling-law tasks? | no | v9 | mean R² 0.706 ± 0.026 (SLDAgent/GPT-5 0.748, Goose 0.695, humans 0.517) |
| 02 | [02-open-data-preregistered](02-open-data-preregistered/) | On open data never seen, is NAI better than the standard ansatz fit and than general symbolic regression? | yes (protocol fixed before the run) | v9 | mean R² 0.733 vs −0.412 for the PySR engine; H2, H3 supported; H1 not |

Every experiment folder has the same structure:

| Item | Content |
|---|---|
| `README.md` | Design: question, data, methods, protocol, how to reproduce |
| `REPORT.md` | Results, analysis, comparison with other methods, caveats |
| `cli/` | The NAI binary (public build of the version used), a manual and a parameter reference |
| `results/` | The outputs of the runs: per-search and per-replicate scores, chosen formulas with constants, all candidate formulas, predictions, run summaries |
| `reproduce/` | Scripts that fetch the data and repeat the protocol |
