#!/bin/sh
# Download the raw sources of experiment 02 into protocol/bench/exp1/data/.
#
# Only needed to regenerate the task files: the task files themselves
# (protocol/bench/exp1/tasks/*.json) are included. After
#     python3 protocol/bench/exp1/prepare.py
# `git status` shows whether the regenerated files differ from the included ones.
# Run from the experiment folder:  sh reproduce/fetch_sources.sh
set -e
cd "$(dirname "$0")/../protocol/bench/exp1"
mkdir -p data/datadecide
get() { echo "  $2"; curl -sSfL -o "$2" "$1"; }

echo "== Epoch AI, Chinchilla reconstruction (github.com/epoch-research/analyzing-chinchilla; no license stated)"
get https://raw.githubusercontent.com/epoch-research/analyzing-chinchilla/main/data/svg_extracted_data.csv \
    data/epoch_chinchilla_svg_extracted_data.csv
echo "== Porian et al. 2024 (github.com/formll/resolving-scaling-law-discrepancies; MIT)"
get https://github.com/formll/resolving-scaling-law-discrepancies/raw/main/data/experiment_results.pickle.xz \
    data/porian_experiment_results.pickle.xz
echo "== Ruan et al. 2024, ObsScaling (github.com/ryoungj/ObsScaling; Apache-2.0) - downloaded, not used by any task"
get https://raw.githubusercontent.com/ryoungj/ObsScaling/main/eval_results/base_llm_benchmark_eval.csv \
    data/obsscaling_base_llm_benchmark_eval.csv
echo "== DataDecide (huggingface.co/datasets/allenai/DataDecide-eval-results; ODC-BY), about 690 MB"
for i in 0 1 2 3; do
    get "https://huggingface.co/api/datasets/allenai/DataDecide-eval-results/parquet/default/train/$i.parquet" \
        data/datadecide/$i.parquet
done
echo "done. Hugging Face re-converts parquet files from time to time; if a regenerated"
echo "task file differs from the included one, the included one is the one the run used."
