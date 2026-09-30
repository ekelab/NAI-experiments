#!/bin/sh
# Reproduce the SLDBench result of NAI v9: nine searches, three pooled
# replicates, each checked with SLDBench's official evaluator.
# Needs: sh reproduce/fetch_data.sh first; Python 3 with numpy.
# Run from the experiment folder:  sh reproduce/run_v9.sh   (about 10 h on a 12-thread PC)
set -e
PY=${PY:-python3}
HERE=$(cd "$(dirname "$0")/.." && pwd)
NAI="$HERE/cli/nai_cli.exe"
OUT=results-repro                      # inside harness/, next to the harness's own paths
cd "$HERE/harness"

COMMON="--engine-groups --holdout all --jobs 2 --threads 12 --seconds 180 --seconds-single 900 \
  --memory-single 3072 --total-memory 4096 --transform both --select nested --monotone --delta 0.003"

# 1. nine searches, candidates kept
for s in 1 2 3 4 5 6 7 8 9; do
  $PY sldbench.py --cli "$NAI" --seed $s $COMMON --out $OUT --tag s$s > $OUT.s$s.log 2>&1
  $PY sld_verify_r2.py --chosen $OUT/chosen_s$s.json --csv $OUT/nai_sldbench_s$s.csv > $OUT/verify_s$s.txt 2>&1
done

# 2. three replicates of NAI v9: the pooled candidates of seeds {1,2,3}, {4,5,6}, {7,8,9}
for p in "1 2 3" "4 5 6" "7 8 9"; do
  set -- $p
  d=$OUT/work/sldbench-p$1$2$3
  mkdir -p $d
  for f in $OUT/work/sldbench-s$1/*.candidates.txt; do
    n=$(basename $f)
    cat $OUT/work/sldbench-s$1/$n $OUT/work/sldbench-s$2/$n $OUT/work/sldbench-s$3/$n > $d/$n
  done
done
$PY sld_rules.py --suite sldbench --cli "$NAI" --out $OUT --tags p123,p456,p789 --rules all-mean \
  --common "--engine-groups --jobs 1 --threads 12 --transform both --select nested --monotone --delta 0.003"

# 3. the official evaluator on each replicate
for p in p123 p456 p789; do
  $PY sld_verify_r2.py --chosen $OUT/chosen_$p-all-mean.json --csv $OUT/nai_sldbench_$p-all-mean.csv \
      > $OUT/verify_$p-all-mean.txt 2>&1
  tail -1 $OUT/verify_$p-all-mean.txt
done
