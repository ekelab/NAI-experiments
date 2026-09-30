#!/bin/sh
# Experiment 02 (experiment 1 in its protocol, PROTOCOL.md): the whole
# protocol, step by step. Run from protocol/bench/:  sh exp1/run.sh
# (reproduce/reproduce.sh does). NAI is the public build in the experiment's
# cli/ folder; Julia and Python come from the environment (JULIA, PY).
set -e
NAI="$(pwd)/../../cli/nai_cli.exe"
JULIA=${JULIA:-julia}
PY=${PY:-python3}
OUT=results/exp1
mkdir -p "$OUT"
COMMON="--suite exp1 --engine-groups --holdout all --jobs 2 --threads 12 --seconds 180 --seconds-single 900 \
  --memory-single 3072 --total-memory 4096 --transform both --select nested --monotone --delta 0.003"

# 1. NAI: nine searches, candidates kept
for s in 1 2 3 4 5 6 7 8 9; do
  $PY sldbench.py $COMMON --cli "$NAI" --seed $s --out $OUT --tag s$s > $OUT.s$s.log 2>&1
done

# 2. NAI v9: three replicates, each the pooled candidates of three searches
for p in "1 2 3" "4 5 6" "7 8 9"; do
  set -- $p
  d=$OUT/work/sldbench-exp1-p$1$2$3
  mkdir -p $d
  for f in $OUT/work/sldbench-exp1-s$1/*.candidates.txt; do
    n=$(basename $f)
    cat $OUT/work/sldbench-exp1-s$1/$n $OUT/work/sldbench-exp1-s$2/$n $OUT/work/sldbench-exp1-s$3/$n > $d/$n
  done
done
$PY sld_rules.py --suite exp1 --cli "$NAI" --out $OUT --tags p123,p456,p789 --rules all-mean \
  --common "--engine-groups --jobs 1 --threads 12 --transform both --select nested --monotone --delta 0.003"

# 3. B1–B3
$PY exp1/baselines.py --out $OUT/baselines

# 4. B4 with NAI's time: replicate r gets, per task, the search seconds of its three seeds
r=0
for p in "1 2 3" "4 5 6" "7 8 9"; do
  r=$((r + 1))
  set -- $p
  for task in chinchilla_epoch porian_chinchilla porian_lrbsz datadecide; do
    secs=$($PY -c "
import csv
t = 0.0
for s in ('$1', '$2', '$3'):
    for row in csv.DictReader(open('$OUT/nai_sldbench_exp1-s' + s + '.csv', encoding='utf-8')):
        if row['task'] == '$task':
            t += float(row['search_seconds'])
print(int(t))")
    "$JULIA" -t 12 --project=exp1/sr exp1/sr/b4_sr.jl exp1/tasks $task $r $secs $OUT/b4-r$r
  done
done

# 5. Scores
$PY exp1/evaluate.py --json $OUT/scores.json \
  --pred "NAI-r1=$OUT/work/sldbench-exp1-p123-all-mean/{task}.{target}.pred.txt" \
  --pred "NAI-r2=$OUT/work/sldbench-exp1-p456-all-mean/{task}.{target}.pred.txt" \
  --pred "NAI-r3=$OUT/work/sldbench-exp1-p789-all-mean/{task}.{target}.pred.txt" \
  --pred "B1=$OUT/baselines/{task}.B1.pred.txt" \
  --pred "B2=$OUT/baselines/{task}.B2.pred.txt" \
  --pred "B3=$OUT/baselines/{task}.B3.pred.txt" \
  --pred "B4-r1=$OUT/b4-r1/{task}.B4.pred.txt" \
  --pred "B4-r2=$OUT/b4-r2/{task}.B4.pred.txt" \
  --pred "B4-r3=$OUT/b4-r3/{task}.B4.pred.txt"
