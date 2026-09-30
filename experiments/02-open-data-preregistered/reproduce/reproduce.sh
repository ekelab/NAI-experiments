#!/bin/sh
# Experiment 02: the protocol, step by step (protocol/bench/exp1/run.sh), after
# installing the Julia packages of baseline B4 at the versions of the run
# (protocol/bench/exp1/sr/Manifest.toml).
# Run from the experiment folder:  JULIA=julia PY=python3 sh reproduce/reproduce.sh
# About 11 h on a 12-thread PC. Results go to protocol/bench/results/exp1/.
set -e
export JULIA=${JULIA:-julia}
export PY=${PY:-python3}
cd "$(dirname "$0")/../protocol/bench"
"$JULIA" --project=exp1/sr -e 'using Pkg; Pkg.instantiate()'
sh exp1/run.sh
