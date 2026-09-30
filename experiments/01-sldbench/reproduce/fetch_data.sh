#!/bin/sh
# Fetch what this experiment needs but cannot redistribute:
#   1. SLDBench's data (Hugging Face pkuHaowei/sldbench; no license stated), into harness/sldbench/
#   2. SLDBench's official evaluator (github.com/linhaowei1/SLD, MIT), into harness/sldbench/upstream/
# The row counts printed should match the table in README.md (section 5);
# harness/sld_verify_data.py compares the data with the source value for value.
# Run from the experiment folder:  sh reproduce/fetch_data.sh
set -e
PY=${PY:-python3}
cd "$(dirname "$0")/../harness"

echo "== 1. SLDBench data (Hugging Face datasets-server)"
$PY -c "import sldbench; sldbench.fetch()"

echo "== 2. SLDBench's official evaluator"
tmp=$(mktemp -d)
git -c core.autocrlf=false clone --quiet --depth 1 https://github.com/linhaowei1/SLD.git "$tmp/SLD"
mkdir -p sldbench/upstream
for f in evaluator.py data_loader.py; do
    src=$(find "$tmp/SLD" -name "$f" -not -path "*/node_modules/*" | head -1)
    [ -n "$src" ] || { echo "not found in the SLD repository: $f"; exit 1; }
    cp "$src" sldbench/upstream/
done
cfg=$(find "$tmp/SLD" -type d -name configs | head -1)
[ -n "$cfg" ] && cp -r "$cfg" sldbench/upstream/
rm -rf "$tmp"
echo "done"
