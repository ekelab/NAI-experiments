"""
The SLDBench data NAI is run on, checked against the source (2026-09-28): every
task's train and test split fetched again from the Hugging Face datasets-server
(pkuHaowei/sldbench) and compared value by value with sldbench/*.json;
the row counts against the source's own totals; the columns against
upstream/data_loader.py's schema. Any difference is printed and the exit code
is 1.

    python sld_verify_data.py
"""
import json, os, sys, time, urllib.request

import sldbench as S

UPSTREAM_SCHEMA = {   # upstream/data_loader.py, TASK_SCHEMA_MAP
    "data_constrained_scaling_law": (["unique_tokens", "params", "tokens"], ["loss"]),
    "domain_mixture_scaling_law": ([f"proportion_domain_{i+1}" for i in range(5)], [f"loss_domain_{i+1}" for i in range(5)]),
    "lr_bsz_scaling_law": (["lr", "bsz", "data_size", "non_embedding_param_size"], ["lm_loss"]),
    "moe_scaling_law": (["num_experts", "dense_parameter_count"], ["loss_validation"]),
    "sft_scaling_law": (["sft_data_size"], ["sft_loss"]),
    "vocab_scaling_law": (["non_vocab_parameters", "vocab_size", "num_characters"], ["unigram_normalized_loss"]),
    "parallel_scaling_law": (["num_params", "parallel_size"], ["loss"]),
    "easy_question_scaling_law": (["log_flops"], ["brier_score"]),
}


def get(url):
    for attempt in range(10):
        try:
            return json.load(urllib.request.urlopen(url, timeout=60))
        except Exception as e:
            time.sleep(3 * (attempt + 1))
    raise SystemExit("could not fetch " + url)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    bad = 0
    for t in S.TASKS:
        feats, targets, _, _ = S.TASKS[t]
        uf, ut = UPSTREAM_SCHEMA[t]
        if feats != uf or targets != ut:
            print("SCHEMA  %s: ours %s → %s, upstream %s → %s" % (t, feats, targets, uf, ut))
            bad += 1
        for split in ("train", "test"):
            path = os.path.join(S.DATA, f"{t}.{split}.json")
            local = json.load(open(path))["rows"]
            rows, off, total = [], 0, None
            while True:
                d = get("https://datasets-server.huggingface.co/rows?dataset=pkuHaowei/sldbench"
                        f"&config={t}&split={split}&offset={off}&length=100")
                total = d["num_rows_total"]
                rows += [r["row"] for r in d["rows"]]
                off += 100
                if off >= total:
                    break
            diff = 0
            if len(rows) != len(local) or len(rows) != total:
                print("ROWS    %s/%s: local %d, source %d (total %d)" % (t, split, len(local), len(rows), total))
                bad += 1
            for i, (a, b) in enumerate(zip(local, rows)):
                if a != b:
                    diff += 1
                    if diff <= 3:
                        print("VALUE   %s/%s row %d: local %s, source %s" % (t, split, i, a, b))
            bad += diff > 0
            groups = sorted({r["group"] for r in local})
            print("%-30s %-5s rows %5d  groups %3d  differences %d" % (t, split, len(local), len(groups), diff))
    print("OK: the local data are the source's, value for value" if not bad else "PROBLEMS: %d" % bad)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
