"""
SLDBench (Lin et al., "Can Language Models Discover Scaling Laws?", ICLR 2026;
github.com/linhaowei1/SLD, data: huggingface.co/datasets/pkuHaowei/sldbench),
run with NAI in group mode and scored the way SLDBench scores: R² on the test
split, which extrapolates, over all test rows of all groups at once.

For every task and target, NAI receives one table with a group column and
returns one form with its constants per group (--forms) and its predictions
for the test rows (--predict). The test rows are used only for the score.
Also runs the blind synthetic suite (--suite blind) and the open-data tasks
of experiment 02 (--suite exp1).

    python sldbench.py --cli ../cli/nai_cli.exe --engine-groups [--seed N] [--only TASK,TASK]
                       [--out DIR] [--tag TAG] [--reselect TAG] [--dev SHARE] ...
"""
import argparse, collections, csv, json, math, os, re, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "sldbench")

# task: (features, targets, variable the test split extrapolates, the rule's constant limit)
TASKS = {
    "parallel_scaling_law":         (["num_params", "parallel_size"], ["loss"], "parallel_size", 4),
    "vocab_scaling_law":            (["non_vocab_parameters", "vocab_size", "num_characters"],
                                     ["unigram_normalized_loss"], "vocab_size", 7),
    "sft_scaling_law":              (["sft_data_size"], ["sft_loss"], "sft_data_size", 4),
    "domain_mixture_scaling_law":   ([f"proportion_domain_{i}" for i in range(1, 6)],
                                     [f"loss_domain_{i}" for i in range(1, 6)], None, 35),
    "moe_scaling_law":              (["num_experts", "dense_parameter_count"], ["loss_validation"],
                                     "dense_parameter_count", 6),
    "data_constrained_scaling_law": (["unique_tokens", "params", "tokens"], ["loss"], "tokens", 7),
    "lr_bsz_scaling_law":           (["lr", "bsz", "data_size", "non_embedding_param_size"], ["lm_loss"],
                                     "non_embedding_param_size", None),
    "easy_question_scaling_law":    (["log_flops"], ["brier_score"], "log_flops", 6),
}

# R² from the paper (Table 2): the human laws and SLDAgent with GPT-5.
PAPER = {
    "parallel_scaling_law": (1.000, 1.000), "vocab_scaling_law": (0.966, 0.987),
    "sft_scaling_law": (0.957, 0.993), "domain_mixture_scaling_law": (0.671, 0.988),
    "moe_scaling_law": (0.703, 0.773), "data_constrained_scaling_law": (0.911, 0.944),
    "lr_bsz_scaling_law": (-0.076, 0.604), "easy_question_scaling_law": (-1.000, -0.305),
}


def fetch():
    """Every task's train and test rows, from the Hugging Face datasets-server."""
    os.makedirs(DATA, exist_ok=True)
    for t in TASKS:
        for split in ("train", "test"):
            rows, off = [], 0
            while True:
                u = ("https://datasets-server.huggingface.co/rows?dataset=pkuHaowei/sldbench"
                     f"&config={t}&split={split}&offset={off}&length=100")
                for attempt in range(8):
                    try:
                        d = json.load(urllib.request.urlopen(u, timeout=60))
                        break
                    except Exception as e:   # 502 and 429 are common there
                        print("retry:", e)
                        time.sleep(3 * (attempt + 1))
                else:
                    raise SystemExit("could not fetch " + u)
                rows += [r["row"] for r in d["rows"]]
                off += 100
                if off >= d["num_rows_total"]:
                    break
            json.dump({"features": [f["name"] for f in d["features"]], "rows": rows},
                      open(os.path.join(DATA, f"{t}.{split}.json"), "w"))
            print(t, split, len(rows))


# --dev SHARE: a development split made of the TRAINING rows only; in each
# group the top SHARE along the variable the test extrapolates plays the test.
# The benchmark's test file is not opened.
DEV_SHARE = 0.0


def load(task, split):
    """{group: rows}, groups sorted as upstream's data_loader sorts them."""
    if DEV_SHARE:
        rows = json.load(open(os.path.join(DATA, f"{task}.train.json")))["rows"]
    else:
        rows = json.load(open(os.path.join(DATA, f"{task}.{split}.json")))["rows"]
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
    if DEV_SHARE:
        extrap = TASKS[task][2]
        out = {}
        for g, rs in groups.items():
            if extrap:
                # Whole values held out, the largest ones, until about SHARE
                # of the rows: a true extrapolation, as the test's (a grid
                # repeats each model size many times; no size on both sides).
                values = sorted({r[extrap] for r in rs}, reverse=True)
                held, count = set(), 0
                for v in values[:-1]:                  # the smallest value always stays
                    c = sum(1 for r in rs if r[extrap] == v)
                    if held and abs(count + c - DEV_SHARE * len(rs)) > abs(count - DEV_SHARE * len(rs)):
                        break
                    held.add(v)
                    count += c
                keep = [r for r in rs if (r[extrap] in held) == (split == "test")]
            else:
                idx = sorted(range(len(rs)), key=lambda i: (i * 2654435761 + 1234) % 4294967296)
                n = max(1, round(DEV_SHARE * len(rs)))
                chosen = set(idx[-n:])
                keep = [r for i, r in enumerate(rs) if (i in chosen) == (split == "test")]
            out[g] = keep
        groups = out
    return dict(sorted(groups.items()))


def write_table(path, names, target, rows, with_y=True):
    # repr keeps each number as the source printed it: a loss of 2.1113 is
    # known to 4 decimals, and NAI reads that precision from the notation.
    # Counts stored as float64 (368123904.0) are written as integers - with
    # ".0" NAI would read them as rounded to a tenth.
    def num(v):
        return str(int(v)) if isinstance(v, float) and v.is_integer() and abs(v) < 2 ** 53 else repr(v)
    with open(path, "w", encoding="utf-8") as f:
        f.write(" ".join(names + ([target] if with_y else [])) + "\n")
        for r in rows:
            f.write(" ".join(num(r[c]) for c in names + ([target] if with_y else [])) + "\n")


def r2(pred, truth):
    """SLDBench's R²: 1 − MSE/Var over the test rows (upstream/evaluator.py)."""
    if not all(math.isfinite(x) for x in pred):
        return -1.0                                   # upstream: NaN or Inf is a failure, R² = −1
    m = sum(truth) / len(truth)
    var = sum((t - m) ** 2 for t in truth) / len(truth)
    try:
        mse = sum((p - t) ** 2 for p, t in zip(pred, truth)) / len(truth)
    except OverflowError:                             # finite predictions, astronomically wrong
        return -math.inf
    return 1.0 - mse / var if var > 1e-9 else float("nan")


def write_group_table(path, names, target, groups):
    """One table for every group: a "group" column, tab-separated (SLDBench's
    labels have spaces and commas)."""
    def num(v):
        return str(int(v)) if isinstance(v, float) and v.is_integer() and abs(v) < 2 ** 53 else repr(v)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(["group"] + names + [target]) + "\n")
        for g, rows in groups.items():
            for r in rows:
                f.write("\t".join([str(g)] + [num(r[c]) for c in names + [target]]) + "\n")


def run_engine(a, tasks, log):
    """--engine-groups: the choice of the form by NAI itself - one
    call per task and target, on a table with a group column; the test's
    predictions come from NAI (--predict), its R² as the benchmark computes it."""
    work = os.path.join(a.out, "work", "sldbench-" + a.tag)
    os.makedirs(work, exist_ok=True)
    out_csv = os.path.join(a.out, "nai_sldbench_%s.csv" % a.tag)
    chosen = {}
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["task", "groups", "r2", "r2_heldout", "human_r2", "sldagent_r2", "constants", "limit",
                    "forms_tried", "search_seconds", "outcomes", "form"])
        for t in tasks:
            feats, targets, extrap, _ = TASKS[t]
            train, test = load(t, "train"), load(t, "test")
            groups = {g: rows for g, rows in train.items() if g in test}
            single = len(groups) == 1
            per_target, cvs, ks, tried, shown, secs = [], [], [], 0, [], 0.0
            for y in targets:
                stem = os.path.join(work, f"{t}.{y}")
                tr, te, pr, fm, lg = (stem + x for x in (".train.txt", ".test.txt", ".pred.txt", ".forms.txt", ".log.txt"))
                write_group_table(tr, feats, y, groups)
                write_group_table(te, feats, y, {g: test[g] for g in groups})
                for x in (pr, fm):
                    if os.path.exists(x):
                        os.remove(x)
                budget = a.seconds_single if a.seconds_single and single else a.seconds
                cmd = [a.cli, "--data", tr, "--lang", "en", "--probability", "no", "--no-learn", "--threads", str(a.threads),
                       "--seconds", str(budget), "--predict", te, pr, "--forms", fm,
                       "--max-constants", str(constant_limit(t)), "--delta", str(SELECT["delta"]), "--seed", str(a.seed),
                       # one search at a time where each may take --memory-single MB per stage
                       "--jobs", str(1 if (a.memory_single and single) else a.jobs)]
                if extrap:
                    cmd += ["--extrap", extrap]
                if a.transform == "raw":
                    cmd += ["--no-log-inputs"]
                if SELECT["monotone"] and MONOTONE.get(t):
                    cmd += ["--monotone", ",".join(MONOTONE[t])]
                if a.memory_single and single:
                    cmd += ["--max-memory", str(a.memory_single)]
                if a.total_memory:
                    cmd += ["--max-total-memory", str(a.total_memory)]
                if a.restarts > 1:
                    cmd += ["--restarts", str(a.restarts)]
                if a.holdout:
                    cmd += ["--holdout-vars", a.holdout]
                if a.holdout_score != "mean":
                    cmd += ["--holdout-score", a.holdout_score]
                # every candidate form kept; --reselect TAG judges an earlier run's candidates again
                if a.reselect:
                    src = os.path.join(a.out, "work", "sldbench-" + a.reselect, f"{t}.{y}.candidates.txt")
                    cmd += ["--candidates-in", src]
                else:
                    cmd += ["--candidates-out", stem + ".candidates.txt"]
                t0 = time.time()
                p = subprocess.run(cmd, capture_output=True)
                secs += time.time() - t0
                out = p.stdout.decode("utf-8", "replace")
                err = p.stderr.decode("utf-8", "replace")
                open(lg, "w", encoding="utf-8").write(out + ("\n--- stderr ---\n" + err if err.strip() else ""))
                if p.returncode not in (0, 1):             # 1: no law found, a normal outcome
                    log(f"  !! {t}/{y}: nai_cli exited with {p.returncode}: {err.strip()[-300:]}")
                pred = [float(v) for v in open(pr, encoding="utf-8").read().split()] if os.path.exists(pr) else []
                truth = [r[y] for g in groups for r in test[g]]
                per_target.append(r2(pred, truth) if len(pred) == len(truth) else -1.0)
                cv, k, form, judged = float("nan"), 0, "", 0
                for line in out.splitlines():
                    m = re.search(r"constants: (\d+), held-out R² ([-\d.]+), forms judged (\d+)", line)
                    if m:
                        k, cv, judged = int(m.group(1)), float(m.group(2)), int(m.group(3))
                    if line.startswith("ONE FORM FOR THE GROUPS: "):
                        form = line.split(": ", 1)[1]
                cvs.append(cv)
                ks.append(k)
                tried += judged
                shown.append(form)
                if os.path.exists(fm):
                    first = open(fm, encoding="utf-8").readline().rstrip("\n").split("\t")
                    if len(first) == 2 and first[0] == "form":
                        chosen.setdefault(t, []).append({"target": y, "form": first[1], "logvars": [], "logy": False})
                log("  %s/%s: %d forms, chosen: R2 held-out %.4f, %d constants: %s" % (t, y, judged, cv, k, form))
            score = sum(per_target) / len(per_target)
            cv = sum(cvs) / len(cvs) if cvs else float("nan")
            human, agent = PAPER.get(t, (float("nan"), float("nan")))
            w.writerow([t, len(groups), "%.6f" % score, "%.6f" % cv, human, agent, max(ks) if ks else "",
                        constant_limit(t), tried, "%.0f" % secs, "engine", " | ".join(shown)])
            f.flush()
            json.dump(chosen, open(os.path.join(a.out, "chosen_%s.json" % a.tag), "w"), indent=1)
            log("%-30s R2=%9.4f (held-out %7.4f)  human %6.3f  SLDAgent %6.3f  constants %s/%d  engine, %.0f s" % (
                t, score, cv, human, agent, max(ks) if ks else "-", constant_limit(t), secs))


SELECT = {"delta": 0.003, "monotone": False}

# Declared assumption (--monotone): the prediction must not rise with these
# variables (model size, amount of data). Variables with an optimum are free.
MONOTONE = {
    "parallel_scaling_law": ["num_params", "parallel_size"],
    "vocab_scaling_law": ["non_vocab_parameters", "num_characters"],
    "sft_scaling_law": ["sft_data_size"],
    "moe_scaling_law": ["num_experts", "dense_parameter_count"],
    "data_constrained_scaling_law": ["unique_tokens", "params", "tokens"],
    "lr_bsz_scaling_law": ["data_size", "non_embedding_param_size"],
    # the blind suite (blind_scaling.py)
    "chinchilla": ["N", "D"], "kaplan": ["N"], "constrained": ["unique_tokens", "params", "tokens"],
    "lr": ["N"], "moe": ["num_experts", "dense_parameter_count"], "finetune": ["sft_data_size"], "lrbsz": ["data_size", "N"],
}


def constant_limit(task):
    limit = TASKS[task][3] or 10                      # lr_bsz: free - ten at most here
    if task == "domain_mixture_scaling_law":
        limit //= len(TASKS[task][1])                  # 35 over five targets: 7 each
    return limit


def main():
    import awake
    awake.keep_awake()                  # no sleeping mid-run: the searches' clocks would run on (awake.py)
    sys.stdout.reconfigure(encoding="utf-8")     # formulas carry Σ, Π, θ: not in the Windows console's code page
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", default=os.path.join(HERE, "..", "build", "nai_cli.exe"))
    ap.add_argument("--seconds", type=float, default=300)
    ap.add_argument("--seconds-single", type=float, default=0)   # the budget of a one-group task (0: --seconds)
    ap.add_argument("--memory-single", type=int, default=0)      # MB of values per search stage for a one-group task (0: NAI's 768)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--suite", default="sldbench", choices=["sldbench", "blind", "exp1"])   # blind: blind_scaling.py; exp1: exp1/prepare.py
    ap.add_argument("--select", default="nested")                                   # accepted for compatibility; the choice is NAI's
    ap.add_argument("--monotone", action="store_true")                               # no rise of the loss with scale (MONOTONE)
    ap.add_argument("--delta", type=float, default=0.003)
    ap.add_argument("--engine-groups", action="store_true")                          # required: NAI in group mode
    ap.add_argument("--total-memory", type=int, default=0)   # engine: MB of values of all stages together (0: no limit)
    ap.add_argument("--holdout-score", default="mean", choices=["mean", "min"])   # engine: how the held-out scores are combined
    ap.add_argument("--dev", type=float, default=0.0)         # a development split of the training rows (share held out); the test is not read
    ap.add_argument("--restarts", type=int, default=1)       # engine: searches with seeds seed…seed+K−1, candidates pooled
    ap.add_argument("--holdout", default="")          # engine: held-out rows beyond the fitted range along these variables ("all"; default: the extrapolated one)
    ap.add_argument("--reselect", default="")         # engine: no search, the candidates of run TAG judged again
    ap.add_argument("--seed", type=int, default=1)                                   # NAI's seed (its split of the rows)
    ap.add_argument("--transform", default="raw", choices=["raw", "both", "all"])    # both: + ln of scale variables; all: + log–log
    a = ap.parse_args()
    global DATA
    if a.dev:
        global DEV_SHARE
        DEV_SHARE = a.dev
        a.tag = "dev-" + a.tag
        if a.reselect:
            a.reselect = "dev-" + a.reselect
    SELECT["monotone"] = a.monotone
    SELECT["delta"] = a.delta
    if not os.path.exists(a.cli):
        raise SystemExit("no NAI at " + a.cli)    # a missing program must stop the run, not end it quietly
    if not a.engine_groups:
        raise SystemExit("this harness runs NAI in group mode only: pass --engine-groups")
    if a.suite == "blind":
        DATA = os.path.join(HERE, "blind", "scaling")
        TASKS.clear()
        for name, t in json.load(open(os.path.join(DATA, "tasks.json"))).items():
            TASKS[name] = (t["features"], t["targets"], t["extrap"], t["limit"])
        a.tag = "blind-" + a.tag
        if a.reselect:
            a.reselect = "blind-" + a.reselect
    if a.suite == "exp1":
        # Experiment 02: open scaling data prepared by exp1/prepare.py; the
        # variables that may not raise the loss are stated in its tasks.json.
        DATA = os.path.join(HERE, "exp1", "tasks")
        TASKS.clear()
        for name, t in json.load(open(os.path.join(DATA, "tasks.json"))).items():
            TASKS[name] = (t["features"], t["targets"], t["extrap"], t["limit"])
            MONOTONE[name] = t.get("monotone", [])
        a.tag = "exp1-" + a.tag
        if a.reselect:
            a.reselect = "exp1-" + a.reselect
    if a.suite == "sldbench" and (a.fetch or not os.path.exists(os.path.join(DATA, "sft_scaling_law.test.json"))):
        fetch()
    tasks = [t for t in TASKS if not a.only or t in a.only.split(",")]
    work = os.path.join(a.out, "work", "sldbench-" + a.tag)
    os.makedirs(work, exist_ok=True)
    report = open(os.path.join(a.out, "nai_sldbench_%s.log" % a.tag), "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        report.write(msg + "\n")
        report.flush()

    run_engine(a, tasks, log)


if __name__ == "__main__":
    main()
