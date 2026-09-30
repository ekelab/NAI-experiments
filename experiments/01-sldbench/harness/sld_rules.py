"""
The choice of form re-run on saved candidates: every run of sldbench.py
--engine-groups keeps each target's candidate forms, and --reselect judges
them again (no search; seconds a task). Used to form the pooled replicates of
NAI v9 and to re-check a published choice.

    python sld_rules.py --suite sldbench --cli ../cli/nai_cli.exe --out DIR --tags p123 --rules all-mean
"""
import argparse, csv, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))

RULES = {                       # name: extra arguments of sldbench.py
    "base": [],                     # the choice of v8
    "all-mean": ["--holdout", "all"],   # the choice of v9
}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="blind")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tags", required=True)
    ap.add_argument("--cli", default=os.path.join(HERE, "..", "build", "nai_cli.exe"))
    ap.add_argument("--rules", default=",".join(RULES))
    ap.add_argument("--dev", default="")         # the development split of sldbench.py (--dev SHARE)
    ap.add_argument("--common", default="--engine-groups --jobs 2 --threads 12 --transform both --select nested "
                                        "--monotone --delta 0.003")
    a = ap.parse_args()
    prefix = {"blind": "blind-", "exp1": "exp1-"}.get(a.suite, "dev-" if a.dev else "")
    table = {}                  # (rule, tag) -> {task: r2}
    for tag in a.tags.split(","):
        for rule in a.rules.split(","):
            newtag = f"{tag}-{rule}"
            cmd = [sys.executable, os.path.join(HERE, "sldbench.py"), "--suite", a.suite, "--cli", os.path.abspath(a.cli),
                   "--out", a.out, "--tag", newtag, "--reselect", tag] + a.common.split() + RULES[rule] + (["--dev", a.dev] if a.dev else [])
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            path = os.path.join(a.out, f"nai_sldbench_{prefix}{newtag}.csv")
            rows = list(csv.DictReader(open(path, encoding="utf-8")))
            table[(rule, tag)] = {r["task"]: float(r["r2"]) for r in rows}
    tasks = list(next(iter(table.values())).keys())
    tags = a.tags.split(",")
    for rule in a.rules.split(","):
        print(f"\n== {rule}")
        print("%-30s" % "task" + "".join("%10s" % t for t in tags) + "%10s" % "mean")
        means = []
        for t in tasks:
            v = [table[(rule, tag)][t] for tag in tags]
            print("%-30s" % t + "".join("%10.4f" % x for x in v) + "%10.4f" % (sum(v) / len(v)))
        for tag in tags:
            means.append(sum(table[(rule, tag)].values()) / len(tasks))
        m = sum(means) / len(means)
        sd = (sum((x - m) ** 2 for x in means) / max(1, len(means) - 1)) ** 0.5
        print("%-30s" % "MEAN" + "".join("%10.4f" % x for x in means) + "%10.4f ± %.4f" % (m, sd))


if __name__ == "__main__":
    main()
