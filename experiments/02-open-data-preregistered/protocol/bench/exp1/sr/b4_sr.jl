# Baseline B4 of experiment 1 (docs/prereg/EXP1.md): general symbolic regression
# - SymbolicRegression.jl, the engine of PySR (Cranmer 2023), version pinned by
# Manifest.toml - on the same rows NAI sees.
#
#   operators    + − × ÷ ^, exp, log, sqrt, sin, cos (NAI's alphabet on continuous data)
#   inputs       the variables, and ln of the scale variables (positive, a decade
#                or more in the training rows) - as NAI's second search sees them
#   groups       one form with K = 4 parameters of its own for each group
#                (a template expression); one group: a plain expression
#   budget       wall-clock seconds given per call (the rule: NAI's time on the task)
#   choice       SymbolicRegression.jl's default (choose_best: the best score among
#                the equations whose loss is within 1.5× of the lowest)
#
#   julia -t 12 --project=bench/exp1/sr bench/exp1/sr/b4_sr.jl DATA_DIR TASK SEED SECONDS OUT_DIR

using SymbolicRegression, JSON3, Random
using SymbolicRegression: machine, fit!, predict, report

function rows_of(path)
    d = JSON3.read(read(path, String))
    rows = collect(d.rows)
    groups = sort(unique(String(r.group) for r in rows))
    ordered = reduce(vcat, [[r for r in rows if String(r.group) == g] for g in groups])
    return ordered, groups
end

function main(data, task, seed, seconds, out)
    tasks = JSON3.read(read(joinpath(data, "tasks.json"), String))
    t = tasks[Symbol(task)]
    feats = [String(f) for f in t.features]
    target = String(t.targets[1])
    train, groups = rows_of(joinpath(data, "$task.train.json"))
    test, _ = rows_of(joinpath(data, "$task.test.json"))
    col(rs, f) = Float64[Float64(r[Symbol(f)]) for r in rs]
    # scale variables: positive in every training row, a decade or more
    scale = [f for f in feats if minimum(col(train, f)) > 0 && maximum(col(train, f)) / minimum(col(train, f)) >= 10]
    names = vcat(feats, ["ln_" * f for f in scale])
    table(rs) = NamedTuple{Tuple(Symbol.(names))}(Tuple(vcat([col(rs, f) for f in feats], [log.(col(rs, f)) for f in scale])))
    Xtr, Xte = table(train), table(test)
    ytr = col(train, target)
    gidx(rs) = [findfirst(==(String(r.group)), groups) for r in rs]
    common = (; niterations=10^6, timeout_in_seconds=Float64(seconds), maxsize=30,
               binary_operators=[+, -, *, /, ^], unary_operators=[exp, log, sqrt, sin, cos],
               parallelism=:multithreading, seed=seed)
    if length(groups) == 1
        model = SRRegressor(; common...)
        mach = machine(model, Xtr, ytr)
        fit!(mach; verbosity=0)
        pred = predict(mach, Xte)
    else
        G = length(groups)
        nv = length(names)
        # f(inputs…, p1[g], p2[g], p3[g], p4[g]): one form, four parameters per group
        spec = if nv == 1
            @template_spec(expressions=(f,), parameters=(p1=G, p2=G, p3=G, p4=G)) do x1, class
                f(x1, p1[class], p2[class], p3[class], p4[class])
            end
        elseif nv == 2
            @template_spec(expressions=(f,), parameters=(p1=G, p2=G, p3=G, p4=G)) do x1, x2, class
                f(x1, x2, p1[class], p2[class], p3[class], p4[class])
            end
        elseif nv == 3
            @template_spec(expressions=(f,), parameters=(p1=G, p2=G, p3=G, p4=G)) do x1, x2, x3, class
                f(x1, x2, x3, p1[class], p2[class], p3[class], p4[class])
            end
        elseif nv == 4
            @template_spec(expressions=(f,), parameters=(p1=G, p2=G, p3=G, p4=G)) do x1, x2, x3, x4, class
                f(x1, x2, x3, x4, p1[class], p2[class], p3[class], p4[class])
            end
        else
            error("B4: $nv inputs with groups - no template written")
        end
        Xtr = merge(Xtr, (; class=gidx(train)))
        Xte = merge(Xte, (; class=gidx(test)))
        model = SRRegressor(; common..., expression_spec=spec)
        mach = machine(model, Xtr, ytr)
        fit!(mach; verbosity=0)
        pred = predict(mach, Xte)
    end
    r = report(mach)
    mkpath(out)
    open(joinpath(out, "$task.B4.pred.txt"), "w") do io
        for v in pred
            println(io, string(Float64(v)))
        end
    end
    open(joinpath(out, "$task.B4.log.txt"), "w") do io
        println(io, "task $task seed $seed seconds $seconds groups $(length(groups)) inputs $(join(names, ","))")
        println(io, "chosen: ", r.equations[r.best_idx])
        println(io, "front:")
        for (i, e) in enumerate(r.equations)
            println(io, "  ", r.complexities[i], "  ", r.losses[i], "  ", e)
        end
    end
    println("$task: ", r.equations[r.best_idx])
end

main(ARGS[1], ARGS[2], parse(Int, ARGS[3]), parse(Float64, ARGS[4]), ARGS[5])
