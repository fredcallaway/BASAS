using Distributed
using ProgressMeter
using Sobol

# println("Building RCall")
# using Pkg
# Pkg.build("RCall")

@everywhere begin
    include("base.jl")
    include("regressions.jl")
end

jls_dir = results_path("summary_stats/jls"; create=true)
json_dir = results_path("summary_stats/json"; create=true)


function sobol(n::Int, box::Box)
    seq = SobolSeq(length(box))
    skip(seq, n)
    [box(Sobol.next!(seq)) for i in 1:n]
end

function write_stats(name, data, box)
    study = parse(Int, name[1])
    results = @showprogress name pmap(grid(30, box)) do prm
        if get(prm, :subjective_slope, -1.) == 0.
            subjective_offset = prm.confidence_slope * mean(flatten(data.confidence))
            prm = (;prm..., subjective_offset)
        end
        model = BDDM(;prm...)
        df = DataFrame(make_sim(model, data; repeats=30))

        (;prm, fit_regressions(df; study)...)
    end
    serialize("$jls_dir/$name", results)
    write("$json_dir/$name.json", json(results))
end


# %% ==================== Load data ====================

data1 = load_human_data(1)
data2 = load_human_data(2)
avg_conf = mean(flatten(data2.confidence))

# use average confidence from study 2 for study 1 (no confidence judgments)
for d in data1
    d.confidence .= avg_conf
end

# %% ==================== configure search space ====================

box = Box(
    base_precision = 0.,
    attention_factor = 0.,
    confidence_slope = (0.001, 0.1, :log),
    cost = (0.001, 0.1, :log),
)

box1 = let 
    µ, σ = empirical_prior(data1)
    update(box; 
        prior_mean = µ,
        prior_precision = 1 / σ^2,
    )
end

box2 = let 
    µ, σ = empirical_prior(data2)
    update(box; 
        prior_mean = µ,
        prior_precision = 1 / σ^2,
    )
end

# %% ==================== jobs ====================

function sbatch_job(job_ids)
    jobs = Dict(
        1 => () -> write_stats("2-main", data2, box2),
        2 => () -> write_stats("2-nometa", data2, update(box2, subjective_slope = 0.)),
        3 => () -> write_stats("2-zero_mean", data2, update(box2, prior_mean = 0.)),
        4 => () -> write_stats("1-main", data1, box1),
        5 => () -> write_stats("1-zero_mean", data1, update(box1, prior_mean = 0.)),
        6 => () -> write_stats("1-flat_prior", data1, update(box1, prior_precision = 1e-8))
    )

    if isempty(job_ids)
        for job in values(jobs)
            job()
        end
    else
        for job_id in job_ids
            jobs[job_id]()
        end
    end
end

# Call the function with a specific job number or without arguments to run all
sbatch_job(parse.(Int, ARGS))
