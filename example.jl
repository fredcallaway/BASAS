using Pkg; Pkg.activate(".")

import DataFrames: DataFrame
import CSV

read_csv(file) = CSV.read(file, DataFrame)
function write_csv(file, df; quiet=false)
    @assert endswith(file, ".csv") "File must end with .csv"
    mkpath(dirname(file))
    CSV.write(file, df)
    !quiet && println("Wrote $file")
end

include("model.jl")
include("dc.jl")

# Note: time is always in seconds
const PRESENTATION_DURATIONS = Dict(
    "shortfirst" => [Normal(.2, .05), Normal(.5, .1)],
    "longfirst" => [Normal(.5, .1), Normal(.2, .05)]
)

function prepare_data(data::DataFrame)::Vector{SimTrial}
    map(eachrow(data)) do d
        SimTrial(
            value=[d.val1, d.val2],  # NOTE: val1 is for the first-shown item
            presentation_distributions=PRESENTATION_DURATIONS[d.order],
        )
    end
end

function create_model(trials::Vector{SimTrial}; base_precision=0.05, cost=0.02, bias=1., attention_factor=1.)::BDDM
    µ, σ = empirical_prior(trials; bias)  # bias scales the mean, α in PLoS paper
    model = BDDM(;
        prior_mean=µ,
        prior_precision=σ^-2,
        attention_factor,
        base_precision, 
        cost,
    )
end

function simulation_frame(model::BDDM, trials::Vector{SimTrial})::DataFrame
    map(trials) do t
        sim = simulate(model, t; save_presentation=true, max_rt=5.)
    
        val1, val2 = t.value
        # conf1, conf2 = t.confidence
        order = let
            m1, m2 = mean.(t.presentation_distributions)
            m1 > m2 ? :longfirst : :shortfirst
        end
        pt1, pt2 = map((1,2)) do item
            round(sum(sim.presentation_durations[item:2:end]); digits=3)
        end
        initpresdur1, initpresdur2 = map((1, 2)) do item
            get(sim.presentation_durations, item, missing)
        end
    
        (;
            val1, val2, 
            order,
            sim.choice, 
            rt = round(sim.rt; digits=3), 
            pt1, pt2,
            initpresdur1, initpresdur2
        )
    end |> DataFrame
end

# %% --------

data = read_csv("data/study1.csv")
trials = prepare_data(data)
model = create_model(trials)
sim = simulation_frame(model, trials)
write_csv("data/study1_sim.csv", sim)
