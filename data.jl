using JSON
using TypedTables
using SplitApplyCombine

MAX_RT = 5
max_rt(t::Trial) = Int(MAX_RT / t.dt)

if !@isdefined PRESENTATION_DURATIONS
    const PRESENTATION_DURATIONS = Dict(
        :shortfirst => [Normal(.2, .05), Normal(.5, .1)],
        :longfirst => [Normal(.5, .1), Normal(.2, .05)]
    )
end

function parse_indpres(x)
    if x isa Real
        return [x]
    elseif x[1] isa Real
        @assert x[1] == 0
        @assert length(x) == 2
        return [x[2]]
    else
        x1, x2 = x
        @assert length(x) == 2
        # @assert x1 isa Vector && x2 isa Vector
        if !(x1 isa Vector && x2 isa Vector)
            @show x1 x2
            error()
        end
        return x1 .+ x2
    end
end

# %% --------

function get_confidence(d)
    if "fstConfidence" in keys(d)
        Float64[d["fstConfidence"], d["sndConfidence"]]
    else
        #guess = median(flatten(load_human_data().confidence))
        guess = 4.  # this is what we get from the line above
        [guess, guess]
    end
end

function load_human_data(path::String)
    raw_data = open(JSON.parse, path);
    map(raw_data) do d
        presentation_duration = parse_indpres(d["IndPresDur"])
        avg_first = mean(presentation_duration[1:2:end-1])
        avg_second = mean(presentation_duration[2:2:end-1])
        (
            subject = d["SubNum"],
            value = [d["fstItemV"], d["sndItemVal"]],
            confidence = get_confidence(d),
            presentation_duration,
            # nfix = length(presentation_duration),
            order = avg_first > avg_second ? :longfirst : :shortfirst,
            choice = d["isFirstChosen"] ? 1 : 2,
            rt = d["RT"]
        )
    end |> skipmissing |> collect |> Vector{NamedTuple} |> Table
end
load_human_data(number::Int) = load_human_data("data/Study$number.json")

"Discretize presentation times while preventing rounding error from accumulating"
function discretize_presentation_times(durations, dt)
    # see test in tests.jl
    out = Int[]
    err = 0.
    for d in durations
        push!(out, Int(d ÷ dt))
        err += d % dt
        if err > dt/2
            out[end] += 1
            err -= dt
        end
    end
    out
end

function HumanTrial(d::NamedTuple; µ, σ, dt)
    presentation_distributions = PRESENTATION_DURATIONS[d.order]
    real_presentation_times = discretize_presentation_times(d.presentation_duration, dt)
    rt = sum(real_presentation_times)  # discretized
    HumanTrial((d.value .- µ) ./ σ, d.confidence, presentation_distributions, 
               real_presentation_times, d.subject, d.choice, rt, dt)
end

function prepare_trials(data; dt=DEFAULT_DT, normalize_value=false)
    µ, σ = normalize_value ? juxt(mean, std)(flatten(data.value)) : (0, 1)
    trials = map(data) do d
        HumanTrial(d; µ, σ, dt)
    end
    map(trials) do t
        if t.rt > max_rt(t)
            mutate(t, rt=max_rt(t))
        else
            t
        end
    end
end 

function empirical_prior(data; α=1)
    µ, σ = juxt(mean, std)(flatten(data.value))
    α * µ, σ
end

# %% --------

function table(trials::Vector{HumanTrial})
    map(trials) do t
        x = ntfromstruct(t)
        m1, m2 = mean.(x.presentation_times)
        order = m1 > m2 ? :longfirst : :shortfirst
        presentation_duration = x.real_presentation_times .* t.dt
        total_presentation = [sum(presentation_duration[1:2:end]), sum(presentation_duration[2:2:end])]
        (;x.subject, x.value, x.confidence, presentation_duration, order, x.choice, rt=x.rt * t.dt)
    end |> Table
end

