using DataStructures: OrderedDict
using NamedTupleTools
using Statistics
using StatsBase
using Distributions
using SplitApplyCombine
using ProgressMeter: progress_map
flatten = SplitApplyCombine.flatten

macro infiltry(ex)
    return quote
        try
            $(esc(ex))
        catch
            $(Infiltrator.start_prompt)($(__module__), Base.@locals, $(String(__source__.file)), $(__source__.line))
        end
    end
end

named_tuple(x::NamedTuple) = x
function named_tuple(x)
    n = Tuple(propertynames(x))
    NamedTuple{n}(getproperty.(Ref(x), n))
end

Base.get(name::Symbol) = Base.Fix2(getproperty, name)
Base.get(i::Int) = Base.Fix2(getindex, i)
Base.get(x, name::Symbol) = getproperty(x, name)
Base.get(x, i::Int) = getindex(x, i)
Base.getindex(key) = Base.Fix2(getindex, key)

basetype(f::Type) = f.name.wrapper
basetype(f) = basetype(typeof(f))

flatmap(f, xs...; kws...) = mapreduce(f, vcat, xs...; kws...)

function monte_carlo(f, N=10000)
    N \ mapreduce(+, 1:N) do i
        f()
    end
end

function repeatedly(f, N=10000)
    map(1:N) do i
        f()
    end
end

function mutate(x::T; kws...) where T
    for field in keys(kws)
        if !(field in fieldnames(T))
            error("$(T.name) has no field $field")
        end
    end
    return T([get(kws, fn, getfield(x, fn)) for fn in fieldnames(T)]...)
end

function map_product(f, args...; kws...)
    ks = keys(kws)
    n_pos = length(args)
    map(Iterators.product(args..., values(kws)...)) do x
        pos_args = x[1:n_pos]; kw_args = x[n_pos+1:end]
        f(pos_args...; Dict(zip(ks, kw_args))...)
    end
end

call(f) = f()

function logspace(low, high, n)
    x = range(0, 1, length=n)
    @. exp(log(low) + x * (log(high) - log(low)))
end

function softmax!(x)
    x .= exp.(x .- maximum(x))
    x ./= sum(x)
end
softmax(x) = softmax!(copy(x))

function softmax!(x, i)
    x .= exp.(x .- maximum(x))
    x[i] / sum(x)
end

dictkeys(d::Dict) = (collect(keys(d))...,)
dictvalues(d::Dict) = (collect(values(d))...,)

round1(x) = round(x; digits=1)
round2(x) = round(x; digits=2)
round3(x) = round(x; digits=3)
round4(x) = round(x; digits=4)

fillmissing(x, rep) = ismissing(x) ? rep : x
fillmissing(rep) = x-> ismissing(x) ? rep : x

# namedtuple(d::Dict{String,T}) where {T} =
#     NamedTuple{Symbol.(dictkeys(d))}(dictvalues(d))
# namedtuple(d::Dict{Symbol,T}) where {T} =
#     NamedTuple{dictkeys(d)}(dictvalues(d))
# namedtuple(x) = namedtuple(Dict(fn => getfield(x, fn) for fn in fieldnames(typeof(x))))

Base.map(f::Function) = xs -> map(f, xs)
Base.map(f::Type) = xs -> map(f, xs)
Base.map(f, d::Dict) = [f(k, v) for (k, v) in d]

Base.filter(f::Function) = xs -> filter(f, xs)

Base.dropdims(idx::Int...) = X -> dropdims(X, dims=idx)
Base.reshape(idx::Union{Int,Colon}...) = x -> reshape(x, idx...)


valmap(f, d::AbstractDict) = Dict(k => f(v) for (k, v) in d)
valmap(f, d::OrderedDict) = OrderedDict(k => f(v) for (k, v) in d)
# valmap(f, d::T) where T <: AbstractDict = T(k => f(v) for (k, v) in d)

valmap(f) = d->valmap(f, d)
keymap(f, d::Dict) = Dict(f(k) => v for (k, v) in d)
juxt(fs...) = x -> Tuple(f(x) for f in fs)
repeatedly(f, n) = [f() for i in 1:n]

nanreduce(f, x) = f(filter(!isnan, x))
nanmean(x) = nanreduce(mean, x)
nanstd(x) = nanreduce(std, x)

function Base.write(fn)
    obj -> open(fn, "w") do f
        write(f, string(obj))
    end
end

function writev(fn)
    x -> begin
        write(fn, x)
        run(`du -h $fn`)
    end
end

# type2dict(x::T) where T = Dict(fn=>getfield(x, fn) for fn in fieldnames(T))

function mutate(x::T; kws...) where T
    for field in keys(kws)
        if !(field in fieldnames(T))
            error("$(T.name) has no field $field")
        end
    end
    return T([get(kws, fn, getfield(x, fn)) for fn in fieldnames(T)]...)
end

getfields(x) = (getfield(x, f) for f in fieldnames(typeof(x)))

function grid(; kws...)
    X = map(Iterators.product(values(kws)...)) do x
        (; zip(keys(kws), x)...)
    end
    X
end


function flexmap(f, args; progress=:default, parallel=false, cache="")
    if progress == :default
        progress = parallel && isa(stderr, Base.TTY)
    end
    mapfun = parallel ? pmap : map
    mkpath(cache)
    func = (i, args) -> begin
        if cache == ""
            return f(args)
        end
        file = "$cache/$i"
        if isfile(file)
            try
                return deserialize(file)
            catch
                println("Error deserializing $file; removing it")
                rm(file)
            end
        end
        # cache not available
        res = f(args)
        serialize(file, res)
        return res
    end
    if progress
        progress_map(func, eachindex(args), args; mapfun)
    else
        result, t = @timed mapfun(func, eachindex(args), args)
        println("flexmap completed in $t seconds")
        return result
    end
end