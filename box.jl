using DataStructures: OrderedDict

struct Box
    dims::OrderedDict
end

Box(dims...) = Box(OrderedDict(dims))
Box(;dims...) = Box(OrderedDict(dims...))

Base.length(b::Box) = length(b.dims)
Base.getindex(box::Box, k) = box.dims[k]

function update(box::Box; dims...)
    d = OrderedDict{Symbol, Any}(box.dims)
    for (k, v) in dims
        d[k] = v
    end
    Box(d)
end

function Base.display(box::Box)
    println("Box")
    for p in pairs(box.dims)
        println("  ", p)
    end
end

linscale(x, low, high) = low + x * (high-low)
logscale(x, low, high) = exp(log(low) + x * (log(high) - log(low)))

unlinscale(x, low, high) = (x - low) / (high-low)
unlogscale(x, low, high) = (log(x) - log(low)) / (log(high) - log(low))

function rescale(d, x::Real)
    scale = :log in d ? logscale : linscale
    scale(x, d[1], d[2])
end

rescale(d, x::Union{Vector{<:Real},AbstractRange{<:Real}}) = [rescale(d, xi) for xi in x]

function unscale(d, x)
    scale = :log in d ? unlogscale : unlinscale
    scale(x, d[1], d[2])
end



is_free(::Union{<:Number, Missing, Nothing, <:AbstractString}) = false
is_free(x) = try
    length(x) > 1
catch
    0
end

n_free(b::Box) = sum(is_free, values(b.dims))
free(b::Box) = [k for (k,d) in b.dims if is_free(d)]


function apply(box::Box, x::Vector{Float64})
    xs = Iterators.Stateful(x)
    prs = map(collect(box.dims)) do (name, dim)
        if is_free(dim)
            name => rescale(dim, popfirst!(xs))
        else
            name => dim
        end
    end
    (;prs...)
end

function apply(box::Box, d::AbstractDict)
    x = Float64[]
    for (name, dim) in box.dims
        if is_free(dim)
            push!(x, unscale(dim, d[name]))
        end
    end
    return x
end

(box::Box)(x) = apply(box, x)

# valmap(f, d::AbstractDict) = Dict(k => f(v) for (k, v) in d)
function grid(n::Int, box::Box)
    xs = range(0, 1, length=n)
    kws = map(box.dims) do (k, d)
        length(d) == 1 && return [d]
        k => [rescale(d, x) for x in xs]
    end
    grid(;kws...)
end
