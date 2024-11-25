export Global, Varying, Fixed, PartitionedFunction, HybridModel, setbounds, setup
export HybridSymbolic, SymbolTypes

abstract type SymbolTypes end

struct Varying <: SymbolTypes end
struct Global <: SymbolTypes end
struct Fixed <: SymbolTypes end

abstract type HybridSymbolic end

struct PartitionedFunction{F,O,A1,A2,A3,A4,V} <: HybridSymbolic
    func::F
    args::A1
    global_args::A2
    fixed_args::A3
    varying_args::A4
    fixed_vals::V
    opt_func::O

    function PartitionedFunction(
        func::F, 
        args::A1,
        global_args::A2,
        fixed_args::A3,
        varying_args::A4,
        fixed_vals::V,
        opt_func::O
    ) where {F,O,A1,A2,A3,A4,V}
        new{F,O,A1,A2,A3,A4,V}(
            func,
            args,
            global_args,
            fixed_args,
            varying_args,
            fixed_vals,
            opt_func
        )
    end
end

@proto struct HybridModel{T} <: HybridSymbolic
    nn::Lux.Chain
    func::PartitionedFunction
    p_min::T 
    p_max::T 
end

function HybridModel(nn::Lux.Chain, func::PartitionedFunction)
    return HybridModel(nn, func, nothing, nothing)
end
# TODO: This needs to be more general. i.e. ŷ = NN(α * NN(x) + β).

function (m::HybridModel)(X::Matrix{Float32}, params, st)
    ps = params.nn
    globals = params.globals
    n_varargs = length(m.func.varying_args)
    out_NN = m.nn(X, ps, st)[1]
    out = m.func.opt_func(tuple([out_NN[i,:] for i = 1:n_varargs]...), globals)
    return out
end
function (m::HybridModel)(X::Vector{Float32}, params, st)
    ps = params.nn
    globals = params.globals
    n_varargs = length(m.func.varying_args)
    out_NN = m.nn(X, ps, st)[1]
    out = m.func.opt_func(tuple([[out_NN[1]] for i = 1:n_varargs]...), globals)
    return out[1]
end

# Assumes that the last layer has sigmoid activation function
function setbounds(m::HybridModel, bounds::Dict{Symbol, Tuple{T,T}}) where {T}
    n_args = length(m.func.varying_args)
    p_min = zeros(Float32, n_args)
    p_max = zeros(Float32, n_args)
    for (i,arg) in enumerate(Symbol.(m.func.varying_args))
        @assert arg in keys(bounds)
        p_min[i] = bounds[arg][1]
        p_max[i] = bounds[arg][2]
    end
    p_range = p_max .- p_min
    wf = WrappedFunction((x) -> x .* (p_range) .+ p_min)
    new_nn = Chain(m.nn, wf)
    return HybridModel(new_nn, m.func, p_min, p_max)
end

function setup(rng::AbstractRNG, m::HybridModel)
    return Lux.setup(rng, m.nn)
end
