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
#
function (m::HybridModel)(X::VecOrMat{Float32}, params, st; forcings = nothing, return_parameters::Val{T} = Val(false)) where {T}
    if T 
        return runHybridModelAll(m, X, params, st; forcings = forcings, return_parameters = return_parameters)
    else
        return runHybridModelSimple(m, X, params, st; forcings = forcings)
    end
end

function runHybridModelSimple(m::HybridModel, X::Matrix{Float32}, params, st; forcings)
    ps = params.nn
    globals = params.globals
    n_varargs = length(m.func.varying_args)
    out_NN = m.nn(X, ps, st)[1]
    out = m.func.opt_func(tuple([out_NN[i,:] for i = 1:n_varargs]...), globals; forcings = forcings)
    return out
end
function runHybridModelSimple(m::HybridModel, X::Vector{Float32}, params, st; forcings)
    ps = params.nn
    globals = params.globals
    n_varargs = length(m.func.varying_args)
    out_NN = m.nn(X, ps, st)[1]
    out = m.func.opt_func(tuple([[out_NN[1]] for i = 1:n_varargs]...), globals; forcings = forcings)
    return out[1]
end

function runHybridModelAll(m::HybridModel, X::Vector{Float32}, params, st; return_parameters::Val{true}, forcings)
    ps = params.nn
    globals = params.globals
    n_varargs = length(m.func.varying_args)
    out_NN = m.nn(X, ps, st)[1]
    y = m.func.opt_func(tuple([out_NN[i,:] for i = 1:n_varargs]...), globals; forcings = forcings)
    D = Dict{Symbol, Float32}()
    D[:out] = y[1]
    for (i, param) in enumerate(m.func.varying_args)
        D[Symbol(param)] = out_NN[i,1]
    end
    for (i, param) in enumerate(m.func.global_args)
        D[Symbol(param)] = globals[i]
    end
    for (i, param) in enumerate(m.func.fixed_args)
        D[Symbol(param)] = m.func.fixed_vals[i]
    end
    return D
end
function runHybridModelAll(m::HybridModel, X::Matrix{Float32}, params, st; return_parameters::Val{true}, forcings)
    ps = params.nn
    globals = params.globals
    n_varargs = length(m.func.varying_args)
    out_NN = m.nn(X, ps, st)[1]
    y = m.func.opt_func(tuple([[out_NN[1]] for i = 1:n_varargs]...), globals; forcings = forcings)
    D = Dict{Symbol, Vector{Float32}}()
    D[:out] = y[1]
    for (i, param) in enumerate(m.func.varying_args)
        D[Symbol(param)] = out_NN[i,:]
    end
    for (i, param) in enumerate(m.func.global_args)
        D[Symbol(param)] = ones(Float32, size(X,1)) .* globals[i]
    end
    for (i, param) in enumerate(m.func.fixed_args)
        D[Symbol(param)] = ones(Float32, size(X,1)) .* m.func.fixed_vals[i]
    end
    return D
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
