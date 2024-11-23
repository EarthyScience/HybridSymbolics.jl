export Global, Varying, Fixed, PartitionedFunction, HybridModel
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

struct HybridModel <: HybridSymbolic
    nn::Lux.Chain
    func::PartitionedFunction
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