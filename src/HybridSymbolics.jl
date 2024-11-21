module HybridSymbolics

using Symbolics
using SymbolicUtils
using Lux
using Optimisers
using NNlib
using ADTypes
using ForwardDiff
using ProgressMeter


struct Varying
end
struct Global
end
struct Fixed
end

struct StructuredFunction{T1}
    func::T1
    args::Vector{SymbolicUtils.BasicSymbolic}
    global_args::Vector{SymbolicUtils.BasicSymbolic}
    fixed_args::Vector{SymbolicUtils.BasicSymbolic}
    varying_args::Vector{SymbolicUtils.BasicSymbolic}
end

function StructuredFunction(func, args, global_args, fixed_args, varying_args)
    type_wanted = SymbolicUtils.BasicSymbolic
    StructuredFunction(
        func,
        convert(Vector{type_wanted}, args),
        convert(Vector{type_wanted}, global_args),
        convert(Vector{type_wanted}, fixed_args),
        convert(Vector{type_wanted}, varying_args),
    )
end

struct HybridModel{T1, T2, T3}
    nn::Lux.Chain{T1}
    func::StructuredFunction{T2}
end

function name_and_type(expr::Expr)
    @assert expr.head == :(::)
    return expr.args[1], expr.args[2]
end
macro hybrid(fun)
    fun_header = fun.args[1]
    fun_args = fun_header.args[2:end]
    global_args = []
    fixed_args = []
    varying_args = []
    all_args = []
    for (i,arg) in enumerate(fun_args)
        name, type = name_and_type(arg)
        if type == :Global
            push!(global_args, name)
        elseif type == :Fixed
            push!(fixed_args, name)
        elseif type == :Varying
            push!(varying_args, name)
        end
        push!(all_args, name)
        fun.args[1].args[1+i] = name
    end
    global_args = Expr(:vect, global_args...)
    fixed_args = Expr(:vect, fixed_args...)
    varying_args = Expr(:vect, varying_args...)
    args_syms = Expr(:vect, all_args...)
    return quote
        StructuredFunction($(esc(fun)), $(esc(args_syms)), $(esc(global_args)), $(esc(fixed_args)), $(esc(varying_args)))
    end
end

export Global, Varying, Fixed, @hybrid, StructuredFunction

# Write your package code here.
end
