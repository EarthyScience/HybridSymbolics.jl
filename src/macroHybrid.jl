
export @hybrid
function name_and_type(expr::Expr)
    @assert expr.head == :(::)
    return expr.args[1], expr.args[2]
end

function parse_args(fun_args, fun)
    global_args = []
    fixed_args = []
    varying_args = []
    # Perhaps a NamedTuple or a Dict is better
    fixed_vals = []
    all_args = []
    for (i,arg) in enumerate(fun_args)
        if arg.head == :(::)
            name, type = name_and_type(arg)
            if type == :Global
                push!(global_args, name)
            elseif type == :Varying
                push!(varying_args, name)
            end
            push!(all_args, name)
            fun.args[1].args[1+i] = name
        elseif arg.head == :kw
            name = arg.args[1].args[1]
            val = arg.args[2]
            push!(fixed_args, name)
            push!(fixed_vals, val)
            fun.args[1].args[1+i] = name
        end
    end
    return global_args, fixed_args, varying_args, fixed_vals, all_args
end

function substitute_vars(expr, subs)
    if expr isa Symbol && haskey(subs, expr)
        return subs[expr]
    elseif expr isa Expr
        new_args = [substitute_vars(arg, subs) for arg in expr.args]
        return Expr(expr.head, new_args...)
    else
        return expr
    end
end

function optimize_func(fun, global_args, fixed_args, varying_args, fixed_vals)
    func_expr = copy(fun)
    func_body = func_expr.args[2]
    func_args = Expr(:tuple)
    varying_tuple = Expr(:(::), :varying_params, Expr(:curly, :Tuple, [:(Vector{Float32}) for _ in varying_args]...))
    push!(func_args.args, varying_tuple)
    global_array = Expr(:(::), :global_params, :(Vector{Float32}))
    push!(func_args.args, global_array)
    substitutions = Dict()
    for (arg, val) in zip(fixed_args, fixed_vals)
        substitutions[arg] = Float32(val)
    end
    for (i, arg) in enumerate(varying_args)
        substitutions[arg] = :(varying_params[$i])
    end
    for (i, arg) in enumerate(global_args)
        substitutions[arg] = :(global_params[$i])
    end
    new_body = substitute_vars(func_body, substitutions)   
    new_func = Expr(:->,
        func_args,
        new_body
    )
    return new_func
end

# TODO : we should define first the symbolic function, and then apply the macro
macro hybrid(fun)
    fun_header = fun.args[1]
    fun_args = fun_header.args[2:end]
    global_args, fixed_args, varying_args, fixed_vals, all_args = parse_args(fun_args, fun)
    opt_func = optimize_func(fun, global_args, fixed_args, varying_args, fixed_vals)
    global_args = Expr(:vect, global_args...)
    fixed_args = Expr(:vect, fixed_args...)
    varying_args = Expr(:vect, varying_args...)
    args_syms = Expr(:vect, all_args...)
    return quote
        PartitionedFunction($(esc(fun)), $(esc(args_syms)), $(esc(global_args)), $(esc(fixed_args)), $(esc(varying_args)), $(fixed_vals), $(esc(opt_func)))
    end
end