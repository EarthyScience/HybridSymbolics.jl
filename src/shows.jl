function Base.show(io::IO,  pf::PartitionedFunction)
    println(io, "PartitionedFunction with:")
    println(io, "  Function: ", pf.func) # TODO: print the symbolic representation
    if !isempty(pf.args) 
        print(io, "  Args: ", join(pf.args, ", "))
        printstyled(io, " ::$(typeof(pf.args))\n"; color=:light_black)
    end
    !isempty(pf.global_args) && println(io, "  Global args: ", join(pf.global_args, ", "))
    !isempty(pf.fixed_args) && println(io, "  Fixed args: ", join(pf.fixed_args, ", "))
    !isempty(pf.varying_args) && println(io, "  Varying args: ", join(pf.varying_args, ", "))
    if !isempty(pf.fixed_vals)
        print(io, "  Fixed values: ", join(pf.fixed_vals, ", "))
        printstyled(io, " ::$(typeof(pf.fixed_vals))\n"; color=:light_black)
    end
    println(io, "  Optimizer: ", pf.opt_func)
end

function Base.show(io::IO, hm::HybridModel)
    printstyled(io, "HybridModel with:\n"; color=:light_blue)
    print(io, " ")
    printstyled(io, "Neural Network:\n"; color=:cyan)
    print(io, "  ")
    show(io, MIME"text/plain"(), hm.nn)
    print(io, " ")
    printstyled(io, "\nHybrid Function:\n"; color=:cyan)
    print(io, " ")
    show(io, MIME"text/plain"(), hm.func)
end
