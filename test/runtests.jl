using HybridSymbolics
using Test

include("./core.jl")
@testset "HybridSymbolics.jl" begin
    @testset test_structuredfunc()
    @testset test_hybridmodel()
    @testset test_gradcalc()
end
