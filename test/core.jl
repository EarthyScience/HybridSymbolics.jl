using HybridSymbolics, Symbolics, Lux, Random
using Optimisers
using NNlib
using ADTypes
using Zygote


function test_structuredfunc()
    local α, β, γ, δ
    @syms α::Real β::Real γ::Real δ::Real

    structured = @hybrid function testfunc(α::Varying, β::Varying, γ::Fixed=1.0, δ::Global)
        return (exp.(α) .- β)./(γ .* δ)
    end
    @test structured isa PartitionedFunction
    @test length(structured.varying_args) == 2
    @test length(structured.global_args) == 1
    @test length(structured.fixed_args) == 1
    @test length(structured.fixed_vals) == 1
    @test structured.fixed_vals[1] == 1.0f0
    opt_func = structured.opt_func
    @test opt_func(([1.0f0], [1.0f0]), [1.0f0])[1] == exp(1.0f0) - 1.0f0
end

function test_hybridmodel()
    local input_size = 5
    local α, β, γ, δ
    @syms α::Real β::Real γ::Real δ::Real
    structured = @hybrid function testfunc(α::Varying, β::Varying, γ::Fixed=1.0, δ::Global)
        return (exp.(α) .- β)./(γ .* δ)
    end
    NN = Chain(
       Dense(input_size => 4, sigmoid_fast),
       Dense(4 => 2, sigmoid_fast)
    )
    NN = f32(NN)
    rng = MersenneTwister()
    ps, st = Lux.setup(rng, NN)
    model = HybridModel(
        NN,
        structured
    )
    @test model isa HybridModel
    globals = [1.2f0]
    model_params = (nn = ps, globals = globals)
    @test model(rand(Float32, 5), model_params, st) isa Float32
    @test model(rand(Float32, 5,5), model_params, st) isa Vector{Float32}
end

function test_gradcalc()
    local input_size = 5
    local α, β, γ, δ
    @syms α::Real β::Real γ::Real δ::Real
    structured = @hybrid function testfunc(α::Varying, β::Varying, γ::Fixed=1.0, δ::Global)
        return (exp.(α) .- β)./(γ .* δ)
    end
    NN = Chain(
       Dense(input_size => 4, sigmoid_fast),
       Dense(4 => 2, sigmoid_fast)
    )
    NN = f32(NN)
    rng = MersenneTwister()
    ps, st = Lux.setup(rng, NN)
    model = HybridModel(
        NN,
        structured
    )
    globals = [1.2f0]
    model_params = (nn = ps, globals = globals)
    model(rand(Float32, 5), model_params, st)
    model(rand(Float32, 5,5), model_params, st)
    # "Training"
    X = rand(Float32, 5, 5)
    y = rand(Float32, 5)
    loss(X, y, model_params, st) = sum((y .- model(X, model_params, st)) .^ 2)
    grads = gradient((model_params) -> loss(X, y, model_params, st), model_params)[1]
    @test grads isa NamedTuple
    opt = Optimisers.setup(AdamW(), model_params)
    new_params = Optimisers.update(opt, model_params, grads)[1]
    @test new_params != model_params
end

