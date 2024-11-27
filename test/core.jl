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
    @test opt_func(([1.0f0], [1.0f0]), [1.0f0]; forcings = nothing)[1] == exp(1.0f0) - 1.0f0
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
    @testset "Total Output" begin
        @test model(rand(Float32, 5), model_params, st; return_parameters = Val(true)) isa Dict
        output = model(rand(Float32, 5), model_params, st; return_parameters = Val(true))
        @testset "Arguments in output" for arg in [:α, :β, :γ, :δ]
            @test arg in keys(output)
        end
        @test :out in keys(output)
        @test output[:γ] == 1.0f0
    end
end

function test_bounds()
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
    model = HybridModel(
        NN,
        structured
    )
    model = setbounds(model, Dict(:α => (-1.0f0, 1.0f0), :β => (-1.0f0, 1.0f0)))
    @test model isa HybridModel
    ps, st = setup(rng, model)
    globals = [1.2f0]
    model_params = (nn = ps, globals = globals)
    @test model(rand(Float32, 5), model_params, st) isa Float32
    @test model(rand(Float32, 5,5), model_params, st) isa Vector{Float32}
    @test all(model.p_min .== -1.0f0)
    @test all(model.p_max .== 1.0f0)
    @testset "Testing Bounds: input $item" for item in [rand(Float32, 5) for _ in 1:10]
        output_params = model.nn(item, ps, st)[1]
        @test all(output_params .>= -1.0f0) && all(output_params .<= 1.0f0)
    end
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

function test_forcings()
    local input_size = 5
    local α, β, γ, δ
    @syms α::Real β::Real γ::Real δ::Real
    structured = @hybrid function testfunc(α::Varying, β::Varying, γ::Fixed=1.0, δ::Global; forcings)
        T = forcings[1,:]
        return (exp.(α) .- β)./(γ .* δ) .+ T
    end
    NN = Chain(
       Dense(input_size => 4, sigmoid_fast),
       Dense(4 => 2, sigmoid_fast)
    )
    NN = f32(NN)
    rng = MersenneTwister()
    model = HybridModel(
        NN,
        structured
    )
    ps, st = setup(rng, model)
    globals = [1.2f0]
    model_params = (nn = ps, globals = globals)
    @test model(rand(Float32, 5), model_params, st; forcings = ones(Float32, 1,1)) isa Float32
    @test model(rand(Float32, 5,5), model_params, st; forcings = ones(Float32, 1,5)) isa Vector{Float32}
    @test all(model(rand(Float32, 5,5), model_params, st; forcings = ones(Float32, 1,5)) .> 1.0f0)
end
