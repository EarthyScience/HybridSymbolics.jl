using HybridSymbolics, Symbolics, Lux, Random

const input_size = 5
local α, β, γ, δ
@syms α::Real β::Real γ::Real δ::Real

structured = @hybrid function testfunc(α::Varying, β::Varying, γ::Fixed=1.0, δ::Global; forcings)
    T = forcings[1,:]
    return (exp.(α) .- β)./(γ .* δ) .+ T
end

# TODO, define first symbolic function and then apply macro?
# g(α, β, γ, δ) = (exp.(α) .- β)./(γ .* δ)
# @hybrid g # ? do we even need the macro if we type annotate the input args, these are only used later to split args in the HybridModel. Needs thought!


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

ps, st = setup(rng, model)
globals = [1.2f0]
model_params = (nn = ps, globals = globals)

model(rand(Float32, 5), model_params, st)
model(rand(Float32, 5,5), model_params, st)

# "Training"
using Optimisers
using NNlib
using ADTypes
using Zygote

X = rand(Float32, 5, 5)
y = rand(Float32, 5)

loss(X, y, model_params, st) = sum((y .- model(X, model_params, st)) .^ 2)

grads = gradient((model_params) -> loss(X, y, model_params, st), model_params)[1]
opt = Optimisers.setup(AdamW(), model_params)
Optimisers.update!(opt, model_params, grads)

