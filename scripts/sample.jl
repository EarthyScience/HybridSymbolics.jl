using HybridSymbolics, Symbolics, Lux, Random

const input_size = 5
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

model(rand(Float32, 5), (ps, [1.2f0]), st)
model(rand(Float32, 5,5), (ps, [1.2f0]), st)

# "Training"
using Optimisers
using NNlib
using ADTypes
using Zygote

X = rand(Float32, 5, 5)
y = rand(Float32, 5)

loss(X, y, ps, globals, st) = sum((y .- model(X, (ps, globals), st)) .^ 2)

globals_init = [1.0f0]

grads = gradient((ps, globals) -> loss(X, y, ps, globals, st), ps, globals_init)

opt = Optimisers.setup(AdamW(), (ps, globals_init))

Optimisers.update!(opt, (ps, globals_init), grads)

