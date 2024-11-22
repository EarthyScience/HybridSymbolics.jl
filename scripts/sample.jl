using HybridSymbolics, Symbolics, Lux, Random

const input_size = 5
local α, β, γ, δ
@syms α::Real β::Real γ::Real δ::Real

structured = @hybrid function testfunc(α::Varying, β::Varying, γ::Fixed=1.0, δ::Global)
    return (exp(α) - β)/(γ * δ)
end

NN = Chain(
   Dense(input_size => 4, sigmoid_fast),
   Dense(4 => 3, sigmoid_fast)
)

NN = f32(NN)
rng = MersenneTwister()
ps, st = Lux.setup(rng, NN)

model = HybridModel(
    NN,
    structured
)

model(rand(5), (ps, [1.2f0]), st)
