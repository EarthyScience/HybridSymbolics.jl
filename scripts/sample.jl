using HybridSymbolics, Symbolics, Lux

const input_size = 5
local α, β, γ, δ
@syms α::Real β::Real γ::Real δ::Real

structured = @hybrid function testfunc(α, β, γ, δ)
    return (exp(α) - β)/(γ * δ)
end

NN = Chain(
   Dense(input_size => 4, sigmoid_fast),
   Dense(4 => 3, sigmoid_fast)
)

HybridModel(testfunc(α, β, γ, δ),
            global_vars = [δ],
            input_size = 5,
            nn = NN
)
