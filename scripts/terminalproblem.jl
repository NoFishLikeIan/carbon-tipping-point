using Revise

using Model, Utils

using NLsolve, PreallocationTools
using Plots

using FiniteDiff

using DotEnv, JLD2

include("../src/timederivative.jl")

const economy = Model.Economy();
const hogg = Model.Hogg();
const albedo = Model.Albedo();

const instance = (economy, hogg, albedo);

const domain = [(hogg.T₀, hogg.T̄, 51), (log(economy.Y̲), log(economy.Ȳ), 51)];
const Ω = Utils.makegrid(domain);
const X = Utils.fromgridtoarray(Ω);

Vguess = [ -2f0 + (exp(y) / economy.Y₀)^2 - (T / hogg.Tᵖ)^3 for T ∈ Ω[1], y ∈ Ω[2] ];

const tmp = DiffCache(Array{Float32}(undef, length.(Ω)..., 4));
terminalsteadystate(V̂) = terminalsteadystate!(similar(V̂), V̂);
function terminalsteadystate!(∂ₜV, V̂)
    V = -exp.(V̂)
    vec(terminalG!(∂ₜV, get_tmp(tmp, V), X, V, Ω, instance))
end

V̂₀ = log.(-Vguess);
∂ₜV₀ = terminalsteadystate(V̂₀);

println("Non-linear solver...")
sol = nlsolve(terminalsteadystate!, V̂₀; show_trace = true);

V̄ = -exp.(sol.zero)
∂ₜV̄ = terminalG(X, V̄, Ω, instance)
println("Sup. norm = $(maximum(abs.(∂ₜV̄)))")


println("Saving solution...")
env = DotEnv.config()
DATAPATH = get(env, "DATAPATH", "data/") 
@save joinpath(DATAPATH, "terminal.jld2") V̄  