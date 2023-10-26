using Revise

using Model, Utils
using DifferentialEquations, NonlinearSolve
using PreallocationTools

using DotEnv, JLD2

include("../src/timederivative.jl")

const economy = Model.Economy();
const hogg = Model.Hogg();
const albedo = Model.Albedo();

const instance = (economy, hogg, albedo);

const domain = [(hogg.T₀, hogg.T̄, 101), (log(economy.Y̲), log(economy.Ȳ), 51)];
const Ω = Utils.makegrid(domain);
const X = Utils.fromgridtoarray(Ω);

const tmpcache = DiffCache(Array{Float32}(undef, length.(Ω)..., 4));
terminalsteadystate(Vₜ, p) = terminalsteadystate!(similar(Vₜ), Vₜ, p, 0f0);
function terminalsteadystate!(∂ₜV, Vₜ, p, t)
    tmp = get_tmp(first(p), Vₜ)
    terminalG!(∂ₜV, tmp, X, Vₜ, Ω, instance)
end

V₀ = [ -2f0 + (exp(y) / economy.Y₀)^2 - (T / hogg.Tᵖ)^3 for T ∈ Ω[1], y ∈ Ω[2] ];
∂ₜV₀ = similar(V₀);

params = (tmpcache, );
prob = ODEProblem(terminalsteadystate!, V₀, (0f0, 20f0), params);
sol = solve(prob);

V̄ = sol[end];
∂ₜV̄ = terminalG(X, V̄, Ω, instance);
println("Sup. norm = $(maximum(abs.(∂ₜV̄)))")

println("Saving solution...")
env = DotEnv.config()
DATAPATH = get(env, "DATAPATH", "data/") 
@save joinpath(DATAPATH, "terminal.jld2") V̄