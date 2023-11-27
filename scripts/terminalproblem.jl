using Revise

using DifferentialEquations: SteadyStateProblem, ODEProblem, solve, init, step!, Tsit5
using PreallocationTools
using DotEnv, JLD2

using Model
using Utils

include("../src/evolution.jl")

const economy = Model.Economy();
const hogg = Model.Hogg(σ²ₜ = 0f0);
const albedo = Model.Albedo();

const instance = (economy, hogg, albedo);

const domain = [
    (hogg.T₀, hogg.T̄, 101), 
    (log(hogg.M̲), log(hogg.M̄), 51), 
    (log(economy.Y̲), log(economy.Ȳ), 51)
];

const grid = RegularGrid(domain);
begin
    const χcache = DiffCache(Array{Float32}(undef, size(grid)))
    const ẏcache = DiffCache(Array{Float32}(undef, size(grid)))
    const ∂V∂Tcache = DiffCache(Array{Float32}(undef, size(grid)))
    const ∂V∂ycache = DiffCache(Array{Float32}(undef, size(grid)))
    const ∂²V∂T²cache = DiffCache(Array{Float32}(undef, size(grid)))
    const parameters = (χcache, ẏcache, ∂V∂Tcache, ∂V∂ycache, ∂²V∂T²cache)
end;

function terminalsteadystate!(∂ₜV, Vₜ, p, t)
    χ = get_tmp(p[1], ∂ₜV)
    ẏ = get_tmp(p[2], ∂ₜV)
    ∂V∂T = get_tmp(p[3], ∂ₜV)
    ∂V∂y = get_tmp(p[4], ∂ₜV)
    ∂²V∂T² = get_tmp(p[5], ∂ₜV)

    terminalG!(∂ₜV, Vₜ, ∂V∂T, ∂V∂y, ∂²V∂T², χ, ẏ, grid, instance)
end

vfunc(T, m, y) = -100f0 + economy.Y₀ * (y / log(economy.Y₀))^2 - Model.d(T, economy, hogg);
V₀ = [ vfunc(T, m, y) for T ∈ grid.Ω[1],  m ∈ grid.Ω[2], y ∈ grid.Ω[3] ];
∂ₜV₀ = Array{Float32}(undef, size(grid));

terminalsteadystate!(∂ₜV₀, V₀, parameters, 0f0);

# Solver - plot
prob = ODEProblem(terminalsteadystate!, V₀, (0f0, 1f0), parameters);

# Analyse
using Plots

function surfaceslice(grid::RegularGrid, F::AbstractArray; idx = 1, plotkwargs...)
    surface(grid.Ω[1], grid.Ω[3], F[:, idx, :]'; 
        c = :viridis, xlabel = "\$T\$", ylabel = "\$y\$", 
        plotkwargs...)
end
function contourfslice(grid::RegularGrid, F::AbstractArray; idx = 1, plotkwargs...)
    contourf(grid.Ω[1], grid.Ω[3], F[:, idx, :]'; 
        c = :viridis, xlabel = "\$T\$", ylabel = "\$y\$", 
        linewidth = 0,
        plotkwargs...)
end

iterator = init(prob, Tsit5())
anim = @animate for iter in 1:10

    step!(iterator)
    surfaceslice(grid, iterator.u)
end

gif(anim, "diff-eq-test.gif", fps = 5)