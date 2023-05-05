using DrWatson; @quickactivate "scc-tipping-points"

using Roots
using ApproxFun
using Interpolations
using StaticArrays

using Plots

include("../src/model/climate.jl")
include("../src/model/economic.jl")

include("../src/statecostate/optimalpollution.jl")

m = MendezFarazmand()
l = LinearQuadratic()

@unpack γ, τ = l # Using default parameters
data = wload(
    datadir("manifolds", savename(Dict("γ" => γ, "τ" => τ), "jld2"))
)

nullclines, equilibria = getequilibria(m, l)
@unpack manifolds, tipping_points = data

# Constracting man
û = vcat(manifolds[2][:p], reverse(manifolds[2][:n], dims = 1))[:, 1:2] # x and c

uₗ, uₕ = equilibria[1], equilibria[3]

xₗ, cₗ = uₗ[1:2]
xₕ, cₕ = uₕ[1:2]


sortidx = sortperm(û[:, 2])
ĉ = û[sortidx, 2]
x̂ = û[sortidx, 1]

manifoldinterpolation = linear_interpolation(ĉ, x̂; extrapolation_bc = Interpolations.Line())
separatingplane = c -> manifoldinterpolation(c)

# Differential equations
