using DrWatson; @quickactivate "scc-tipping-points"

using Plots
default(size = 600 .* (√2, 1), dpi = 300, margins = 5Plots.mm, linewidth = 1.5)

include("../src/utils/plotting.jl")

include("../src/model/climate.jl")
include("../src/model/economic.jl")

include("../src/statecostate/optimalpollution.jl")


params = Dict("γ" => 7.51443e-4, "τ" => 0.0)
filename = savename(params, "jld2")

@unpack γ, τ = params

m = MendezFarazmand() # Climate model
l = LinearQuadratic(τ = τ, γ = γ, xₛ = m.xₚ) # Economic model

nullcline, equilibria = getequilibria(m, l)
ψ, ω, ϕ = nullcline

wload(datadir("manifolds", filename))

@unpack c₀, x₀ = m 

# -- (x, c)
xspace = range(l.xₛ - 2, 299; length = 2001)
csteadystate = (x -> φ(x, m)).(xspace)

xticks = (280.5:5:295.5, (280:5:300) .- 273.5)
aspect_ratio = (maximum(csteadystate) - minimum(csteadystate)) / (xspace[end] - xspace[1])

xcfig = plot(
    xlims = extrema(csteadystate), ylims = extrema(xspace), 
    aspect_ratio = aspect_ratio,
    xlabel = "Carbon concentration \$c\$ in p.p.m.", 
    ylabel = "Temperature \$x\$ in °C",
    yticks = xticks
)

plot!(xcfig, csteadystate, xspace; c = :darkred, label = false)
scatter!(xcfig, [c₀], [x₀]; c = :black, label = "Initial state")


# -- (e, x)
espace = range(-30, 45; length = 1001)

xefig = plot(
    xlims = extrema(espace), ylims = extrema(xspace), 
    aspect_ratio = (espace[end] - espace[1]) / (xspace[end] - xspace[1]) ,
    ylabel = "Temperature \$x\$ in °C", xlabel = "Emissions \$e\$",
    yicks = xticks
    
)

hline!(xefig, [x₀]; c = :black, label = "Initial state", linestyle = :dash)


# -- (x, λ)	
λspace = range(-650, 0; length = 1001)
xλfig = plot(
    xlims = extrema(xspace), ylims = extrema(λspace), 
    aspect_ratio = (xspace[end] - xspace[1]) / (λspace[end] - λspace[1]),
    ylabel = "Shadow price \$\\lambda_x\$", xlabel = "Temperature \$x\$",
    xticks = xticks
)

plot!(xλfig, xspace, ω; c = :darkred, label = nothing)
vline!(xλfig, [x₀]; c = :black, linestyle = :dash, label = "Initial state")
vline!(xλfig, tipping_points, c = :black, label = "Tipping points")

# -- (c, e)
cspace = range(100, 1000, length = 1001)
cefig = plot(
    ylims = extrema(espace), xlims = extrema(cspace), 
    aspect_ratio = (cspace[end] - cspace[1]) / (espace[end] - espace[1]),
    xlabel = "Carbon concentration \$c\$ in p.p.m.", 
    ylabel = "Emissions \$e\$"
)

plot!(cefig, cspace, c -> c * m.δ; c = :darkred, label = false)
vline!(cefig, [c₀]; c = :black, linestyle = :dash, label = "Initial state")


# Manifolds and steady states
figures = [xcfig, xefig, xλfig, cefig]

colors = [:darkgreen, :darkorange, :darkblue]

for (i, ū) ∈ enumerate(equilibria)
    x, c, λ, e = ū
    
    stablemanifolds = manifolds[i]
    
    # -- (x, c)
    for (dir, curve) ∈ stablemanifolds
        plot!(xcfig, curve[:, 2], curve[:, 1]; c = colors[i], label = nothing)
    end
    scatter!(xcfig, [c], [x]; c = colors[i], label = nothing)

    # -- (x, e)
    for (dir, curve) ∈ stablemanifolds
        plot!(xefig, curve[:, 4], curve[:, 1]; c = colors[i], label = nothing)
    end
    scatter!(xefig, [e], [x]; c = colors[i], label = nothing)
    
    # -- (x, λ)
    for (dir, curve) ∈ stablemanifolds
        plot!(xλfig, curve[:, 1], curve[:, 3]; c = colors[i], label = nothing)
    end
    scatter!(xλfig, [x], [λ]; c = colors[i], label = nothing)
    
    # -- (c, e)
    for (dir, curve) ∈ stablemanifolds
        plot!(cefig, curve[:, 2], curve[:, 4]; c = colors[i], label = nothing)
    end
    scatter!(cefig, [c], [e]; c = colors[i], label = nothing)
end

jointfig = plot(figures..., layout = (2, 2), size = (1200, 1200))

plotfilename = savename("state-costate-manifolds", params, "png")
savefig(jointfig, joinpath(plotsdir(), plotfilename))