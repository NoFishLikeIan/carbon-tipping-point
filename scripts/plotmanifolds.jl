using DrWatson

@quickactivate "scc-tipping-points"

using Roots

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

equilibria = computesteadystates(m, l)
tipping_points = find_zeros(x -> μₓ(x, m), (290, 300))

wload(datadir("manifolds", filename))

@unpack c₀, x₀ = m 

begin
    # -- (x, c)
    xspace = range(l.xₛ - 2., 300; length = 2001)
    
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

    colors = [:darkgreen, :darkorange, :darkblue]

    for (i, ū) ∈ enumerate(equilibria[1:2])
        x, c, λ, e = ū
        
        stablemanifolds = manifolds[i]
        
        # -- (x, c)
        for curve ∈ stablemanifolds
            plot!(xcfig, curve[:, 2], curve[:, 1]; c = colors[i], label = nothing)
        end
        scatter!(xcfig, [c], [x]; c = colors[i], label = nothing)

    end
end

plotfilename = savename("xcmanifolds", params, "png")
savefig(xcfig, joinpath(plotsdir(), plotfilename))

xcfig