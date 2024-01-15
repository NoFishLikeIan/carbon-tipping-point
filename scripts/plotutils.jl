using Plots: surface!, contourf!, wireframe!, plot
using UnPack: @unpack

using Model: ModelInstance

plotsection(F::AbstractArray{Float64, 3}, z, model::ModelInstance; kwargs...) = plotsection!(plot(), F, z, model; kwargs...)
function plotsection!(fig, F::AbstractArray{Float64, 3}, z, model::ModelInstance; zdim = 3, surf = false, labels = ("\$T\$", "\$m\$", "\$y\$"), kwargs...)
    @unpack grid = model

    Nᵢ = size(grid)

    Ω = [range(Δ...; length = Nᵢ[i]) for (i, Δ) in enumerate(grid.domains)]

    ydim, xdim = filter(!=(zdim), 1:3) # Note that x and y are flipped
    jdx = findfirst(x -> x ≥ z, Ω[zdim])

    if isnothing(jdx)
        throw("z = $z ∉ $(Ω[zdim]) for dimension $zdim")
    end

    aspect_ratio = grid.Δ[xdim] / grid.Δ[ydim]

    Z = selectdim(F, zdim, jdx)

    if surf
        wireframe!(fig,
            Ω[xdim], Ω[ydim], Z; 
            aspect_ratio, 
            xlims = grid.domains[xdim], 
            ylims = grid.domains[ydim], 
            xlabel = labels[xdim], ylabel = labels[ydim],
            kwargs...
        )

        surface!(fig,
            Ω[xdim], Ω[ydim], Z; 
            aspect_ratio, 
            xlims = grid.domains[xdim], 
            ylims = grid.domains[ydim], 
            alpha = 0.4,
            kwargs...
        )

    else
        contourf!(fig, 
            Ω[xdim], Ω[ydim], Z; 
            aspect_ratio, 
            xlims = grid.domains[xdim], 
            ylims = grid.domains[ydim], 
            xlabel = labels[xdim], ylabel = labels[ydim],
            kwargs...
        )

    end
end