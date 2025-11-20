function Base.show(io::IO, grid::G) where {N₁, N₂, S, G <: AbstractGrid{N₁, N₂, S}}
    gridtype = string(typeof(grid).name.name)
    println(io, "┌─ $(gridtype){$(N₁)×$(N₂), $(S)}")
    Tdomain, mdomain = grid.domains
    # Print domain information
    println(io, "│  Temperature Domain:")
    Tmin, Tmax = Tdomain
    ΔT = Tmax - Tmin
    println(io, "│    ├─ Range: [$(round(Tmin, digits=2)), $(round(Tmax, digits=2))] °C")
    println(io, "│    └─ Step size: $(round(ΔT/(N₁-1), digits=4)) °C")

    # Print carbon domain
    println(io, "│  GHG Domain:")
    mmin, mmax = mdomain
    mrange = mmax - mmin
    println(io, "│    ├─ Range: [$(round(mmin, digits=2)), $(round(mmax, digits=2))]")
    println(io, "│    └─ Step size: $(round(mrange/(N₂-1), digits=4))")
    println(io, "└─────────────────────────────────")
end
function Base.show(io::IO, ::MIME"text/plain", grid::RegularGrid)
    show(io, grid)
end