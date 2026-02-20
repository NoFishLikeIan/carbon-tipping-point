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

function Base.show(io::IO, grid::ParameterGrid{N, S}) where {N, S}
    gridtype = string(typeof(grid).name.name)
    println(io, "┌─ $(gridtype){$(N), $(S)}")
    pmin, pmax = grid.domain
    prange = pmax - pmin
    println(io, "│  Parameter Domain:")
    println(io, "│    ├─ Range: [$(round(pmin, digits=4)), $(round(pmax, digits=4))]")
    println(io, "│    └─ Step size: $(round(prange/(N-1), digits=6))")
    println(io, "└─────────────────────────────────")
end

function Base.show(io::IO, ::MIME"text/plain", grid::ParameterGrid)
    show(io, grid)
end

function Base.show(io::IO, grid::RegretGrid)
    gridtype = string(typeof(grid).name.name)
    stategrid = grid.state
    pgrid = grid.parameter

    N₁ = length(stategrid.ranges[1])
    N₂ = length(stategrid.ranges[2])
    Np = length(pgrid.range)
    S = eltype(stategrid.ranges[1])
    P = eltype(pgrid.range)

    println(io, "┌─ $(gridtype){state: $(N₁)×$(N₂), parameter: $(Np)}")

    Tdomain, mdomain = stategrid.domains
    Tmin, Tmax = Tdomain
    mmin, mmax = mdomain
    pmin, pmax = pgrid.domain

    println(io, "│  State Domain:")
    println(io, "│    ├─ Type: $(S)")
    println(io, "│    ├─ Temperature: [$(round(Tmin, digits=2)), $(round(Tmax, digits=2))] °C")
    println(io, "│    └─ GHG: [$(round(mmin, digits=2)), $(round(mmax, digits=2))]")

    println(io, "│  Parameter Domain:")
    println(io, "│    ├─ Type: $(P)")
    println(io, "│    ├─ Range: [$(round(pmin, digits=4)), $(round(pmax, digits=4))]")
    println(io, "│    └─ Step size: $(round((pmax - pmin)/(Np-1), digits=6))")
    println(io, "└─────────────────────────────────")
end

function Base.show(io::IO, ::MIME"text/plain", grid::RegretGrid)
    show(io, grid)
end