Base.size(::G) where {N₁, N₂, G <: AbstractGrid{N₁, N₂}} = (N₁, N₂)
Base.size(grid::G, d) where {N₁, N₂, G <: AbstractGrid{N₁, N₂}} = size(grid)[d]
Base.axes(grid::G, d) where G <: AbstractGrid = axes(grid.ranges[d], 1)
Base.length(::G) where {N₁, N₂, G <: AbstractGrid{N₁, N₂}} = N₁ * N₂
Base.eltype(::G) where {N₁, N₂, S, G <: AbstractGrid{N₁, N₂, S}} = S
Base.IteratorSize(::G) where G <: AbstractGrid = Base.HasShape{2}()

function Base.extrema(grid::G) where G <: AbstractGrid
    ntuple(i -> grid.domains[i][2] - grid.domains[i][1], 2)
end

function LinearIndex(idx::CartesianIndex{2}, grid::G) where {G <: AbstractGrid}
    LinearIndex(idx.I, grid)
end

function LinearIndex((i, j)::NTuple{2, Int}, ::G) where {N₁, G <: AbstractGrid{N₁}}
    i + (j - 1) * N₁
end

function Base.CartesianIndex(k::Int, ::G) where {N₁, G <: AbstractGrid{N₁}}
    j, i = divrem(k - 1, N₁)
    return CartesianIndex(i + 1, j + 1)
end

function closestgridpoint(x::Point, grid::G) where G <: AbstractGrid
    Tspace, mspace = grid.ranges
    i = argmin(i -> abs(Tspace[i] - x.T), axes(grid, 1))
    j = argmin(j -> abs(mspace[j] - x.m), axes(grid, 2))
    
    return i, j
end

function Base.CartesianIndices(::G) where {N₁, N₂, G <: AbstractGrid{N₁, N₂}}
    CartesianIndices((1:N₁, 1:N₂))
end
