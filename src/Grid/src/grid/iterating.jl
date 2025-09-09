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

function closestgridpoint(Xᵢ::Point, grid::G) where G <: AbstractGrid
    Tspace, mspace = grid.ranges
    i = argmin(i -> abs(Tspace[i] - Xᵢ.T), axes(grid, 1))
    j = argmin(j -> abs(mspace[j] - Xᵢ.m), axes(grid, 2))
    
    return i, j
end

Base.iterate(grid::G) where G <: AbstractGrid = iterate(grid, 1)
function Base.iterate(grid::G, state::Int) where G <: AbstractGrid    
    if state > length(grid)
        return nothing
    end

    return (grid[state], state + 1)
end

function Base.getindex(grid::G, k::Int) where G <: AbstractGrid
    getindex(grid, CartesianIndex(k, grid))
end

function Base.getindex(grid::G, i, j) where G <: AbstractGrid
    T = grid.ranges[1][i]
    m = grid.ranges[2][j]
    return Point(T, m)
end
function Base.getindex(grid::G, idx::CartesianIndex{2}) where G <: AbstractGrid
    i, j = idx.I
    return getindex(grid, i, j)
end

function Base.getindex(grid::G, I::AbstractVector, J::AbstractVector) where G <: AbstractGrid
    [grid[i, j] for i in I, j in J]
end

function Base.getindex(grid::G, I::AbstractVector, j::Int) where G <: AbstractGrid
    [grid[i, j] for i in I]
end

function Base.getindex(grid::G, i::Int, J::AbstractVector) where G <: AbstractGrid
    [grid[i, j] for j in J]
end

function Base.getindex(grid::G, ::Colon, J::AbstractVector) where G <: AbstractGrid
    [grid[i, j] for i in 1:size(grid, 1), j in J]
end

function Base.getindex(grid::G, I::AbstractVector, ::Colon) where G <: AbstractGrid
    [grid[i, j] for i in I, j in 1:size(grid, 2)]
end

function Base.getindex(grid::G, ::Colon, ::Colon) where G <: AbstractGrid
    [grid[i, j] for i in 1:size(grid, 1), j in 1:size(grid, 2)]
end

struct GridView{G<:AbstractGrid, I, J}
    parent::G
    idxs::I
    jdxs::J
end
function Base.view(grid::G, I, J) where G <: AbstractGrid
    GridView(grid, I, J)
end

# GridView interface
Base.size(gv::GridView) = (length(gv.idxs), length(gv.jdxs))
Base.length(gv::GridView) = prod(size(gv))

function Base.getindex(gv::GridView, i::Int, j::Int)
    ai = gv.idxs[i]
    aj = gv.jdxs[j]
    return gv.parent[ai, aj]
end

function Base.getindex(gv::GridView, k::Int)
    i, j = divrem(k - 1, size(gv, 1))
    return gv[j + 1, i + 1]
end

function Base.iterate(gv::GridView, state=(1, 1))
    i, j = state
    rows, cols = size(gv)
    
    if i > rows
        return nothing
    end
    
    point = gv[i, j]
    nextstate = ifelse(j < cols, (i, j + 1), (i + 1, 1))

    return (point, nextstate)
end

function Base.CartesianIndices(::G) where {N₁, N₂, G <: AbstractGrid{N₁, N₂}}
    CartesianIndices((1:N₁, 1:N₂))
end
