const maxN = floor(Int, 1 / sqrt(eps(Float64))) # Maximum grid size

struct RegularGrid{N₁, N₂, S <: Real, R <: StepRangeLen{S}}
    domains::NTuple{2,Domain{S}}
    ranges::NTuple{2,R}

    function RegularGrid(N::NTuple{2, Int}, domains::NTuple{2, Domain{S}}) where S <: Real
        ranges = ntuple(i -> range(domains[i][1], domains[i][2]; length = N[i]), Val{2}())
        N₁, N₂ = N

        return new{N₁, N₂, S, typeof(ranges[1])}(domains, ranges)
    end
end

Base.size(::RegularGrid{N₁, N₂}) where {N₁, N₂} = (N₁, N₂)
Base.size(grid::RegularGrid{N₁, N₂}, d) where {N₁, N₂} = size(grid)[d]
Base.axes(grid::RegularGrid, d) = axes(grid.ranges[d], 1)
Base.length(::RegularGrid{N₁, N₂}) where {N₁, N₂} = N₁ * N₂
Base.eltype(::RegularGrid{N₁, N₂, S}) where {N₁, N₂, S} = S

function Base.extrema(grid::RegularGrid)
    ntuple(i -> grid.domains[i][2] - grid.domains[i][1], 2)
end

function Base.step(grid::RegularGrid)
    ntuple(i -> step(grid.ranges[i]), 2)
end

function inversestep(grid::RegularGrid)
    ntuple(i -> inv(step(grid.ranges[i])), 2)
end

function LinearIndex(idx::CartesianIndex{2}, G::RegularGrid)
    LinearIndex(idx.I, G)
end

function LinearIndex((i, j)::NTuple{2, Int}, ::RegularGrid{N₁}) where N₁
    i + (j - 1) * N₁
end

function Base.CartesianIndex(k::Int, ::RegularGrid{N₁}) where N₁
    j, i = divrem(k - 1, N₁)
    return CartesianIndex(i + 1, j + 1)
end

function closestgridpoint(Xᵢ::Point, grid::RegularGrid)
    Tspace, mspace = grid.ranges
    i = argmin(i -> abs(Tspace[i] - Xᵢ.T), axes(grid, 1))
    j = argmin(j -> abs(mspace[j] - Xᵢ.m), axes(grid, 2))
    
    return i, j
end

Base.iterate(grid::RegularGrid) = iterate(grid, 1)
function Base.iterate(grid::RegularGrid, state::Int)
    if state > length(grid) return nothing end
    
    i, j = CartesianIndex(state, grid).I
    
    T = grid.ranges[1][i]
    m = grid.ranges[2][j]
    point = Point(T, m)
    
    return (point, state + 1)
end

function Base.getindex(grid::RegularGrid, k)
    i, j = CartesianIndex(k, grid).I
    getindex(grid, i, j)
end

function Base.getindex(grid::RegularGrid, i, j)
    T = grid.ranges[1][i]
    m = grid.ranges[2][j]
    return Point(T, m)
end

# Slice indexing - returns a matrix of Points
function Base.getindex(grid::RegularGrid, I::AbstractVector, J::AbstractVector)
    [grid[i, j] for i in I, j in J]
end

function Base.getindex(grid::RegularGrid, I::AbstractVector, j::Int)
    [grid[i, j] for i in I]
end

function Base.getindex(grid::RegularGrid, i::Int, J::AbstractVector)
    [grid[i, j] for j in J]
end

# Colon indexing
function Base.getindex(grid::RegularGrid, ::Colon, J::AbstractVector)
    [grid[i, j] for i in 1:size(grid, 1), j in J]
end

function Base.getindex(grid::RegularGrid, I::AbstractVector, ::Colon)
    [grid[i, j] for i in I, j in 1:size(grid, 2)]
end

function Base.getindex(grid::RegularGrid, ::Colon, ::Colon)
    [grid[i, j] for i in 1:size(grid, 1), j in 1:size(grid, 2)]
end

# Support for views
struct GridView{G<:RegularGrid, I, J}
    parent::G
    idxs::I
    jdxs::J
end

function Base.view(grid::RegularGrid, I, J)
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

# Make GridView iterable
function Base.iterate(gv::GridView, state=(1, 1))
    i, j = state
    rows, cols = size(gv)
    
    if i > rows
        return nothing
    end
    
    point = gv[i, j]
    
    # Calculate next state
    if j < cols
        next_state = (i, j + 1)
    else
        next_state = (i + 1, 1)
    end
    
    return (point, next_state)
end