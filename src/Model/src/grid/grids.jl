Domain = Tuple{Float32, Float32};

struct RegularGrid{T}
    scales::NTuple{T, Float32}
    h::Float32

    function RegularGrid(domains::AbstractArray{Domain}, h::Float32)
        N = length(domains)
        scales = ntuple(i -> (domains[i][2] - domains[i][1]) * h, N)

        new{N}(scales, h)
    end
end

dimensions(grid::RegularGrid) = length(grid.scales)
Base.size(grid::RegularGrid{N}) where N = ntuple(_ -> length(0f0:grid.h:1f0), Val(N))
Base.CartesianIndices(grid::RegularGrid) = CartesianIndices(size(grid))

