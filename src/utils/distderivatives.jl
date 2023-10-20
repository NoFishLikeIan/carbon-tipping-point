function dir∇(V::SharedFieldGrid, w, grid)::SharedVectorGrid
    D = SharedArray{Float32, length(grid) + 1}((length.(grid)..., length(grid) + 1))
    dir∇!(D, V, w, grid)
    return D
end

function central∇(V::SharedFieldGrid, grid)::SharedVectorGrid
    D = SharedArray{Float32, 4}((length.(grid)..., length(grid)))
    central∇!(D, V, grid)
    return D
end

function ∂²(V::SharedFieldGrid, grid; dim = 1)::SharedFieldGrid
    D² = SharedArray{Float32, 3}(size(V))
    ∂²!(D², V, grid; dim = dim)
    return D²
end