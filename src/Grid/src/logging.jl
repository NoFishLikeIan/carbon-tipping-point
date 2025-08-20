function Base.show(io::IO, grid::RegularGrid{N, T}) where {N, T}
    println(io, "┌─ RegularGrid{$(N)×$(N), $(T)}")
    
    # Print grid properties
    println(io, "│  Grid Properties:")
    println(io, "│    ├─ Dimensions: $(N)×$(N) ($(N*N) points)")
    println(io, "│    └─ Grid spacing: h = $(round(grid.h, digits=6))")
    
    # Print domain information
    println(io, "│  Temperature Domain:")
    T_domain = grid.domains[1]
    T_min, T_max = T_domain
    T_range = T_max - T_min
    println(io, "│    ├─ Range: [$(round(T_min, digits=2)), $(round(T_max, digits=2))] °C")
    println(io, "│    └─ Step size: $(round(T_range/(N-1), digits=4)) °C")
    
    # Print carbon domain
    println(io, "│  Carbon Domain:")
    m_domain = grid.domains[2]
    m_min, m_max = m_domain
    m_range = m_max - m_min
    
    # Check if m domain is in log space
    if m_min < 0 && abs(m_min) > 1
        println(io, "│    ├─ Range: [$(round(exp(m_min), digits=2)), $(round(exp(m_max), digits=2))] ppm (log-space)")
        println(io, "│    └─ Step size: $(round(m_range/(N-1), digits=4)) (log-space)")
    else
        println(io, "│    ├─ Range: [$(round(m_min, digits=2)), $(round(m_max, digits=2))]")
        println(io, "│    └─ Step size: $(round(m_range/(N-1), digits=4))")
    end
    
    # Print example points
    println(io, "│  Example Points:")
    println(io, "│    ├─ Corner (1,1): T=$(round(grid.X[1,1].T, digits=2)), m=$(round(grid.X[1,1].m, digits=2))")
    println(io, "│    ├─ Corner ($(N),$(N)): T=$(round(grid.X[N,N].T, digits=2)), m=$(round(grid.X[N,N].m, digits=2))")
    println(io, "│    └─ Center ($(N÷2+1),$(N÷2+1)): T=$(round(grid.X[N÷2+1,N÷2+1].T, digits=2)), m=$(round(grid.X[N÷2+1,N÷2+1].m, digits=2))")
    
    println(io, "└─────────────────────────────────")
end

# Add support for display in notebooks and REPLs
function Base.show(io::IO, ::MIME"text/plain", grid::RegularGrid)
    show(io, grid)
end