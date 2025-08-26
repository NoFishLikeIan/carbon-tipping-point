function Base.show(io::IO, grid::RegularGrid{N₁,N₂,S,R}) where {N₁,N₂,S,R}
    println(io, "┌─ RegularGrid{$(N₁)×$(N₂), $(S)}")

    # Print domain information
    println(io, "│  Temperature Domain:")
    T_domain = grid.domains[1]
    T_min, T_max = T_domain
    T_range = T_max - T_min
    println(io, "│    ├─ Range: [$(round(T_min, digits=2)), $(round(T_max, digits=2))] °C")
    println(io, "│    └─ Step size: $(round(T_range/(N₁-1), digits=4)) °C")

    # Print carbon domain
    println(io, "│  GHG Domain:")
    m_domain = grid.domains[2]
    m_min, m_max = m_domain
    m_range = m_max - m_min

    # Check if m domain is in log space
    if m_min < 0 && abs(m_min) > 1
        println(io, "│    ├─ Range: [$(round(exp(m_min), digits=2)), $(round(exp(m_max), digits=2))] ppm (log-space)")
        println(io, "│    └─ Step size: $(round(m_range/(N₂-1), digits=4)) (log-space)")
    else
        println(io, "│    ├─ Range: [$(round(m_min, digits=2)), $(round(m_max, digits=2))]")
        println(io, "│    └─ Step size: $(round(m_range/(N₂-1), digits=4))")
    end
    
    # Print example points if available
    if !isempty(grid.X)
        println(io, "│  Example Points:")
        println(io, "│    ├─ Corner (1,1): T=$(round(grid.X[1,1].T, digits=2)), m=$(round(grid.X[1,1].m, digits=2))")
        println(io, "│    ├─ Corner ($(N₁),$(N₂)): T=$(round(grid.X[N₁,N₂].T, digits=2)), m=$(round(grid.X[N₁,N₂].m, digits=2))")
        println(io, "│    └─ Center ($(N₁÷2+1),$(N₂÷2+1)): T=$(round(grid.X[N₁÷2+1,N₂÷2+1].T, digits=2)), m=$(round(grid.X[N₁÷2+1,N₂÷2+1].m, digits=2))")
    end

    println(io, "└─────────────────────────────────")
end

# Add support for display in notebooks and REPLs
function Base.show(io::IO, ::MIME"text/plain", grid::RegularGrid)
    show(io, grid)
end