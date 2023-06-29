function constructinterpolation(nodes, matrix)
    itp = interpolate(nodes, matrix, Gridded(Linear()))
    extp = extrapolate(itp, Line())

    return (x, c, σ) -> extp(x, c, σ)
end

function extractpoliciesfromsim(simulationresults)
    @unpack Γ = first(simulationresults)

    xₗ, xᵤ = extrema(Γ[1])
    cₗ, cᵤ = extrema(Γ[2])

    σspace = [sim[:climate].σ²ₓ for sim in simulationresults] |> unique |> sort
    vsizes = unique((size(sim[:V]) for sim in simulationresults))

    maxgridsize = (maximum(first.(vsizes)), maximum(last.(vsizes)))
    intsize = (maxgridsize..., length(σspace))

    Xgrid = range(xₗ, xᵤ; length = maxgridsize[1])
    Cgrid = range(cₗ, cᵤ; length = maxgridsize[2])

    Va = Array{Float64, 3}(undef, intsize...)
    Ea = similar(Va)

    for (i, σₓ) in enumerate(σspace)
        cidx = findfirst(sim -> sim[:climate].σ²ₓ == σₓ, simulationresults)

        @unpack Γ, V, E = simulationresults[cidx]

        if size(V) == maxgridsize
            Va[:, :, i] .= V
            Ea[:, :, i] .= E
        else
            v = Spline2D(Γ[1], Γ[2], V)
            e = Spline2D(Γ[1], Γ[2], E)
            
            Va[:, :, i] .= v.(Xgrid, Cgrid')
            Ea[:, :, i] .= e.(Xgrid, Cgrid')
        end
    end 

    grid = (Xgrid, Cgrid, σspace)
    v = constructinterpolation(grid, Va)
    e = constructinterpolation(grid, Ea)

    return v, e
end

function extractecondatafromsim(ensemblesim, e, climate::MendezFarazmand, economy::Ramsey)

    consumption(e, x) = c(e ,x, economy)

    nsimulations = length(ensemblesim)
    timepath = first(ensemblesim).t
    
    data = Array{Float64}(undef, nsimulations, length(timepath), 2)

    for (i, sim) in enumerate(ensemblesim)
        x = first.(sim.u)
        
        data[i, :, 1] = [e(x, m, climate.σ²ₓ) for (x, m) in sim.u]
        data[i, :, 2] = consumption.(data[i, :, 1], x)
    end

    return data
end