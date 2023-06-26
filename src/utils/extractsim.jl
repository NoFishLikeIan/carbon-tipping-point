function constructinterpolation(nodes, matrix)
    itp = interpolate(nodes, matrix, Gridded(Linear()))
    extp = extrapolate(itp, Line())

    return (x, c, σ, γ) -> extp(x, c, σ, γ)
end

function extractpoliciesfromsim(simulationresults)
    @unpack Γ = first(simulationresults)

    xₗ, xᵤ = extrema(Γ[1])
    cₗ, cᵤ = extrema(Γ[2])

    σspace = [p[2] for p in parameters] |> unique |> sort
    γspace = first.(parameters) |> unique |> sort
    vsizes = unique((size(sim[:V]) for sim in simulationresults))

    maxgridsize = (maximum(first.(vsizes)), maximum(last.(vsizes)))
    intsize = (maxgridsize..., length(σspace), length(γspace))

    Xgrid = range(xₗ, xᵤ; length = maxgridsize[1])
    Cgrid = range(cₗ, cᵤ; length = maxgridsize[2])

    Vₐ = Array{Float64, 4}(undef, intsize...)
    Vconstₐ = similar(Vₐ)
    Eₐ = similar(Vₐ)
    Econstₐ = similar(Vₐ)

    for (i, σₓ) in enumerate(σspace), (j, γ) in enumerate(γspace)
        cidx, uncidx = findall(p -> p[1] == γ && p[2] == σₓ, parameters)

        @unpack Γ, V, E = simulationresults[cidx]

        if size(V) == maxgridsize
            Vₐ[:, :, i, j] .= V
            Eₐ[:, :, i, j] .= E
        else
            v = Spline2D(Γ[1], Γ[2], V)
            e = Spline2D(Γ[1], Γ[2], E)
            
            Vₐ[:, :, i, j] .= v.(Xgrid, Cgrid')
            Eₐ[:, :, i, j] .= e.(Xgrid, Cgrid')
        end

        @unpack Γ, V, E = simulationresults[uncidx]

        if size(V) == maxgridsize
            Vconstₐ[:, :, i, j] .= V
            Econstₐ[:, :, i, j] .= E
        else
            vconst = Spline2D(Γ[1], Γ[2], V)
            econst = Spline2D(Γ[1], Γ[2], E)
            
            Vconstₐ[:, :, i, j] .= vconst.(Xgrid, Cgrid')
            Econstₐ[:, :, i, j] .= econst.(Xgrid, Cgrid')
        end
    end 

    grid = (Xgrid, Cgrid, σspace, γspace)
    v = constructinterpolation(grid, Vₐ)
    e = constructinterpolation(grid, Eₐ)
    vconst = constructinterpolation(grid, Vconstₐ)
    econst = constructinterpolation(grid, Econstₐ)

    return v, e, vconst, econst
end