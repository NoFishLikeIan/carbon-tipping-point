function constructNN(n::Int, m::Int)::Tuple{Lux.Chain, Function}

    χchain = Chain(
        Dense(m, m, Lux.tanh), Dense(m, m, Lux.tanh),
        Dense(m, 1, Lux.σ)
    )

    αchain = Chain(
        Dense(m, m, Lux.tanh), Dense(m, m, Lux.tanh),
        Dense(m, 1, Lux.σ)
    )

    valuechain = Chain(
        Dense(m, m, Lux.relu), Dense(m, m, Lux.relu), 
        Dense(m, 1, x -> -Lux.softplus(x))
    )

    NN = Chain(
        Dense(n, m, Lux.tanh), Dense(m, m, Lux.tanh),
        BranchLayer(αchain, χchain, valuechain)
    )

    function L(Θ, st, X, σ²)
        (α, χ, V), st = NN(X, Θ, st)

        X̃ = fromunit(X)

        Y = exp.(X̃[[4], :])
        χY = χ .* Y
        
        ∇V = Buffer(V, (3, size(V, 2)))
        ∇V′μ!(∇V, V, drift(X̃, α, χ), Fα(X̃, α), Fχ(X̃, χ))

        ∇V[[1], :] += f.(χY, V, Ref(economy)) .+ (σ² / 2f0) .* ∂²₁(V)
        ∇V[[3], :] += Y .* ∂f_∂c.(χY, V, Ref(economy)) 

        return mean(abs2, ∇V), (α, χ, V)
    end

    return NN, L

end