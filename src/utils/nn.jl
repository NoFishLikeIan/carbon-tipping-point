function constructNN(n::Int, m::Int)::Tuple{Lux.Chain, Function}

    χskip = SkipConnection(
        Chain(
            Dense(n, m),
            Dense(m, m),
            Dense(m, 1, Lux.σ)
        ), 
        (χ, X) -> (χ, Fχ(X, χ))
    ) # X → (χ, Fχ)

    αskip = SkipConnection(
        Chain(
            Dense(n, m),
            Dense(m, m),
            Dense(m, 1, Lux.σ)
        ), 
        (α, X) -> (α, Fα(X, α))
    ) # α → (α, Fα)
    
    controlchain = Parallel(
        (αt, χt, X) -> (
            X, αt[1], χt[1], w(X, αt[1], χt[1]), αt[2], χt[2]
        ),
        αskip, χskip, NoOpLayer() # Passes on X
    ) # X → X, α, χ, w, Fα, Fχ

    valuechain = Chain(
        Dense(n, m), Dense(m, m),
        Dense(m, 1),
        BranchLayer(
            WrappedFunction(x -> -Lux.softplus(x)), 
            WrappedFunction(∂²₁)
        )
    ) # X → (V, ∂²V)

    disjointchain = Parallel(
        (c, v) -> (c[1], c[2], c[3], v[1], v[2], c[4], c[5], c[6]),
        controlchain, valuechain
    ) # X -> X, α, χ, V, ∂²V, w, Fα, Fχ

    NN = Chain(
        disjointchain, 
        SkipConnection(
            WrappedFunction(tup -> ∇V′w(tup[4], tup[5], tup[6], tup[7])),
            (grad, tup) -> (tup[2], tup[3], tup[4], tup[5], grad)
        )
    ) # X -> α, χ, V, ∂²v, ∇V′w


    function L(Θ, st, X, σ²; weights = ones(Float32, 3))
        (_, χ, V, ∂²v, ∇V), st = NN(X, Θ, st)
        
        Y = exp.(X[[4], :])
        χY = χ .* Y

        ∇V[[1], :] += f.(χY, V, Ref(economy)) .+ (σ² / 2f0) .* ∂²v
        ∇V[[3], :] += Y .* ∂f_∂c.(χY, V, Ref(economy)) 

        return mean(abs2, weights'∇V)
    end

    return NN, L

end