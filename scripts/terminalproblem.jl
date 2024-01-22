using Model, Grid

using JLD2, DotEnv
using UnPack: @unpack
using Polyester: @batch
using FastClosures: @closure
using Roots: find_zero

const env = DotEnv.config()
const DATAPATH = get(env, "DATAPATH", "data/")


"Computes the Jacobi iteration for the terminal problem, V̄."
function terminaljacobi!(
    V::AbstractArray{Float64, 3}, 
    policy::AbstractArray{Float64, 3}, 
    model::ModelInstance, G::RegularGrid;
    indices = CartesianIndices(G)) # Can use internal indices to keep boundary constant

    L, R = extrema(CartesianIndices(G))
    
    @batch for idx in indices
        σₜ² = (model.hogg.σₜ / (model.hogg.ϵ * G.Δ.T))^2
        σₖ² = (model.economy.σₖ / G.Δ.y)^2
        Xᵢ = G.X[idx]
        dT = μ(Xᵢ.T, Xᵢ.m, model.hogg, model.albedo) / (model.hogg.ϵ * G.Δ.T)

        # Neighbouring nodes
        # -- GDP
        Vᵢy₊ = V[min(idx + I[3], R)]
        Vᵢy₋ = V[max(idx - I[3], L)]

        # -- Temperature
        VᵢT₊ = V[min(idx + I[1], R)]
        VᵢT₋ = V[max(idx - I[1], L)]

        value = @closure χ -> begin
            dy = bterminal(Xᵢ, χ, model) / G.Δ.y
            Q = σₜ² + σₖ² + G.h * (abs(dT) + abs(dy))

            py₊ = (G.h * max(dy, 0.) + σₖ² / 2) / Q
            py₋ = (G.h * max(-dy, 0.) + σₖ² / 2) / Q

            pT₊ = (G.h * max(dT, 0.) + σₜ² / 2) / Q
            pT₋ = (G.h * max(-dT, 0.) + σₜ² / 2) / Q

            # Expected value
            EV̄ = py₊ * Vᵢy₊ + py₋ * Vᵢy₋ + pT₊ * VᵢT₊ + pT₋ * VᵢT₋

            Δt = G.h^2 / Q

            return f(χ, Xᵢ, EV̄, Δt, model.preferences)
        end

        # Optimal control
        v, χ = gss(value, 0., 1.; tol = 1e-6)
        
        V[idx] = v
        policy[idx] = χ
    end

    return V, policy
end

function terminaliteration(V₀::AbstractArray{Float64, 3}, model::ModelInstance, grid::RegularGrid; tol = 1e-3, maxiter = 100_000, verbose = false, indices = CartesianIndices(grid))
    policy = similar(V₀);

    verbose && println("Starting iterations...")

    Vᵢ, Vᵢ₊₁ = copy(V₀), copy(V₀);
    for iter in 1:maxiter
        terminaljacobi!(Vᵢ₊₁, policy, model, grid; indices = indices)

        ε = maximum(abs.(Vᵢ₊₁ .- Vᵢ))

        verbose && print("Iteration $iter / $maxiter, ε = $ε...\r")

        if ε < tol
            return Vᵢ₊₁, policy
        end
        
        Vᵢ .= Vᵢ₊₁
    end

    verbose && println("\nDone without convergence.")
    return Vᵢ₊₁, policy
end

function computeterminal(N::Int, Δλ; 
    verbose = true, withsave = true, v₀ = -1.,
    iterkwargs...)

    calibration = load_object(joinpath(DATAPATH, "calibration.jld2"))
    hogg = Hogg()
    economy = Economy()
    albedo = Albedo(λ₂ = Albedo().λ₁ - Δλ)
    model = ModelInstance(economy, hogg, albedo, calibration)

    # -- Critical domain
    Tᶜ = find_zero(T -> bterminal(T, 0., economy, hogg), (hogg.Tᵖ, hogg.Tᵖ + 15.))
    
    criticaldomains = [
        (Tᶜ, Tᶜ + 5.),
        (mstable(hogg.Tᵖ, hogg, albedo), mstable(Tᶜ + 5., hogg, albedo)),
        (log(economy.Y₀ / 2), log(3economy.Y₀)),
    ]

    criticalgrid = RegularGrid(criticaldomains, N);
    criticalindices = CartesianIndices(criticalgrid, Dict(3 => (true, false))); # Excludes the values y₀ in the iteration process

    Vᶜ₀ = v₀ .* ones(size(criticalgrid))
    criticalpolicy = similar(Vᶜ₀);

    verbose && println("Solving critical region T > Tᶜ = $Tᶜ...")
    Vᶜ, criticalpolicy = terminaliteration(Vᶜ₀, model, criticalgrid; indices = criticalindices, verbose, iterkwargs...)

    # -- Regular domain
    domains = [
        (hogg.T₀, Tᶜ), 
        (mstable(hogg.T₀ - 0.5, hogg, albedo), mstable(Tᶜ + 0.5, hogg, albedo) + 1),
        (log(economy.Y₀ / 2), log(2economy.Y₀)), 
    ]
    
    grid = RegularGrid(domains, N);
    
    V₀ = min.(interpolateovergrid(criticalgrid, Vᶜ, grid), 0.)

    indices = permutedims(
        reverse(CartesianIndices(criticalgrid, Dict(1 => (false, true)))),
        (3, 2, 1)
    )

    verbose && println("Solving inside region of interest...")
    V̄, policy = terminaliteration(V₀, model, grid; indices, verbose, iterkwargs...)

    if withsave
        savepath = joinpath(DATAPATH, "terminal", "N=$(N)_Δλ=$(Δλ).jld2")
        println("\nSaving solution into $savepath...")

        jldsave(savepath; V̄, policy, model, grid)
    end

    return V̄, policy, model
end