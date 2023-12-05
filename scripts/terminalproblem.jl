using DrWatson; @quickactivate @__DIR__;

include("../src/evolution.jl")

using Model
using JLD2: @load

function computeterminal(N::Int, Δλ = 0.08; iterkwargs...)
    @load joinpath(datadir(), "calibration.jld2") calibration 
    hogg = Hogg()
    economy = Economy()
    albedo = Albedo(λ₂ = Albedo().λ₁ - Δλ)

    domains = [
        (hogg.Tᵖ, hogg.Tᵖ + 10.), 
        (log(hogg.M₀), Model.mstable(hogg.Tᵖ + 10., hogg, albedo)), 
        (log(economy.Y₀ / 2), log(economy.Y₀ * 2))
    ]

    model = ModelInstance(
        economy = economy, hogg = hogg, albedo = albedo,
        grid = RegularGrid(domains, N),
        calibration = calibration
    )

    params = @ntuple Δλ N
    V₀ = -model.grid.h * ones(size(model.grid))

    V̄, policy, model = terminaliteration(V₀, model; iterkwargs...)

    savepath = datadir("terminal", savename(params, "jld2"))
    println("\nSaving solution into $savepath...")

    h = [model.grid.h]
    d = Iterators.flatten(domains) |> collect

    data = @dict V̄ policy d h
    wsave(savepath, data)
end