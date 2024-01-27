function filename(model::ModelInstance, G::RegularGrid)
    N = size(G, 1)
    Δλ = round(model.albedo.λ₁ - model.albedo.λ₂; digits = 2)

    return filename(N, Δλ, model.preferences)
end

function filename(N, Δλ, preferences::CRRA)
    filename = "N=$(N)_Δλ=$(Δλ)_ρ=$(preferences.ρ)_θ=$(preferences.θ)"

    return "$(replace(filename, "." => ",")).jld2"
end

function filename(N, Δλ, preferences::LogUtility)
    filename = "N=$(N)_Δλ=$(Δλ)_ρ=$(preferences.ρ)"

    return "$(replace(filename, "." => ",")).jld2"
end

function filename(N, Δλ, preferences::EpsteinZin)
    filename = "N=$(N)_Δλ=$(Δλ)_ρ=$(preferences.ρ)_θ=$(preferences.θ)_ψ=$(preferences.ψ)"

    return "$(replace(filename, "." => ",")).jld2"
end