"Emissivity rate implied by abatement `α` at time `t` and carbon concentration `M`"
function ε(t, M, α, model::ModelInstance)
    1. - M * (δₘ(M, model.hogg) + γ(t, model.economy, model.calibration) - α) / (Gtonoverppm * Eᵇ(t, model.economy, model.calibration))
end
function ε′(t, M, model::ModelInstance)
    M / (Gtonoverppm * Eᵇ(t, model.economy, model.calibration))
end

"Drift of dy in the terminal state, t ≥ τ."
bterminal(Xᵢ::Point, args...) = bterminal(Xᵢ.T, args...)
bterminal(T::Float64, χ, model::ModelInstance) = bterminal(T, χ, model.economy, model.hogg) 
function bterminal(T::Float64, χ, economy::Economy, hogg::Hogg)
    investment = ϕ(economy.τ, χ, economy)
    depreciaton = economy.δₖᵖ
    damage = d(T, economy, hogg)

    investment - depreciaton - damage
end

"Drift of dy."
b(t, Xᵢ::AbstractVector{Float64}, χ::Float64, α::Float64, model::ModelInstance) = b(t, Point(Xᵢ[1], Xᵢ[2], Xᵢ[3]), χ, α, model);
b(t, Xᵢ::Point, χ::Float64, α::Float64, model::ModelInstance) = b(t, Xᵢ, Policy(χ, α), model) 
function b(t, Xᵢ::Point, pᵢ::Policy, model::ModelInstance)
    @unpack economy, hogg = model

    εₜ = ε(t, exp(Xᵢ.m), pᵢ.α, model)
    Aₜ = A(t, economy)

    abatement = Aₜ * β(t, εₜ, economy)

    economy.ϱ + ϕ(t, pᵢ.χ, economy) - economy.δₖᵖ - abatement - d(Xᵢ.T, economy, hogg)
end

"Largest drift on the unit cube."
driftbounds(t, model::ModelInstance) = driftbounds!(similar(Array{Drift}, axes(model.grid)), t, model)
function driftbounds(t, Xᵢ::Point, model::ModelInstance)
    dT = μ(Xᵢ.T, Xᵢ.m, model.hogg, model.albedo) / (model.hogg.ϵ * model.grid.Δ.T)
    dm = γ(t, model.economy, model.calibration) / model.grid.Δ.m

    b₁ = b(t, Xᵢ, 1., 0., model)
    b₀ = b(t, Xᵢ, 0., 0., model)
    dy = ifelse(abs(b₁) > abs(b₀), b₁, b₀) / model.grid.Δ.y

    return Drift(dT, dm, dy)
end
function driftbounds!(maxdrift::AbstractArray{Drift, 3}, t, model::ModelInstance)
    @batch for idx in CartesianIndices(model.grid)
        maxdrift[idx] = driftbounds(t, model.grid.X[idx], model)
    end

    return maxdrift
end