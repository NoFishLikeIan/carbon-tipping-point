
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
    ϕ(economy.τ, χ, economy) - economy.δₖᵖ - d(T, economy, hogg)
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

"Computes maximum absolute value of the drift of y."
function boundb(t, Xᵢ::Point, model::ModelInstance)
    γₜ = γ(t, model.economy, model.calibration)
    ll = b(t, Xᵢ, 0., 0., model)
    lr = b(t, Xᵢ, 0., γₜ, model)
    rl = b(t, Xᵢ, 1., 0., model)
    rr = b(t, Xᵢ, 1., γₜ, model)

    return max(abs(ll), abs(lr), abs(rl), abs(rr))
end