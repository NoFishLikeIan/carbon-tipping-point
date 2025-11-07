NpParamaters = Tuple{IAM,Calibration}
"Drift of system in the no-policy scenario."
function Fnp(u::V, parameters::NpParamaters, t) where {R<:Real, V <: StaticVector{2, R}}
    model, calibration = parameters
    T, m = u

    dT = μ(T, m, model.climate) / model.climate.hogg.ϵ
    dm = γ(t, calibration)

    return SVector(dT, dm)
end
function noise(u, parameters::NpParamaters, t)
    model = first(parameters)
    T = u[1]
    return SVector(Model.std(T, model.climate.hogg), 0.)
end


SimulationParameters = Tuple{IAM, Calibration, Interpolations.Extrapolation}
"Drift of system."
function F(u::V, parameters::SimulationParameters, t) where {R<:Real, V <: StaticVector{3, R}}
    model, calibration, αitp = parameters
    T, m = @view u[1:2]
    state = Point(T, m)
    α = αitp(T, m, t)
    χ = χopt(t, model.economy, model.preferences)
    policy = Policy(χ, α)

    dT = μ(T, m, model.climate) / model.climate.hogg.ϵ
    dm = γ(t, calibration) - α
    
    growth = model.economy.investments.ϱ + ϕ(t, policy.χ, model.economy.investments)
    damage = d(state.T, state.m, model.economy.damages, model.climate)
    abatement = A(t, model.economy.investments) * β(t, ε(t, state, α, model, calibration), model.economy.abatement)
    dy = growth - damage - abatement

    return SVector(dT, dm, dy)
end
function noise(u::V, parameters::SimulationParameters, t) where {R<:Real, V <: StaticVector{3, R}}
    model = first(parameters)
    T = u[1]
    return SVector(Model.std(T, model.climate.hogg), 0., model.economy.investments.σₖ)
end
"Drift of system which cumulates abatement, adjustments, and damages."
function F(u::V, parameters::SimulationParameters, t) where {R<:Real, V <: StaticVector{6, R}}
    model, calibration, αitp = parameters
    T, m = @view u[1:2]
    state = Point(T, m)
    α = αitp(T, m, t)
    χ = χopt(t, model.economy, model.preferences)
    policy = Policy(χ, α)

    dT = μ(T, m, model.climate) / model.climate.hogg.ϵ
    dm = γ(t, calibration) - α
    
    growth = model.economy.investments.ϱ + ϕ(t, policy.χ, model.economy.investments)
    damage = d(state.T, state.m, model.economy.damages, model.climate)
    abatement = A(t, model.economy.investments) * β(t, ε(t, state, α, model, calibration), model.economy.abatement)
    dy = growth - damage - abatement

    adjustment = (model.economy.investments.κ / 2) * abatement^2

    return SVector(dT, dm, dy, abatement, adjustment, damage)
end
function noise(u::V, parameters::SimulationParameters, t) where {R<:Real, V <: StaticVector{6, R}}
    model = first(parameters)
    T = u[1]
    return SVector(Model.std(T, model.climate.hogg), 0., model.economy.investments.σₖ, 0.0, 0.0, 0.0)
end

function F!(du, u, parameters::SimulationParameters, t)
    du .= F(u, parameters, t)
end
function noise!(Σ, u, parameters::SimulationParameters, t)
    Σ .= noise(u, parameters, t)
end

# Holds the running simulation, the true model and interpolations, the discovery time, and whether discovery has happened
DiscoveryParameters{S <: SimulationParameters, M <: IAM, α <: Interpolations.Extrapolation, R <: Real} = Tuple{S, M, α, R, Bool}
function F!(du, u, parameters::DiscoveryParameters, t)
    du .= F(u, first(parameters), t)
end
function noise!(Σ, u, parameters::DiscoveryParameters, t)
    Σ .= noise(u, first(parameters), t)
end

"Constructs linear interpolation of results"
function buildinterpolations(values::VS, G::GR) where { N₁, N₂, S, GR <: AbstractGrid{N₁, N₂, S}, VS <: AbstractDict{S, ValueFunction{S, N₁, N₂}} }
    Tspace, mspace = G.ranges
    tspace = collect(keys(values))

    H = Array{S, 3}(undef, N₁, N₂, length(tspace))
    α = similar(H)

    for (i, (t, V)) in enumerate(values)
        H[:, :, i] .= V.H
        α[:, :, i] .= V.α
    end
    
    knots = (Tspace, mspace, tspace)
    Hitp = linear_interpolation(knots, H; extrapolation_bc = Interpolations.Flat())
    αitp = linear_interpolation(knots, α; extrapolation_bc = Interpolations.Flat())

    return Hitp, αitp
end

function timequantiles(M::AbstractMatrix, ps; kwargs...)
    T = size(M, 1)
    qs = Matrix{Float64}(undef, T, length(ps))

    for t in axes(M, 1)
        qs[t, :] .= quantile(M[t, :], ps; kwargs...)
    end

    return qs
end
function smooth!(v, window)
    vo = copy(v)

    for j in eachindex(v)
        l = max(j - window, 1)
        r = min(j + window, length(v))
        v[j] = mean(vo[l:r])
    end
end

"Computes the social cost of carbon at a given point Xᵢ"
function scc(t, Y, Xᵢ::Point, itp, model::IAM)
    Fₘ = FiniteDiff.finite_difference_derivative(m -> itp[:F](Xᵢ.T, m, t), Xᵢ.m)
    Fᵢ = itp[:F](Xᵢ.T, Xᵢ.m, t)
    dm = γ(t, model.calibration) + δₘ(exp(Xᵢ.m), model.climate.hogg) - itp[:α](T, m, t)

    outputfactor = Y / (1 - model.preferences.θ)

    return -outputfactor * (Fₘ / Fᵢ) * Model.Gtonoverppm / dm
end