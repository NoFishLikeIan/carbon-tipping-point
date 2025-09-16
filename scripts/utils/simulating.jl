function b(t, state::Point, policy::Policy, model::IAM, calibration::Calibration)
    @unpack economy, climate = model
    @unpack abatement, damages, investments = economy

    growth = investments.ϱ + ϕ(t, policy.χ, investments) 
end

SimulationParameters = Tuple{IAM, Calibration, Interpolations.Extrapolation}
"Drift of system."
function F(u::SVector{3,R}, parameters::SimulationParameters, t) where R<:Real
    model, calibration, αitp = parameters
    T, m = @view u[1:2]
    state = Point(T, m)
    α = αitp(T, m, t)
    χ = χopt(t, model.economy, model.preferences)
    policy = Policy(χ, α)

    dT = μ(T, m, model.climate) / model.climate.hogg.ϵ
    dm = γ(t, calibration) - α
    
    growth = investments.ϱ + ϕ(t, policy.χ, model.economy.investments)
    damage = d(state.T, state.m, model.economy.damages, model.climate)
    abatement = A(t, model.economy.investments) * β(t, ε(t, state, α, model, calibration), model.economy.abatement)
    dy = growth - damage - abatement

    return SVector(dT, dm, dy)
end
"Drift of system which cumulates abatement, adjustments, and damages."
function F(u::SVector{6,R}, parameters::SimulationParameters, t) where R<:Real
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

NpParamaters = Tuple{IAM,Calibration}
"Drift of system in the no-policy scenario."
function Fnp(u::SVector{2,R}, parameters::NpParamaters, t) where R<:Real
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
function noise(u::SVector{3}, parameters::SimulationParameters, t)
    model = first(parameters)
    T = u[1]
    return SVector(Model.std(T, model.climate.hogg), 0., model.economy.investments.σₖ)
end
function noise(u::SVector{6}, parameters::SimulationParameters, t)
    model = first(parameters)
    T = u[1]
    return SVector(Model.std(T, model.climate.hogg), 0., model.economy.investments.σₖ, 0.0, 0.0, 0.0)
end

"Constructs linear interpolation of results"
function buildinterpolations(values::VS, G::GR) where { N₁, N₂, S, GR <: AbstractGrid{N₁, N₂, S}, VS <: AbstractDict{S, ValueFunction{S, N₁, N₂}} }
    Tspace, mspace = G.ranges
    
    times = diff(collect(keys(values)))
    timestep = only(unique(times))
    t₀, t₁ = extrema(keys(values))
    tspace = t₀:timestep:t₁
    
    nodes = (Tspace, mspace, tspace)

    H = Array{S, 3}(undef, N₁, N₂, length(tspace))
    α = similar(H)

    for (i, (t, V)) in enumerate(values)
        H[:, :, i] .= V.H
        α[:, :, i] .= V.α
    end

    Hitp = extrapolate(scale(interpolate(H, BSpline(Linear())), nodes...), Flat())
    αitp = extrapolate(scale(interpolate(α, BSpline(Linear())), nodes...), Flat())

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

sampletemperature(model::IAM, trajectories) = sampletemperature(default_rng(), model, trajectories)
function sampletemperature(rng, model::IAM, trajectories; σ=0.15)
    T̄ = minimum(Model.Tstable(log(model.climate.hogg.M₀), model))
    z = randn(rng, trajectories)

    return @. T̄ + z * σ
end