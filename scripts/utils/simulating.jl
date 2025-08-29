struct TimeStateInterpolation{S, I <: Interpolations.Extrapolation{S}}
    itp::I
end

function apply(t, x::Point, itp::TimeStateInterpolation)
    itp.itp(x.T, x.m, t)
end

function costbreakdown(t, state, policy, model, calibration)
    ε = policy.α / ᾱ(t, state, model, calibration)
    abatement = β(t, ε, model.economy)
    adjustment = model.economy.κ * abatement^2 / 2
    damages = d(state.T, state.m, model)

    return abatement, adjustment, damages
end

SimulationParameters = Tuple{AbstractModel, Calibration, TimeStateInterpolation}
"Drift of system."
function F(u::SVector{3,R}, parameters::SimulationParameters, t) where R<:Real
    model, calibration, αitp = parameters
    T, m = @view u[1:2]
    state = Point(T, m)
    α = apply(t, state, αitp)
    χ = χopt(t, model.economy, model.preferences)
    policy = Policy(χ, α)

    dT = μ(T, m, model) / model.hogg.ϵ
    dm = γ(t, calibration) - α
    dy = b(t, state, policy, model, calibration)

    return SVector(dT, dm, dy)
end
"Drift of system which cumulates abatement, adjustments, and damages."
function F(u::SVector{6,R}, parameters::SimulationParameters, t) where R<:Real
    model, calibration, αitp = parameters
    T, m = @view u[1:2]
    state = Point(T, m)
    α = apply(t, state, αitp)
    χ = χopt(t, model.economy, model.preferences)
    policy = Policy(χ, α)

    dT = μ(T, m, model) / model.hogg.ϵ
    dm = γ(t, calibration) - α
    dy = b(t, state, policy, model, calibration)

    # Cost breakdown
    abatement, adjustment, damages = costbreakdown(t, state, policy, model, calibration)

    return SVector(dT, dm, dy, abatement, adjustment, damages)
end

NpParamaters = Tuple{AbstractModel,Calibration}
"Drift of system in the no-policy scenario."
function Fnp(u::SVector{2,R}, parameters::NpParamaters, t) where R<:Real
    model, calibration = parameters
    T, m = u

    dT = μ(T, m, model) / model.hogg.ϵ
    dm = γ(t, calibration)

    return SVector(dT, dm)
end

NpGameParameters = Tuple{NTuple{2,AbstractModel},Calibration}
"Drift of game system without policies."
function Fnp(u::SVector{3,R}, parameters::NpGameParameters, t) where R<:Real
    models, calibration = parameters
    oecdmodel, rowmodel = models
    T₁, T₂, m = u

    dT₁ = μ(T₁, m, oecdmodel) / oecdmodel.hogg.ϵ
    dT₂ = μ(T₂, m, rowmodel) / rowmodel.hogg.ϵ
    dm = γ(t, calibration)

    return SVector(dT₁, dT₂, dm)
end

function noise(u, parameters::NpParamaters, t)
    model = first(parameters)
    σT = model.hogg.σₜ / model.hogg.ϵ
    σm = model.hogg.σₘ
    return SVector(σT, σm)
end
function noise(u::SVector{3}, parameters::SimulationParameters, t)
    model = first(parameters)
    σT = model.hogg.σₜ / model.hogg.ϵ
    σm = model.hogg.σₘ
    σk = model.economy.σₖ
    return SVector(σT, σm, σk)
end
function noise(u::SVector{6}, parameters::SimulationParameters, t)
    model = first(parameters)
    σT = model.hogg.σₜ / model.hogg.ϵ
    σm = model.hogg.σₘ
    σk = model.economy.σₖ
    return SVector(σT, σm, σk, 0.0, 0.0, 0.0)
end
function noise(u, parameters::NpGameParameters, t)
    oecdmodel, rowmodel = first(parameters)
    σT₁ = oecdmodel.hogg.σₜ / oecdmodel.hogg.ϵ
    σT₂ = rowmodel.hogg.σₜ / rowmodel.hogg.ϵ
    σm = oecdmodel.hogg.σₘ
    return SVector(σT₁, σT₂, σm)
end

rate(u, parameters::Tuple, t) = rate(u, first(parameters), t)
rate(u, model::JumpModel, t) = intensity(u[1], model.hogg, model.jump)

function tippingopt!(integrator)
    model = first(integrator.p)
    q = increase(integrator.u[1], model.hogg, model.jump)
    integrator.u[1] += q
end
function tipping!(integrator)
    model = integrator.p
    q = increase(integrator.u[1], model.hogg, model.jump)
    integrator.u[1] += q
end

"Constructs linear interpolation of results"
function buildinterpolations(values, model, G::RegularGrid{N₁, N₂, S}) where {N₁, N₂, S}
    Tdomain, Gdomain = G.domains
    Tspace = range(Tdomain[1], Tdomain[2]; length=size(G, 1))
    mspace = range(Gdomain[1], Gdomain[2]; length=size(G, 2))
    timespace = collect(keys(values))

    nodes = (Tspace, mspace, timespace)
    H = Array{S, 3}(undef, size(G, 1), size(G, 2), length(timespace))
    α = similar(H)

    for (i, pair) in enumerate(values)
        V = pair.second
        H[:, :, i] .= V.H
        α[:, :, i] .= V.α
    end

    Hitp = linear_interpolation(nodes, H; extrapolation_bc=Line()) |> TimeStateInterpolation
    αitp = linear_interpolation(nodes, α; extrapolation_bc=Line()) |> TimeStateInterpolation

    return Hitp, αitp
end

GameResult = Tuple{Vector{Float64},Dict{<:AbstractModel,Array{Float64,3}},Dict{<:AbstractModel,Array{Float64,4}},RegularGrid,Vector{<:AbstractModel}}
function buildinterpolations(result::GameResult)
    throw("Not implemented!")
end

computeonsim(sim::RODESolution, f) = computeonsim!(similar(sim.t), sim, f)
function computeonsim!(y, sim::RODESolution, f)
    for i in eachindex(y)
        t, u = sim.t[i], sim.u[i]

        y[i] = f(t, u)
    end

    return y
end
function computeonsim(sol::EnsembleSolution, f)
    N = length(sol)
    T = length(first(sol).t)
    M = Matrix{eltype(sol)}(undef, T, N)

    for j in axes(M, 2)
        yᵢ = @view M[:, j]
        computeonsim!(yᵢ, sol[j], f)
    end

    return M
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
function scc(t, Y, Xᵢ::Point, itp, model::AbstractModel)
    Fₘ = FiniteDiff.finite_difference_derivative(m -> itp[:F](Xᵢ.T, m, t), Xᵢ.m)
    Fᵢ = itp[:F](Xᵢ.T, Xᵢ.m, t)
    dm = γ(t, model.calibration) + δₘ(exp(Xᵢ.m), model.hogg) - itp[:α](T, m, t)

    outputfactor = Y / (1 - model.preferences.θ)

    return -outputfactor * (Fₘ / Fᵢ) * Model.Gtonoverppm / dm
end

sampletemperature(model::AbstractModel, trajectories) = sampletemperature(default_rng(), model, trajectories)
function sampletemperature(rng, model::AbstractModel, trajectories; σ=0.15)
    T̄ = minimum(Model.Tstable(log(model.hogg.M₀), model))
    z = randn(rng, trajectories)

    return @. T̄ + z * σ
end