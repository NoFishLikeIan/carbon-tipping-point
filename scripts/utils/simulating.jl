using Model, Grid
using Dierckx: Spline2D
using Interpolations
using FiniteDiff
using DifferentialEquations
using Statistics
using Random: default_rng

PolicyFunction = Union{Interpolations.Extrapolation, Function};
PoliciesFunctions = NTuple{2, PolicyFunction};

GameParameters = Tuple{Tuple{M1, M2}, NTuple{2, PoliciesFunctions}, Calibration} where {M1 <: AbstractModel, M2 <: AbstractModel}

function Fgame!(du, u, parameters::GameParameters, t)
    models, policies, calibration = parameters
    oecdmodel, rowmodel = models
    oecdpolicies, rowpolicies = policies

    T₁, T₂, m = @view u[1:3]
    χ₁ = oecdpolicies[1](T₁, m, t)
    χ₂ = rowpolicies[1](T₂, m, t)

    α₁ = oecdpolicies[2](T₁, m, t)
    α₂ = rowpolicies[2](T₂, m, t)

    # Temperature
    du[1] = μ(T₁, m, oecdmodel) / oecdmodel.hogg.ϵ
    du[2] = μ(T₂, m, rowmodel) / rowmodel.hogg.ϵ
    
    # Carbon concentration
    du[3] = γ(t, calibration) - α₁ - α₂

    # Output
    du[4] = b(t, Point(T₁, m), (χ₁, α₁), oecdmodel, calibration)
    du[5] = b(t, Point(T₂, m), (χ₂, α₂), rowmodel, calibration)

    return
end

function Ggame!(Σ, u, parameters::GameParameters, t)
    models = first(parameters)
    oecdmodel, rowmodel = models

    Σ[1] = oecdmodel.hogg.σₜ / oecdmodel.hogg.ϵ
	Σ[2] = rowmodel.hogg.σₜ / rowmodel.hogg.ϵ
    Σ[3] = oecdmodel.hogg.σₘ
    Σ[4] = oecdmodel.economy.σₖ
    Σ[5] = rowmodel.economy.σₖ

    return
end

SimulationParameters = Tuple{AbstractModel, PoliciesFunctions, Calibration}
function Fbreakdown!(du, u, parameters::SimulationParameters, t)
    model, pols, calibration = parameters
    χitp, αitp = pols
    T, m = @view u[1:2]

    du[1] = μ(T, m, model) / model.hogg.ϵ
	du[2] = γ(t, calibration) - αitp(T, m, t)
    du[3] =  b(t, Point(T, m), (χitp(T, m, t), αitp(T, m, t)), model, calibration)

    # Cost breakdown breakdown
    damage, adjcosts, abatement = costbreakdown(t, Point(T, m), (χitp(T, m, t), αitp(T, m, t)), model, calibration)

    du[4] = adjcosts
    du[5] = abatement
    du[6] = damage
end

function F!(du, u, parameters::SimulationParameters, t)
    model, pols, calibration = parameters
    χitp, αitp = pols
    T, m = @view u[1:2]

    du[1] = μ(T, m, model) / model.hogg.ϵ
	du[2] = γ(t, calibration) - αitp(T, m, t)
    du[3] = b(t, Point(T, m), (χitp(T, m, t), αitp(T, m, t)), model, calibration)
end

function F!(du, u, parameters::SimulationParameters, t)
    model, αitp, calibration  = parameters
    T, m = u

    du[1] = μ(T, m, model) / model.hogg.ϵ
	du[2] = γ(t, calibration) - αitp(T, m, t)
end

BAUParameters = Tuple{AbstractModel, Calibration}
function Fbau!(du, u, parameters::BAUParameters, t)
    model, calibration = parameters
	du[1] = μ(u[1], u[2], model) / model.hogg.ϵ
	du[2] = γ(t, calibration)
end

BAUGameParameters = Tuple{NTuple{2, AbstractModel}, Calibration}
function Fbau!(du, u, parameters::BAUGameParameters, t)
    models, calibration = parameters
    oecdmodel, rowmodel = models
    T₁, T₂, m = @view u[1:3]

    du[1] = μ(T₁, m, oecdmodel) / oecdmodel.hogg.ϵ
    du[2] = μ(T₂, m, rowmodel) / rowmodel.hogg.ϵ
    du[3] = γ(t, calibration)
end

function G!(Σ, u, parameters::BAUParameters, t)
    model = first(parameters)
	Σ[1] = model.hogg.σₜ / model.hogg.ϵ
	Σ[2] = model.hogg.σₘ
end
function G!(Σ, u, parameters::SimulationParameters, t)
    model = first(parameters)

    Σ[1] = model.hogg.σₜ / model.hogg.ϵ
	Σ[2] = model.hogg.σₘ
    Σ[3] = model.economy.σₖ
end
function G!(Σ, u, parameters::BAUGameParameters, t)
    oecdmodel, rowmodel = first(parameters)

    Σ[1] = oecdmodel.hogg.σₜ / oecdmodel.hogg.ϵ
    Σ[2] = rowmodel.hogg.σₜ / rowmodel.hogg.ϵ
    Σ[3] = oecdmodel.hogg.σₘ
end
function Gbreakdown!(Σ, u, parameters::SimulationParameters, t)
    model = first(parameters)

    Σ[1] = model.hogg.σₜ / model.hogg.ϵ
	Σ[2] = model.hogg.σₘ
    Σ[3] = model.economy.σₖ
    Σ[4:end] .= 0.
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

Result = Tuple{Vector{Float64}, Array{Float64, 3}, Array{Float64, 4}, RegularGrid, AbstractModel}
"Constructs linear interpolation of results"
function buildinterpolations(result::Result)
    timespace, F, policy, G, _ = result

    Tspace = range(G.domains[1]...; length = size(G, 1))
    mspace = range(G.domains[2]...; length = size(G, 2))

    nodes = (Tspace, mspace, timespace)
    Fitp = linear_interpolation(nodes, F; extrapolation_bc = Line())
    χitp = linear_interpolation(nodes, policy[:, :, 1, :]; extrapolation_bc = Line())
    αitp = linear_interpolation(nodes, policy[:, :, 2, :]; extrapolation_bc = Line())

    Dict{Symbol, typeof(χitp)}(:F => Fitp, :χ => χitp, :α => αitp)
end

GameResult = Tuple{Vector{Float64}, Dict{<:AbstractModel, Array{Float64, 3}}, Dict{<:AbstractModel, Array{Float64, 4}}, RegularGrid, Vector{<:AbstractModel}}
function buildinterpolations(result::GameResult)
    timespace, F, policy, G, models = result

    Tspace = range(G.domains[1]...; length = size(G, 1))
    mspace = range(G.domains[2]...; length = size(G, 2))

    nodes = (Tspace, mspace, timespace)

    itps = Dict{
        AbstractModel, 
        Dict{Symbol, Extrapolation}
    }()

    for model in models
        Fitp = linear_interpolation(nodes, F[model]; extrapolation_bc = Line())
        χitp = linear_interpolation(nodes, policy[model][:, :, 1, :]; extrapolation_bc = Line())
        αitp = linear_interpolation(nodes, policy[model][:, :, 2, :]; extrapolation_bc = Line())

        itps[model] = Dict{Symbol, Extrapolation}(:F => Fitp, :χ => χitp, :α => αitp)
    end

    return itps
end

# Functions
computeonsim(sim::RODESolution, f, timesteps) = computeonsim!(similar(timesteps), sim, f, timesteps) 
computeonsim(sim::RODESolution, f) = computeonsim(sim, f, sim.t)
function computeonsim!(outvec, sim::RODESolution, f, timesteps)
    for i in eachindex(outvec)
        t = timesteps[i]
        simₜ = sim(t)

        xᵢ = simₜ isa DifferentialEquations.ExtendedJumpArray ? simₜ.u : simₜ

        outvec[i] = f(xᵢ..., t)
    end

    return outvec
end

function computeonsim(sol::EnsembleSolution, f, timesteps)
    N = length(sol)
    T = length(timesteps)
    M = Matrix{Float64}(undef, T, N)

    computeonsim!(M, sol, f, timesteps)
end
function computeonsim!(M, sol::EnsembleSolution, f, timesteps)
    for (m, sim) in zip(eachcol(M), sol)
        computeonsim!(m, sim, f, timesteps)
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

function smoothquantile!(v, window)
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
function sampletemperature(rng, model::AbstractModel, trajectories; σ = 0.15)
    T̄ = minimum(Model.Tstable(log(model.hogg.M₀), model))
    z = randn(rng, trajectories)

    return @. T̄ + z * σ
end