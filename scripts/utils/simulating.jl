using Model, Grid
using Dierckx: Spline2D
using Interpolations
using FiniteDiff
using DifferentialEquations
using Statistics
using Random: default_rng

PolicyFunction = Union{Interpolations.Extrapolation, Function};
PoliciesFunctions = NTuple{2, PolicyFunction};

function Fbreakdown!(du, u, parameters::Tuple{AbstractModel, PoliciesFunctions}, t)
    model, pols = parameters
    χitp, αitp = pols
    T, m = @view u[1:2]

    du[1] = μ(T, m, model) / model.hogg.ϵ
	du[2] = γ(t, model.calibration) - αitp(T, m, t)
    du[3] =  b(t, Point(T, m), (χitp(T, m, t), αitp(T, m, t)), model)

    # Cost breakdown breakdown
    damage, adjcosts, abatement = costbreakdown(t, Point(T, m), (χitp(T, m, t), αitp(T, m, t)), model)

    du[4] = adjcosts
    du[5] = abatement
    du[6] = damage
end

function F!(du, u, parameters::Tuple{AbstractModel, PoliciesFunctions}, t)
    model, pols = parameters
    χitp, αitp = pols
    T, m = @view u[1:2]

    du[1] = μ(T, m, model) / model.hogg.ϵ
	du[2] = γ(t, model.calibration) - αitp(T, m, t)
    du[3] = b(t, Point(T, m), (χitp(T, m, t), αitp(T, m, t)), model)
end
function F!(du, u, parameters::Tuple{AbstractModel, PolicyFunction}, t)
    model, αitp = parameters
    T, m = u

    du[1] = μ(T, m, model) / model.hogg.ϵ
	du[2] = γ(t, model.calibration) - αitp(T, m, t)
end
function Fbau!(du, u, model::AbstractModel, t)
	du[1] = μ(u[1], u[2], model) / model.hogg.ϵ
	du[2] = γ(t, model.calibration)
end

function G!(Σ, u, model::AbstractModel, t)    
	Σ[1] = model.hogg.σₜ / model.hogg.ϵ
	Σ[2] = model.hogg.σₘ
end
function G!(Σ, u, parameters::Tuple{AbstractPlannerModel, PoliciesFunctions}, t)
    model = first(parameters)

    Σ[1] = model.hogg.σₜ / model.hogg.ϵ
	Σ[2] = model.hogg.σₘ
    Σ[3] = model.economy.σₖ
end
function Gbreakdown!(Σ, u, parameters::Tuple{AbstractPlannerModel, PoliciesFunctions}, t)
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
"Constructs spline of results"
function buildsplines(result::Result; splinekwargs...)
    timespace, F, policy, G, _ = result

    Tspace = range(G.domains[1]...; length = size(G, 1))
    mspace = range(G.domains[2]...; length = size(G, 2))

    timesplines = Dict{Symbol, Vector{Spline2D}}()

    αvec = Spline2D[]
    χvec = Spline2D[]
    Fvec = Spline2D[]

    for (k, t) in enumerate(timespace)
        χₜ = policy[:, :, 1, k]
        αₜ = policy[:, :, 2, k]
        Fₜ = F[:, :, k]

        spl = Spline2D(Tspace, mspace, χₜ; s = 0.25)
    end

    return timesplines
end

ResultInterpolation = Dict{Symbol, Interpolations.Extrapolation}
"Constructs linear interpolation of results"
function buildinterpolations(result::Result; splinekwargs...)
    timespace, F, policy, G, _ = result

    Tspace = range(G.domains[1]...; length = size(G, 1))
    mspace = range(G.domains[2]...; length = size(G, 2))

    nodes = (Tspace, mspace, timespace)
    Fitp = linear_interpolation(nodes, F; extrapolation_bc = Line())
    χitp = linear_interpolation(nodes, policy[:, :, 1, :]; extrapolation_bc = Line())
    αitp = linear_interpolation(nodes, policy[:, :, 2, :]; extrapolation_bc = Line())

    Dict{Symbol, typeof(χitp)}(:F => Fitp, :χ => χitp, :α => αitp)
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