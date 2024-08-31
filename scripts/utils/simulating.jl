using Model, Grid
using Dierckx: Spline2D
using Interpolations
using DifferentialEquations
using Statistics

function Fopt!(du, u, model::Tuple{AbstractModel, Extrapolation}, t)
    model, αitp = model
    T, m = u

    du[1] = μ(T, m, model) / model.hogg.ϵ
	du[2] = γ(t, model.calibration) - αitp(T, m, t)
end
function Fbau!(du, u, model::AbstractModel, t)
	du[1] = μ(u[1], u[2], model) / model.hogg.ϵ
	du[2] = γ(t, model.calibration)
end

G!(du, u, parameters::Tuple, t) = G!(du, u, first(parameters), t) # Assumes parameters = (model, ...)
function G!(Σ, u, model::AbstractModel, t)    
	Σ[1] = model.hogg.σₜ / model.hogg.ϵ
	Σ[2] = model.hogg.σₘ
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

Result = Tuple{Vector{Float64}, Array{Float64, 3}, Array{Float64, 4}, RegularGrid}
"Constructs spline of results"
function buildsplines(result::Result; splinekwargs...)
    timespace, F, policy, G = result

    Tspace = range(G.domains[1]...; length = size(G, 1))
    mspace = range(G.domains[2]...; length = size(G, 2))

    timesplines = Dict{Float64, NTuple{3, Spline2D}}()

    for (k, t) in enumerate(timespace)
        pol = policy[:, :, k]

        timesplines[t] = map(
            M -> Spline2D(Tspace, mspace, M; splinekwargs...),
            (F[:, :, k], first.(pol), last.(pol))
        )
    end

    return timesplines
end

ResultInterpolation = Dict{Symbol, Extrapolation}
"Constructs linear interpolation of results"
function buildinterpolations(result::Result; splinekwargs...)
    timespace, F, policy, G = result

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
        T, m = sim(t)

        outvec[i] = f(T, m, t)
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