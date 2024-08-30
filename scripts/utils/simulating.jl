using Model, Grid
using Dierckx: Spline2D
using Interpolations: linear_interpolation, Line, Extrapolation

function Fbau!(du, u, model::AbstractModel, t)
	du[1] = μ(u[1], u[2], model) / model.hogg.ϵ
	du[2] = γ(t, model.calibration)
end
function G!(du, u, model::AbstractModel, t)    
	du[1] = model.hogg.σₜ / model.hogg.ϵ
	du[2] = model.hogg.σₘ
end

function rate(u, model::JumpModel, t)
    intensity(u[1], model.hogg, model.jump)
end
function affect!(integrator)
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
