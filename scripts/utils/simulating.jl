using Model, Grid
using Dierckx

function Fbau!(du, u, model::AbstractModel, t)
	du[1] = μ(u[1], u[2], model) / model.hogg.ϵ
	du[2] = γ(t, model.calibration)
end
function G!(du, u, model, t)    
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

Result = Tuple{Vector{Float64}, Array{Float64, 3}, Array{Policy, 3}}

"Constructs linear interpolation of results"
buildinterpolations(result::Result, G; splinekwargs...) = buildinterpolations([result], [G]; splinekwargs...) |> first
function buildinterpolations(results::AbstractVector{Result}, Gs; splinekwargs...)
    itps = Dict{Float64, NTuple{3, Spline2D}}[]

    for (k, result) in enumerate(results)
        G = Gs[k]
        timespace, F, policy = result

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

        push!(itps, timesplines)
    end

    return itps
end
