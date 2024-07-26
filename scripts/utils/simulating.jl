using Model, Grid
using Interpolations

function Fbau!(du, u, model::AbstractModel, t)
	du[1] = μ(u[1], u[2], model) / model.hogg.ϵ
	du[2] = γ(t, model.calibration)
end
function Gbau!(du, u, model, t)    
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
buildinterpolations(result::Result, G) = buildinterpolations([result], [G]) |> first
function buildinterpolations(results::AbstractVector{Result}, Gs)
    itps = NTuple{3, Interpolations.GriddedInterpolation}[]

    for (k, result) in enumerate(results)
        G = Gs[k]
        timespace, F, policy = result

        Tspace = range(G.domains[1]...; length = size(G, 1))
        mspace = range(G.domains[2]...; length = size(G, 2))

        nodes = (Tspace, mspace, timespace)

        Fitp = interpolate(nodes, F, Gridded(Linear()))
        χitp = interpolate(nodes, first.(policy), Gridded(Linear()))
        αitp = interpolate(nodes, last.(policy), Gridded(Linear()))

        push!(itps, (Fitp, χitp, αitp))
    end

    return itps
end
