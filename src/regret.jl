struct SimplexPolicies{S, I <: Interpolations.Extrapolation{S}, PS <: OrderedDict{S, I}}
    policies::PS
end
function SimplexPolicies(paths; tspan = (0, Inf))
    policies = OrderedDict(Tᶜ => loadpolicy(path; tspan) for (Tᶜ, path) in paths)

    return SimplexPolicies(policies)
end
function Base.show(io::IO, sp::SimplexPolicies{S, I, PS}) where {S, I, PS}
    npolicies = length(sp.policies)
    thresholds = collect(keys(sp.policies))
    
    Tmin = minimum(thresholds)
    Tmax = maximum(thresholds)
    print(io, "SimplexPolicies{K = $npolicies, Tᶜ ∈ [$(round(Tmin, digits=2)), $(round(Tmax, digits=2))] °C}")
end
Base.show(io::IO, ::MIME"text/plain", sp::SimplexPolicies) = show(io, sp)

struct ConvexPolicies{S, P <: SimplexPolicies{S}, W <: OrderedDict{S, S}}
    policybasis::P
    weights::W
end

function weightedpolicy(x, t, policybasis::ConvexPolicies)
    weightedpolicy(x, t, policybasis.weights, policybasis.policies)
end
function weightedpolicy(x::Point{S}, t::S, weights, policybasis::P) where {S, P <: SimplexPolicies{S}}
    αʷ = zero(S)

    for (Tᶜ, αfn) in policybasis.policies
        αʷ += weights[Tᶜ] * αfn(x.T, x.m, t)
    end

    return αʷ
end