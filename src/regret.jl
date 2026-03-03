struct SimplexPolicies{S, I <: Interpolations.Extrapolation{S}, PS <: OrderedDict{S, I}}
    policies::PS
end
function SimplexPolicies(paths; tspan = (0, Inf))
    policies = OrderedDict(Tᶜ => loadpolicy(path; tspan) for (Tᶜ, path) in paths)

    return SimplexPolicies(policies)
end
function Base.size(sp::SimplexPolicies)
    length(sp.policies)
end

function Base.show(io::IO, sp::SimplexPolicies{S, I, PS}) where {S, I, PS}
    npolicies = size(sp)
    thresholds = keys(sp.policies)
    
    Tmin = minimum(thresholds)
    Tmax = maximum(thresholds)
    print(io, "SimplexPolicies{K = $npolicies | Tᶜ ∈ {$(join(thresholds, ", "))} °C}")
end
Base.show(io::IO, ::MIME"text/plain", sp::SimplexPolicies) = show(io, sp)

Weight{S} = OrderedDict{S, S}

function weightedpolicy(x::Point{S}, t::S, weights::Weight{S}, policybasis::P) where {S, P <: SimplexPolicies{S}}
    αʷ = zero(S)

    for (Tᶜ, αfn) in policybasis.policies
        αʷ += weights[Tᶜ] * αfn(x.T, x.m, t)
    end

    return αʷ
end