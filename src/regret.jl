struct SimplexPolicies{K, S, I <: Interpolations.Extrapolation{S}, PS <: NTuple{K, I}, TS <: StaticVector{K, S}}
    thresholds::TS
    policies::PS
end
function SimplexPolicies(pathdict::OrderedDict; tspan = (0, Inf))
    K = length(pathdict)
    thresholds = SVector{K}(collect(keys(pathdict)))

    paths = collect(pathdict.vals)
    policies = ntuple(i -> loadpolicy(paths[i]; tspan), K);

    return SimplexPolicies(thresholds, policies)
end

Base.show(io::IO, sp::SimplexPolicies{K, S}) where {K, S} = print(io, "SimplexPolicies{K = $K | Tᶜ ∈ {$(join(sp.thresholds, ", "))} °C}")
Base.show(io::IO, ::MIME"text/plain", sp::SimplexPolicies) = show(io, sp)

function weightedpolicy(x::Point{S}, t::S, weights::W, policybasis::P) where {K, S, W <: StaticVector{K}, P <: SimplexPolicies{K, S}}
    αʷ = zero(S)

    @inbounds for i in 1:K
        αʷ += weights[i] * policybasis.policies[i](x.T, x.m, t)
    end

    return αʷ
end