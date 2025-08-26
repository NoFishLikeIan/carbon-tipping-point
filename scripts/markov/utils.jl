mutable struct Error{S}
    absolute::S
    relative::S
end

function Base.isless(error::Error{S}, tolerance::Error{S}) where S
    (error.absolute < tolerance.absolute) && (error.relative < tolerance.relative)
end

function abserror(a::AbstractArray{S}, b::AbstractArray{S}) where S
    abserror!(Error{S}(zero(S), zero(S)), a, b)
end
function abserror!(error::Error{S}, a::AbstractArray{S}, b::AbstractArray{S}) where S
    for k in eachindex(a)
        abserror!(error, a, b, k)
    end
    return error
end
function abserror!(error::Error{S}, a::AbstractArray{S}, b::AbstractArray{S}, k) where S
    Δ = abs(a[k] - b[k])
    if Δ > error.absolute
        error.absolute = Δ
        error.relative = Δ / abs(b[k])
    end

    return error
end