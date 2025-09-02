Domain{T} = NTuple{2,T}

struct Point{S<:Real} <: FieldVector{2, S}
    T::S
    m::S
end

struct Policy{S<:Real} <: FieldVector{2, S}
    χ::S
    α::S
end

# Extend static array
StaticArrays.similar_type(::Type{<:Point}, ::Type{S}, s::Size{(2,)}) where S = Point{S}
Base.similar(::Type{<:Point}, ::Type{S}) where S = Point(zero(S), zero(S))
StaticArrays.similar_type(::Type{<:Policy}, ::Type{S}, s::Size{(2,)}) where S = Policy{S}
Base.similar(::Type{<:Policy}, ::Type{S}) where S = Policy(zero(S), zero(S))