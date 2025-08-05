abstract type AbstractModel{T <: Real, D <: Damages{T}, P <: Preferences{T}} end

struct LinearModel{T, D, P} <: AbstractModel{T, D, P}
    hogg::Hogg{T}
    preferences::P
    damages::D
    economy::Economy{T}
end

struct TippingModel{T, D, P} <: AbstractModel{T, D, P}
    hogg::Hogg{T}
    preferences::P
    damages::D
    economy::Economy{T}
    feedback::Feedback{T}
end

struct JumpModel{T, D, P} <: AbstractModel{T, D, P}
    hogg::Hogg{T}
    preferences::P
    damages::D
    economy::Economy{T}
    jump::Jump{T}
end

Base.broadcastable(m::AbstractModel) = Ref(m)