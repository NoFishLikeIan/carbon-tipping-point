abstract type AbstractModel{T <: Real, D <: Damages{T}, P <: Preferences{T}} end

abstract type FirstOrderLinearModel{T, D, P} <: AbstractModel{T, D, P} end


struct LinearModel{T, D, P} <: FirstOrderLinearModel{T, D, P}
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

struct JumpModel{T, D, P} <: FirstOrderLinearModel{T, D, P}
    hogg::Hogg{T}
    preferences::P
    damages::D
    economy::Economy{T}
    jump::Jump{T}
end

Base.broadcastable(m::AbstractModel) = Ref(m)

# Extend functions to models
μ(T, m, model::TippingModel) = μ(T, m, model.hogg, model.feedback)
μ(T, m, model::FirstOrderLinearModel) = μ(T, m, model.hogg)

mstable(T, model::TippingModel) = mstable(T, model.hogg, model.feedback)
mstable(T, model::FirstOrderLinearModel) = mstable(T, model.hogg)

Tstable(m, model::TippingModel) = Tstable(m, model.hogg, model.feedback)
Tstable(m, model::FirstOrderLinearModel) = Tstable(m, model.hogg)
