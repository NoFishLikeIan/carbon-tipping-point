struct IAM{S, D <: Damages{S}, P <: Preferences{S}, C <: Climate{S}}
    climate::C
    economy::Economy{S, D}
    preferences::P
end

UnitIAM{S} = IAM{S, D, P, C} where { D <: Damages{S}, P <: LogSeparable{S}, C <: Climate{S}}

Base.broadcastable(m::IAM) = Ref(m)
