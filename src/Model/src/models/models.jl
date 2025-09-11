struct IAM{S, D <: Damages{S}, P <: Preferences{S}, C <: Climate{S}}
    climate::C
    economy::Economy{S, D}
    preferences::P
end

UnitIAM{S} = IAM{S, D, P, C} where { D <: Damages{S}, P <: LogSeparable{S}, C <: Climate{S}}

Base.broadcastable(m::IAM) = Ref(m)

"Construct an equivalent IAM with no noise. Used for the terminal problem."
function determinsticIAM(model::IAM{S}) where S
    IAM(deterministicClimate(model.climate), model.economy, model.preferences)
end
