module Model

# Models
export Economy, Hogg, Albedo, Calibration
export μ, b, bterminal, γ, f, mstable
export ModelInstance

# Packages
using Grid: Point, Policy, Drift
using UnPack: @unpack

include("models/calibration.jl")
include("models/climate.jl")
include("models/economy.jl")

@kwdef struct ModelInstance
    economy::Economy = Economy()
    hogg::Hogg = Hogg()
    albedo::Albedo = Albedo()
    calibration::Calibration
end

include("models/functions.jl")

end # module Model
