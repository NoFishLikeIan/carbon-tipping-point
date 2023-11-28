@kwdef struct ModelInstance{N}
    economy::Economy = Economy()
    hogg::Hogg = Hogg()
    albedo::Albedo = Albedo()
    grid::RegularGrid{N}
    calibration::Calibration
end
