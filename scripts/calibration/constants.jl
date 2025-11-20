# Molecular weights (g/mol)
molweights = Dict(
    "carbon_dioxide" => 44.01,
    "methane" => 16.04,
    "nitrous_oxide" => 44.01, 
    "nf3" => 71.00,      
    "sf6" => 146.06
)

# For some reason the FAIR model does not ouput molecules but elangated names.
moleculetoname = Dict(
    "co2" => "carbon_dioxide",
    "ch4" => "methane",
    "n2o" => "nitrous_oxide",
    "nf3" => "nf3",
    "sf6" => "sf6"
)

# Mass-to-concentration conversion factors (accounts for molecular weight differences)
masstoconcentration = Dict(
    particle => Model.Gtonoverppm * molweights["carbon_dioxide"] / w
    for (particle, w) in molweights
)

# Unit conversions
converter = Dict(
    "carbon_dioxide" => 1.0,
    "methane" => 1e-3,      # ppb to ppm
    "nitrous_oxide" => 1e-3, # ppb to ppm  
    "nf3" => 1e-6,          # ppt to ppm
    "sf6" => 1e-6           # ppt to ppm
)

# GWP values (AR6, 100-year)
gwpvalues = Dict(
    "carbon_dioxide" => 1.0,
    "methane" => 29.8,
    "nitrous_oxide" => 273.0,
    "nf3" => 17_400.0,
    "sf6" => 24_300.0
)