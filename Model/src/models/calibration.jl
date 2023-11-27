struct Calibration
    years::Vector{Int} # Years of the ipcc data
    emissions::Vector{Float32} # Emissions in gton / year
    γparameters::NTuple{3, Float32} # Paramters for γ
end