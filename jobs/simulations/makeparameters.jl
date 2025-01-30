using JSON

rras = [2.0, 10.0];
eiss = [0.75, 1.5];
thresholds = [1.5, 2.5, 3.5];
allownegatives = [false];
withleveldamages = [false];

tippingparameters = Dict{String, Union{Float64, Bool}}[];
benchmarkparameters = Dict{String, Union{Float64, Bool}}[];


for rra in rras, eis in eiss, negative in allownegatives, leveldamages in withleveldamages
    push!(benchmarkparameters, Dict("rra" => rra, "eis" => eis, "allownegative" => negative, "leveldamages" => leveldamages))

    for threshold in thresholds
        push!(tippingparameters, Dict("threshold" => threshold, "rra" => rra, "eis" => eis, "allownegative" => negative, "leveldamages" => leveldamages))
    end
end

obj = Dict(
    "benchmarkparameters" => benchmarkparameters,
    "tippingparameters" => tippingparameters
)

open("jobs/simulations/parameters.json", "w") do f
    JSON.print(f, obj, 2)
end