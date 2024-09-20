using JSON

rras = [2.0, 10.0];
eiss = [0.75, 1.5];
thresholds = [1.5, 2.5];
allownegatives = [true, false];
withleveldamages = [true, false];

parameters = Dict{String, Union{Float64, Bool}}[]

for threshold in thresholds, rra in rras, eis in eiss, negative in allownegatives, leveldamages in withleveldamages
    push!(parameters, Dict("threshold" => threshold, "rra" => rra, "eis" => eis, "negative" => negative, "leveldamages" => leveldamages))
end

open("jobs/parameters.json", "w") do f
    JSON.print(f, Dict("parameters" => parameters), 2)
end