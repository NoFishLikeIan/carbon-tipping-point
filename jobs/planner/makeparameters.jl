using JSON

rras = [10.0];
eiss = [1.];
thresholds = [2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 4.0];
withleveldamages = [false];
withnegatives = [false, true];

tippingparameters = Dict{String, Union{Float64, Bool}}[];
benchmarkparameters = Dict{String, Union{Float64, Bool}}[];

for rra in rras, eis in eiss, leveldamages in withleveldamages, withnegative in withnegatives
    push!(benchmarkparameters, Dict("rra" => rra, "eis" => eis, "leveldamages" => leveldamages, "withnegative" => withnegative))

    for threshold in thresholds
        push!(tippingparameters, Dict("threshold" => threshold, "rra" => rra, "eis" => eis, "leveldamages" => leveldamages, "withnegative" => withnegative))
    end
end

obj = Dict(
    "benchmarkparameters" => benchmarkparameters,
    "tippingparameters" => tippingparameters
)

open("jobs/planner/parameters.json", "w") do f
    JSON.print(f, obj, 2)
end