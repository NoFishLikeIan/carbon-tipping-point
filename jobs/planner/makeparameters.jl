using JSON

rras = [10.0];
eiss = [1.];
thresholds = [2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 4.0];
damages = [:kalkuhl, :nodamages, :weitzman]
withnegatives = [false, true];

tippingparameters = Dict{String, Union{Float64, Symbol, Bool}}[];
linearparameters = Dict{String, Union{Float64, Symbol, Bool}}[];

for rra in rras, eis in eiss, damage in damages, withnegative in withnegatives
    push!(linearparameters, Dict("rra" => rra, "eis" => eis, "damage" => damage, "withnegative" => withnegative))

    for threshold in thresholds
        push!(tippingparameters, Dict("threshold" => threshold, "rra" => rra, "eis" => eis, "damage" => damage, "withnegative" => withnegative))
    end
end

obj = Dict(
    "tipping" => tippingparameters,
    "linear" => linearparameters
)

open("jobs/planner/parameters.json", "w") do f
    JSON.print(f, obj, 2)
end