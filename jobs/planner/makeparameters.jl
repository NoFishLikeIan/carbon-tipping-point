using JSON

thresholds = [2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 4.0, -1.];
damages = ["kalkuhl", "nodamages", "weitzman"]
withnegatives = [false, true];

parameters = Dict{String, Union{Float64, String, Bool}}[];

for damage in damages, withnegative in withnegatives, threshold in thresholds
    obj = Dict(
        "damage" => damage, 
        "withnegative" => withnegative,
        "threshold" => threshold
    )

    push!(parameters, obj)
end

println("Constructed $(length(parameters)) parameters object.")

open("jobs/planner/parameters.json", "w") do f
    JSON.print(f, Dict("parameters" => parameters), 2)
end