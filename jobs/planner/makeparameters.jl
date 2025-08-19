using JSON

thresholds = [2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 5.0, -1.];
damages = ["kalkuhl", "weitzman"]
withnegatives = [false, true];

ParamVal = Union{Float64, String, Bool};
parameters = Dict{String, ParamVal}[];
for damage in damages, withnegative in withnegatives, threshold in thresholds
    obj = Dict(
        "damage" => damage, 
        "withnegative" => withnegative,
        "threshold" => threshold
    )

    push!(parameters, obj)
end

sanitycheck = Dict{String, ParamVal}("damage" => "nodamages", "withnegative" => true, "threshold" => -1.); push!(parameters, sanitycheck)

println("Constructed $(length(parameters)) parameters object.")

open("jobs/planner/parameters.json", "w") do f
    JSON.print(f, Dict("parameters" => parameters), 2)
end