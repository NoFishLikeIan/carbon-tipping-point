using JSON

thresholds = 2:0.05:4;
discoveries = -1:0.25:1

parameters = Dict{String, Float64}[];
for discovery in discoveries, threshold in thresholds
    obj = Dict(
        "discovery" => discovery,
        "threshold" => threshold
    )

    push!(parameters, obj)
end

println("Constructed $(length(parameters)) parameters object.")

open("jobs/certaintyequivalence/parameters.json", "w") do f
    JSON.print(f, Dict("parameters" => parameters), 2)
end