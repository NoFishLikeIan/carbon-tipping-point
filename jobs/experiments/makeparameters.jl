using JSON
include("../../scripts/utils/saving.jl")

inputfiles = listfiles("data/simulation-medium");
obj = Dict("inputfiles" => inputfiles);

open("jobs/experiments/parameters.json", "w") do f
    JSON.print(f, obj, 2)
end