using JSON
include("../../scripts/utils/saving.jl")

inputdir = isempty(ARGS) ? "data/simulation-medium" : first(ARGS);
inputfiles = replace.(listfiles(inputdir), "data/" => "");

obj = Dict("inputfiles" => inputfiles);

open("jobs/experiments/parameters.json", "w") do file
    JSON.print(file, obj, 2)
end